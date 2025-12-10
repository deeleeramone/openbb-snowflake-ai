"""Slash command handlers for Snowflake AI."""

import asyncio
import json
import os

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

from . import SnowflakeAI, SnowflakeAgent
from .helpers import clear_message_signatures, get_row_value, to_sse
from .logger import get_logger


logger = get_logger(__name__)

# Session expiration error codes from Snowflake
SESSION_EXPIRED_CODES = {"390112", "390114", "390111"}


def is_session_expired_error(error: Exception) -> bool:
    """Check if an error indicates Snowflake session expiration."""
    error_str = str(error)
    return any(code in error_str for code in SESSION_EXPIRED_CODES)


async def run_in_thread(func, *args, **kwargs):
    """Run a function in a background thread."""
    return await asyncio.to_thread(func, *args, **kwargs)


def find_snow_cli_binary():
    """Find the snow CLI binary."""
    snow_path = shutil.which("snow")
    if snow_path:
        return snow_path
    return None


async def handle_slash_command(
    user_command: str,
    conv_id: str,
    client: SnowflakeAI,
    agent: SnowflakeAgent,
    selected_model: str,
    selected_temperature: float,
    selected_max_tokens: int,
    model_preferences: dict,
    temperature_preferences: dict,
    max_tokens_preferences: dict,
    token_usage: dict,
) -> AsyncIterator[dict[str, Any]]:
    """
    Handle slash commands and yield SSE events.

    Returns an async generator that yields SSE-formatted dicts.
    If the command is not recognized, yields an error message.
    """
    from openbb_ai import message_chunk, reasoning_step

    # Store current working context
    current_working_db = await run_in_thread(client.get_current_database)
    current_working_schema = await run_in_thread(client.get_current_schema)

    if user_command == "/help":
        help_text = """**Available Commands:**

📁 **File Management**
- `/upload <file_path> [--embed-images]` - Upload a file to Snowflake stage
- `/parse <stage_path> [--embed-images]` - Parse a document from stage
- `/stages` - List all available stages
- `/stage_files <stage>` - List files in a specific stage

*Note: Use `--embed-images` flag to include image embeddings during parsing*

🔧 **Model Settings**
- `/models` - List available AI models
- `/model <name>` - Set the active model
- `/temperature <value>` - Set temperature (0.0-1.0)
- `/max_tokens <value>` - Set max tokens
- `/current` - Show current settings

🗄️ **Database Navigation**
- `/databases` - List all databases
- `/schemas [database]` - List schemas
- `/tables` - List tables in current schema
- `/warehouses` - List available warehouses
- `/use_database <name>` - Switch database
- `/use_schema <name>` - Switch schema
- `/use_warehouse <name>` - Switch warehouse

💬 **Conversation**
- `/clear` - Clear conversation history
- `/history` - Show conversation history
- `/context` - Show database context
- `/suggest <prompt>` - Generate SQL (text2sql) without executing it

Type any command to get started!"""
        yield to_sse(message_chunk(help_text))

    elif user_command.startswith("/upload "):
        upload_args = user_command[8:].strip()

        # Parse --embed-images flag
        embed_images = False
        if "--embed-images" in upload_args:
            embed_images = True
            upload_args = upload_args.replace("--embed-images", "").strip()

        file_path_str = upload_args
        yield to_sse(
            reasoning_step(
                f"Processing '{file_path_str}'"
                + (" with image embedding" if embed_images else "")
                + "..."
            )
        )

        # Expand user path and check if file exists
        file_path = Path(file_path_str).expanduser().resolve()

        if not file_path.exists():
            yield to_sse(message_chunk(f"❌ File not found: {file_path}"))
            return

        if not file_path.is_file():
            yield to_sse(message_chunk(f"❌ Path is not a file: {file_path}"))
            return

        # Check file size
        try:
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            yield to_sse(
                message_chunk(
                    f"📁 File: {file_path.name}\n" f"📊 Size: {file_size_mb:.1f} MB"
                )
            )
        except Exception as e:
            yield to_sse(message_chunk(f"⚠️ Cannot read file size: {e}"))

        stage_name = "CORTEX_UPLOADS"

        # Use the user-specific database and schema
        db_name = "OPENBB_AGENTS"
        snowflake_user = await run_in_thread(client.get_current_user)
        sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
        schema_name = f"USER_{sanitized_user}".upper()

        # First ensure the database exists
        yield to_sse(reasoning_step(f"Ensuring database {db_name} exists..."))
        try:
            await run_in_thread(
                client.execute_statement, f"CREATE DATABASE IF NOT EXISTS {db_name}"
            )
        except Exception as e:
            yield to_sse(message_chunk(f"⚠️ Could not create database: {e}"))

        # Then ensure the schema exists
        yield to_sse(
            reasoning_step(f"Ensuring schema {db_name}.{schema_name} exists...")
        )
        try:
            await run_in_thread(
                client.execute_statement,
                f"CREATE SCHEMA IF NOT EXISTS {db_name}.{schema_name}",
            )
        except Exception as e:
            yield to_sse(message_chunk(f"⚠️ Could not create schema: {e}"))

        qualified_stage_name = f'"{db_name}"."{schema_name}"."{stage_name}"'

        # Now ensure the stage exists with directory table enabled
        yield to_sse(
            reasoning_step(
                f"Ensuring stage {qualified_stage_name} exists and refreshing directory..."
            )
        )
        try:
            create_stage_query = f"""
            CREATE STAGE IF NOT EXISTS {qualified_stage_name}
            DIRECTORY = (ENABLE = TRUE)
            ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
            """
            await run_in_thread(client.execute_statement, create_stage_query)

            # Refresh directory table
            refresh_query = f"ALTER STAGE {qualified_stage_name} REFRESH"
            await run_in_thread(client.execute_statement, refresh_query)
        except Exception as e:
            # Stage might already exist, which is fine, but we still try to refresh.
            yield to_sse(
                message_chunk(
                    f"⚠️ Could not ensure stage {qualified_stage_name} or refresh directory: {str(e)}"
                )
            )

        # Find snow CLI binary
        snow_cli_binary = find_snow_cli_binary()

        if not snow_cli_binary:
            yield to_sse(
                message_chunk(
                    "❌ **Snowflake CLI not found.**\n\n"
                    "The `snow` command is required for file uploads. "
                    "Please install it using pip:\n\n"
                    "```bash\n"
                    "pip install snowflake-cli-python\n"
                    "```\n\n"
                    "Then, configure your connection:\n"
                    "```bash\n"
                    "snow connection add\n"
                    "```"
                )
            )
            return

        # Use snow if found
        yield to_sse(message_chunk(f"✅ Found Snowflake CLI at: {snow_cli_binary}"))

        # COPY FILE TO TEMP DIRECTORY TO AVOID MACOS PERMISSION ISSUES
        yield to_sse(reasoning_step(f"Preparing file for upload..."))

        temp_dir = None
        temp_file_path = None
        upload_successful = False
        # Stage path WITHOUT quotes for storage/display - format: @DB.SCHEMA.STAGE/filename
        stage_path = f"@{db_name}.{schema_name}.{stage_name}/{file_path.name}"

        try:
            # Create temp directory
            temp_dir = tempfile.mkdtemp(prefix="snowflake_upload_")
            temp_file_path = Path(temp_dir) / file_path.name

            # READ THE FILE AND WRITE TO TEMP - DON'T USE shutil.copy2!
            try:
                with open(file_path, "rb") as source_file:
                    file_contents = source_file.read()

                with open(temp_file_path, "wb") as dest_file:
                    dest_file.write(file_contents)

                yield to_sse(message_chunk(f"✅ File prepared for upload"))
            except PermissionError:
                # If we can't read the file, we need to tell user to move it
                yield to_sse(
                    message_chunk(
                        f"❌ **macOS blocked file access**\n\n"
                        f"Cannot read file from Documents folder due to macOS security.\n\n"
                        f"**Solution - Copy file to Desktop first:**\n"
                        f"```bash\n"
                        f'cp "{file_path}" ~/Desktop/\n'
                        f"```\n\n"
                        f"Then upload from Desktop:\n"
                        f"```\n"
                        f"/upload ~/Desktop/{file_path.name}\n"
                        f"```"
                    )
                )
                return
            except Exception as e:
                yield to_sse(message_chunk(f"❌ Failed to read file: {str(e)}"))
                return

            # The snow CLI should pick up connection details from env vars
            # like SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD etc.
            env = os.environ.copy()

            # Ensure current database and schema are used
            env["SNOWFLAKE_DATABASE"] = db_name
            env["SNOWFLAKE_SCHEMA"] = schema_name

            # Build snow command
            # Note: Database and schema are passed via environment variables
            # Normalize path separators for Windows to forward slashes as required by PUT command
            temp_file_str = str(temp_file_path).replace("\\", "/")
            # Stage path for PUT command - NO QUOTES around segments: @DB.SCHEMA.STAGE
            unquoted_stage = f"{db_name}.{schema_name}.{stage_name}"
            put_query = f"PUT 'file://{temp_file_str}' @{stage_name} AUTO_COMPRESS = FALSE OVERWRITE = TRUE"

            snow_cmd = [
                snow_cli_binary,
                "sql",
                "-q",
                put_query,
                "--database",
                db_name,
                "--schema",
                schema_name,
            ]

            yield to_sse(reasoning_step("Auto-compress disabled (enforced)."))

            # Now actually upload the file
            yield to_sse(
                reasoning_step(f"Uploading file to Snowflake stage '{stage_name}'...")
            )

            # Execute snow command
            result = subprocess.run(
                snow_cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
                check=False,
            )

            # Check for success
            stdout_lower = (result.stdout or "").lower()
            stderr_lower = (result.stderr or "").lower()

            if result.returncode == 0 and "error" not in stderr_lower:
                upload_successful = True
                yield to_sse(
                    message_chunk(
                        f"✅ **File uploaded successfully!**\n\n"
                        f"📍 Location: `{stage_path}`"
                    )
                )

                # Store the stage path for this conversation
                await run_in_thread(
                    client.set_conversation_data,
                    conv_id,
                    "last_uploaded_file",
                    stage_path,
                )
            else:
                # Upload failed - show output for debugging
                yield to_sse(
                    message_chunk(
                        f"❌ **Upload failed**\n\n"
                        f"Output:\n```\n{result.stdout[:1000]}\n```\n\n"
                        f"Error:\n```\n{result.stderr[:1000]}\n```"
                    )
                )
                return

        except subprocess.TimeoutExpired:
            yield to_sse(
                message_chunk(
                    "❌ **Upload timed out**\n\n"
                    "The file upload is taking too long. This might happen with very large files.\n"
                    "Try uploading a smaller file or use the Snowflake Web UI."
                )
            )
            return
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Unexpected error: {str(e)}"))
            return
        finally:
            # Clean up temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass  # Ignore cleanup errors

        # If upload was successful, verify and optionally parse
        if upload_successful:
            yield to_sse(reasoning_step("Verifying upload..."))

            # Refresh directory table to ensure file is visible
            try:
                refresh_query = f"ALTER STAGE {qualified_stage_name} REFRESH"
                await run_in_thread(client.execute_statement, refresh_query)
            except Exception as refresh_err:
                logger.debug("Stage refresh warning: %s", refresh_err)
                # Continue anyway - refresh failure is not critical

            # Wait a moment for the file to appear in the stage
            await asyncio.sleep(2)

            try:
                # Use raw SQL to verify with qualified stage name
                list_query = f"LIST {stage_path}"
                result_json = await run_in_thread(client.execute_statement, list_query)
                rows = json.loads(result_json) if result_json else []

                file_found = False
                if rows:
                    for row in rows:
                        # Robust check: search for filename in ANY column value (usually 'name')
                        # This handles variations in column casing or unexpected schema
                        for val in row.values():
                            if isinstance(val, str) and file_path.name in val:
                                file_found = True
                                break
                        if file_found:
                            break

                if file_found:
                    yield to_sse(message_chunk("✅ File verified in stage"))

                    parseable_extensions = [
                        ".pdf",
                        ".txt",
                        ".docx",
                        ".doc",
                        ".rtf",
                        ".md",
                    ]
                    data_extensions = [".json", ".xml"]
                    file_extension = file_path.suffix.lower()

                    if file_extension in parseable_extensions:
                        yield to_sse(
                            message_chunk(
                                f"✅ **Upload complete.**\n\n"
                                f"📄 Received request to process document `{file_path.name}`.\n"
                                f"Parsing will continue in the background and results will be saved to `DOCUMENT_PARSE_RESULTS`."
                            )
                        )

                        # Fire and forget background task - use the KNOWN db_name and schema_name
                        asyncio.create_task(
                            process_document_background(
                                client,
                                stage_path,
                                file_path.name,
                                conv_id,
                                db_name,
                                schema_name,
                                embed_images=embed_images,
                            )
                        )
                    elif file_extension in data_extensions:
                        yield to_sse(
                            reasoning_step(
                                f"Processing {file_extension} as semi-structured data..."
                            )
                        )

                        table_name_raw = file_path.stem
                        table_name = "".join(
                            c if c.isalnum() else "_" for c in table_name_raw
                        ).upper()
                        file_type = "JSON" if file_extension == ".json" else "XML"

                        # Create table in the current working database/schema context
                        qualified_table_name = f'"{current_working_db}"."{current_working_schema}"."{table_name}"'

                        try:
                            # 1. Create file format in the working context
                            file_format_name = f"FORMAT_{table_name}"
                            format_qualified = f'"{current_working_db}"."{current_working_schema}"."{file_format_name}"'

                            if file_type == "XML":
                                create_format_query = f"CREATE OR REPLACE FILE FORMAT {format_qualified} TYPE = {file_type} STRIP_OUTER_ELEMENT = TRUE"
                            else:
                                create_format_query = f"CREATE OR REPLACE FILE FORMAT {format_qualified} TYPE = {file_type} STRIP_OUTER_ARRAY = TRUE"

                            await run_in_thread(
                                client.execute_statement, create_format_query
                            )
                            yield to_sse(
                                reasoning_step(
                                    f"✅ Ensured file format '{file_format_name}' exists."
                                )
                            )

                            # 2. Create table in working context
                            create_table_query = f"CREATE OR REPLACE TABLE {qualified_table_name} (RAW_DATA VARIANT)"
                            await run_in_thread(
                                client.execute_statement, create_table_query
                            )
                            yield to_sse(
                                reasoning_step(
                                    f"✅ Ensured table '{table_name}' exists in {current_working_db}.{current_working_schema}."
                                )
                            )

                            # 3. Copy into table from the stage in OPENBB_AGENTS
                            copy_into_query = f"COPY INTO {qualified_table_name} FROM '{stage_path}' FILE_FORMAT = (FORMAT_NAME = '{format_qualified}') ON_ERROR = 'CONTINUE'"
                            await run_in_thread(
                                client.execute_statement, copy_into_query
                            )
                            yield to_sse(
                                reasoning_step(f"✅ Loading data into '{table_name}'.")
                            )

                            # 4. Check if data was loaded
                            count_query = (
                                f"SELECT COUNT(*) as c FROM {qualified_table_name}"
                            )
                            count_result_str = await run_in_thread(
                                client.execute_query, count_query
                            )
                            count_result = json.loads(count_result_str)

                            num_rows = 0
                            if (
                                count_result.get("rowData")
                                and len(count_result["rowData"]) > 0
                            ):
                                row = count_result["rowData"][0]
                                # The column name from COUNT(*) can be "C" or "COUNT(*)"
                                num_rows = get_row_value(
                                    row, "C", "COUNT(*)", default=0
                                )

                            if num_rows > 0:
                                yield to_sse(
                                    message_chunk(
                                        f"✅ **Data loaded into `{table_name}`!**\n\n"
                                        f"Created table `{qualified_table_name}` with {num_rows} rows from your file.\n\n"
                                        f"You can now query this semi-structured data. For example:\n"
                                        f"```sql\n"
                                        f"SELECT * FROM {qualified_table_name} LIMIT 10;\n"
                                        f"```"
                                    )
                                )
                            else:
                                yield to_sse(
                                    message_chunk(
                                        f"⚠️ **Data loaded, but table is empty.**\n\n"
                                        f"The file from `{stage_path}` was processed, but resulted in 0 rows in the `{table_name}` table. "
                                        f"This can happen if the file format doesn't match the content (e.g., a file of JSON objects vs. an array of objects). "
                                        f"Check the file structure and try loading manually if needed."
                                    )
                                )

                        except Exception as e:
                            yield to_sse(
                                message_chunk(
                                    f"⚠️ **Could not auto-process data file:** {str(e)}\n\n"
                                    f"The file is available at `{stage_path}`. "
                                    f"You can try to create a table and load data manually."
                                )
                            )
                    else:
                        yield to_sse(
                            message_chunk(
                                f"ℹ️ File type '{file_extension}' is not automatically processed.\n\n"
                                f"Automatic parsing is available for documents: `pdf, txt, docx, doc, rtf, md`\n"
                                f"Automatic data loading is available for: `json, xml`"
                            )
                        )
                else:
                    yield to_sse(
                        message_chunk(
                            f"✅ Upload completed, but file not immediately visible in stage.\n\n"
                            f"File location: `{stage_path}`\n\n"
                            f"The file may take a moment to appear. You can try to process it with:\n"
                            f"```\n/parse {stage_path}\n```"
                        )
                    )
            except Exception as e:
                # Don't fail, just inform that verification had issues
                yield to_sse(
                    message_chunk(
                        f"✅ Upload completed!\n\n"
                        f"File location: `{stage_path}`\n\n"
                        f"Verification had issues but the file should be available.\n"
                        f"Try: `/parse {stage_path}`"
                    )
                )
        return

    elif user_command.startswith("/parse "):
        stage_path = user_command[7:].strip()
        yield to_sse(reasoning_step(f"Requesting parsing for '{stage_path}'..."))

        if not stage_path.startswith("@"):
            yield to_sse(message_chunk("❌ Invalid stage path. Must start with @."))
            return

        filename = os.path.basename(stage_path)

        # Parse --embed-images flag for /parse command
        parse_embed_images = False
        if "--embed-images" in stage_path:
            parse_embed_images = True
            stage_path = stage_path.replace("--embed-images", "").strip()
            filename = os.path.basename(stage_path)

        # Fire and forget background task with working context
        asyncio.create_task(
            process_document_background(
                client,
                stage_path,
                filename,
                conv_id,
                current_working_db,
                current_working_schema,
                embed_images=parse_embed_images,
            )
        )
        yield to_sse(
            message_chunk(
                f"✅ **Parsing started.**\n\n"
                f"Document `{filename}` is being processed in the background"
                + (" with image embedding" if parse_embed_images else "")
                + ".\n"
                f"Results will be saved to `DOCUMENT_PARSE_RESULTS` table in {current_working_db}.{current_working_schema}."
            )
        )

    elif user_command == "/models":
        yield to_sse(reasoning_step("Fetching available models..."))
        try:
            models = await run_in_thread(client.get_available_models)
            if models:
                model_list = "**Available Models:**\n\n"
                for model_name, description in models:
                    model_list += f"- `{model_name}` - {description}\n"
                yield to_sse(message_chunk(model_list))
            else:
                yield to_sse(
                    message_chunk("No models found or unable to fetch model list.")
                )
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Error fetching models: {str(e)}"))

    elif user_command == "/clear":
        yield to_sse(reasoning_step("Clearing conversation history..."))
        try:
            if agent:
                await run_in_thread(agent.reset_conversation)
            try:
                await run_in_thread(client.clear_conversation, conv_id)
            except Exception as e:
                if is_session_expired_error(e):
                    # Import refresh_client from server
                    from .server import refresh_client

                    logger.warning(
                        "Session expired during /clear, refreshing client..."
                    )
                    client = refresh_client(conv_id)
                    await run_in_thread(client.clear_conversation, conv_id)
                else:
                    raise
            clear_message_signatures(conv_id)

            # Clear document processor caches for this conversation
            from .document_processor import DocumentProcessor

            DocumentProcessor.instance().clear_conversation_cache(conv_id)

            yield to_sse(message_chunk("✅ Conversation history cleared."))
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Error clearing history: {str(e)}"))

    elif user_command.startswith("/model "):
        model_name = user_command[7:].strip()
        model_preferences[conv_id] = model_name
        current_settings = {
            "model": model_name,
            "temperature": temperature_preferences.get(conv_id, selected_temperature),
            "max_tokens": max_tokens_preferences.get(conv_id, selected_max_tokens),
            "database": await run_in_thread(client.get_current_database),
            "schema": await run_in_thread(client.get_current_schema),
            "token_usage": token_usage.get(conv_id, {}),
        }
        await run_in_thread(
            client.update_conversation_settings, conv_id, json.dumps(current_settings)
        )
        # Also persist to conversation data for reload on server restart
        await run_in_thread(
            client.set_conversation_data, conv_id, "model_preference", model_name
        )
        yield to_sse(message_chunk(f"✅ Model set to: `{model_name}`"))

    elif user_command.startswith("/temperature "):
        try:
            temp_value = float(user_command[13:].strip())
            if 0.0 <= temp_value <= 1.0:
                temperature_preferences[conv_id] = temp_value
                current_settings = {
                    "model": model_preferences.get(conv_id, selected_model),
                    "temperature": temp_value,
                    "max_tokens": max_tokens_preferences.get(
                        conv_id, selected_max_tokens
                    ),
                    "database": await run_in_thread(client.get_current_database),
                    "schema": await run_in_thread(client.get_current_schema),
                    "token_usage": token_usage.get(conv_id, {}),
                }
                await run_in_thread(
                    client.update_conversation_settings,
                    conv_id,
                    json.dumps(current_settings),
                )
                # Also persist to conversation data for reload on server restart
                await run_in_thread(
                    client.set_conversation_data,
                    conv_id,
                    "temperature_preference",
                    str(temp_value),
                )
                yield to_sse(message_chunk(f"✅ Temperature set to: {temp_value}"))
            else:
                yield to_sse(
                    message_chunk("❌ Temperature must be between 0.0 and 1.0")
                )
        except ValueError:
            yield to_sse(
                message_chunk(
                    "❌ Invalid temperature value. Must be a number between 0.0 and 1.0"
                )
            )

    elif user_command.startswith("/max_tokens "):
        try:
            tokens_value = int(user_command[12:].strip())
            if tokens_value > 0:
                max_tokens_preferences[conv_id] = tokens_value
                current_settings = {
                    "model": model_preferences.get(conv_id, selected_model),
                    "temperature": temperature_preferences.get(
                        conv_id, selected_temperature
                    ),
                    "max_tokens": tokens_value,
                    "database": await run_in_thread(client.get_current_database),
                    "schema": await run_in_thread(client.get_current_schema),
                    "token_usage": token_usage.get(conv_id, {}),
                }
                await run_in_thread(
                    client.update_conversation_settings,
                    conv_id,
                    json.dumps(current_settings),
                )
                # Also persist to conversation data for reload on server restart
                await run_in_thread(
                    client.set_conversation_data,
                    conv_id,
                    "max_tokens_preference",
                    str(tokens_value),
                )
                yield to_sse(message_chunk(f"✅ Max tokens set to: {tokens_value}"))
            else:
                yield to_sse(message_chunk("❌ Max tokens must be a positive integer"))
        except ValueError:
            yield to_sse(
                message_chunk("❌ Invalid max tokens value. Must be a positive integer")
            )

    elif user_command == "/current":
        current_model = model_preferences.get(conv_id, selected_model)
        current_temp = temperature_preferences.get(conv_id, selected_temperature)
        current_tokens = max_tokens_preferences.get(conv_id, selected_max_tokens)

        settings_text = f"""**Current Settings:**

- Model: `{current_model}`
- Temperature: {current_temp}
- Max Tokens: {current_tokens}
- Database: {await run_in_thread(client.get_current_database)}
- Schema: {await run_in_thread(client.get_current_schema)}"""

        yield to_sse(message_chunk(settings_text))

    elif user_command == "/databases":
        yield to_sse(reasoning_step("Fetching databases..."))
        try:
            # Use raw SQL to avoid caching
            result_json = await run_in_thread(
                client.execute_statement, "SHOW DATABASES"
            )
            rows = json.loads(result_json)
            databases = []
            for row in rows:
                name = get_row_value(row, "name")
                if name:
                    databases.append(name)

            if databases:
                db_list = "**Available Databases:**\n\n" + "\n".join(
                    f"- {db}" for db in databases
                )
                yield to_sse(message_chunk(db_list))
            else:
                yield to_sse(message_chunk("No databases found."))
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Error listing databases: {str(e)}"))

    elif user_command.startswith("/schemas"):
        parts = user_command.split()
        database = parts[1] if len(parts) > 1 else None

        yield to_sse(
            reasoning_step(
                f"Fetching schemas{f' for {database}' if database else ''}..."
            )
        )
        try:
            query = (
                f"SHOW SCHEMAS IN DATABASE {database}" if database else "SHOW SCHEMAS"
            )
            result_json = await run_in_thread(client.execute_statement, query)
            rows = json.loads(result_json)
            schemas = []
            for row in rows:
                schema_name = get_row_value(row, "name")
                if schema_name:
                    schemas.append(schema_name)

            if schemas:
                schema_list = (
                    f"**Schemas{f' in {database}' if database else ''}:**\n\n"
                    + "\n".join(f"- {s}" for s in schemas)
                )
                yield to_sse(message_chunk(schema_list))
            else:
                yield to_sse(message_chunk("No schemas found."))
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Error listing schemas: {str(e)}"))

    elif user_command == "/warehouses":
        yield to_sse(reasoning_step("Fetching warehouses..."))
        try:
            result_json = await run_in_thread(
                client.execute_statement, "SHOW WAREHOUSES"
            )
            rows = json.loads(result_json)
            warehouses = []
            for row in rows:
                warehouse = get_row_value(row, "name")
                if warehouse:
                    warehouses.append(warehouse)

            if warehouses:
                wh_list = "**Available Warehouses:**\n\n" + "\n".join(
                    f"- {wh}" for wh in warehouses
                )
                yield to_sse(message_chunk(wh_list))
            else:
                yield to_sse(message_chunk("No warehouses found."))
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Error listing warehouses: {str(e)}"))

    elif user_command == "/tables":
        yield to_sse(reasoning_step("Fetching tables..."))
        try:
            current_db = await run_in_thread(client.get_current_database)
            current_schema = await run_in_thread(client.get_current_schema)

            query = f"""
            SELECT TABLE_NAME 
            FROM "{current_db}".INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = '{current_schema}' 
            ORDER BY TABLE_NAME
            """
            result_json = await run_in_thread(client.execute_statement, query)
            rows = json.loads(result_json)
            tables = []
            for row in rows:
                table_name = get_row_value(row, "table_name", "TABLE_NAME")
                if table_name:
                    tables.append(table_name)

            if tables:
                table_list = (
                    f"**Tables in {current_db}.{current_schema}:**\n\n"
                    + "\n".join(f"- {t}" for t in tables)
                )
                yield to_sse(message_chunk(table_list))
            else:
                yield to_sse(message_chunk("No tables found in current schema."))
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Error listing tables: {str(e)}"))

    elif user_command == "/stages":
        yield to_sse(reasoning_step("Fetching stages..."))
        try:
            # Get stages from both contexts
            result_json = await run_in_thread(client.execute_statement, "SHOW STAGES")
            rows = json.loads(result_json)
            stages = []

            for row in rows:
                name = get_row_value(row, "name")
                database_name = get_row_value(row, "database_name") or ""
                schema_name = get_row_value(row, "schema_name") or ""

                if name:
                    # Show fully qualified name if not in current context
                    if database_name and schema_name:
                        if (
                            database_name != current_working_db
                            or schema_name != current_working_schema
                        ):
                            stages.append(f"{database_name}.{schema_name}.{name}")
                        else:
                            stages.append(name)
                    else:
                        stages.append(name)

            # Also check OPENBB_AGENTS stages if not already in that context
            if current_working_db != "OPENBB_AGENTS":
                snowflake_user = await run_in_thread(client.get_current_user)
                sanitized_user = "".join(
                    c if c.isalnum() else "_" for c in snowflake_user
                )
                user_schema = f"USER_{sanitized_user}".upper()

                try:
                    show_user_stages = (
                        f"SHOW STAGES IN SCHEMA OPENBB_AGENTS.{user_schema}"
                    )
                    user_result = await run_in_thread(
                        client.execute_statement, show_user_stages
                    )
                    user_rows = json.loads(user_result)
                    for row in user_rows:
                        name = get_row_value(row, "name")
                        if name:
                            stages.append(f"OPENBB_AGENTS.{user_schema}.{name}")
                except Exception:
                    pass  # Ignore errors accessing user schema

            if stages:
                stage_list = "**Available Stages:**\n\n" + "\n".join(
                    f"- {s}" for s in stages
                )
                yield to_sse(message_chunk(stage_list))
            else:
                yield to_sse(message_chunk("No stages found."))
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Error listing stages: {str(e)}"))

    elif user_command.startswith("/stage_files "):
        stage_name = user_command[13:].strip()
        yield to_sse(reasoning_step(f"Listing files in stage '{stage_name}'..."))
        try:
            # Handle both qualified and unqualified stage names
            if "." in stage_name and not stage_name.startswith("@"):
                # Fully qualified stage name
                list_query = f"LIST @{stage_name}"
            elif stage_name.startswith("@"):
                # Already has @ prefix
                list_query = f"LIST {stage_name}"
            else:
                # Unqualified - use current context
                list_query = f"LIST @{stage_name}"

            result_json = await run_in_thread(client.execute_statement, list_query)
            rows = json.loads(result_json)

            files = []
            for row in rows:
                name = get_row_value(row, "name")
                if name:
                    # Extract filename from path (logic from engine.rs)
                    filename = name.split("/")[-1] if "/" in name else name
                    files.append(filename)

            if files:
                file_list = f"**Files in {stage_name}:**\n\n" + "\n".join(
                    f"- {f}" for f in files
                )
                yield to_sse(message_chunk(file_list))
            else:
                yield to_sse(message_chunk(f"No files found in stage {stage_name}."))
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Error listing files: {str(e)}"))

    elif user_command.startswith("/use_database "):
        db_name = user_command[14:].strip()
        yield to_sse(reasoning_step(f"Switching to database '{db_name}'..."))
        try:
            await run_in_thread(client.use_database, db_name)
            current_settings = {
                "model": model_preferences.get(conv_id, selected_model),
                "temperature": temperature_preferences.get(
                    conv_id, selected_temperature
                ),
                "max_tokens": max_tokens_preferences.get(conv_id, selected_max_tokens),
                "database": db_name,
                "schema": await run_in_thread(client.get_current_schema),
                "token_usage": token_usage.get(conv_id, {}),
            }
            await run_in_thread(
                client.update_conversation_settings,
                conv_id,
                json.dumps(current_settings),
            )
            yield to_sse(message_chunk(f"✅ Switched to database: {db_name}"))
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Error switching database: {str(e)}"))

    elif user_command.startswith("/use_schema "):
        schema_name = user_command[12:].strip()
        yield to_sse(reasoning_step(f"Switching to schema '{schema_name}'..."))
        try:
            await run_in_thread(client.use_schema, schema_name)
            current_settings = {
                "model": model_preferences.get(conv_id, selected_model),
                "temperature": temperature_preferences.get(
                    conv_id, selected_temperature
                ),
                "max_tokens": max_tokens_preferences.get(conv_id, selected_max_tokens),
                "database": await run_in_thread(client.get_current_database),
                "schema": schema_name,
                "token_usage": token_usage.get(conv_id, {}),
            }
            await run_in_thread(
                client.update_conversation_settings,
                conv_id,
                json.dumps(current_settings),
            )
            yield to_sse(message_chunk(f"✅ Switched to schema: {schema_name}"))
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Error switching schema: {str(e)}"))

    elif user_command.startswith("/use_warehouse "):
        warehouse_name = user_command[15:].strip()
        yield to_sse(reasoning_step(f"Switching to warehouse '{warehouse_name}'..."))
        try:
            await run_in_thread(client.use_warehouse, warehouse_name)
            yield to_sse(message_chunk(f"✅ Switched to warehouse: {warehouse_name}"))
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Error switching warehouse: {str(e)}"))

    elif user_command.startswith("/suggest "):
        prompt = user_command[len("/suggest ") :].strip()

        if not prompt:
            yield to_sse(message_chunk("❌ Please provide a prompt for /suggest."))
            return

        yield to_sse(
            reasoning_step("Calling text2sql to generate SQL without executing it...")
        )
        try:
            raw_response = await run_in_thread(client.text2sql, prompt)
            parsed = json.loads(raw_response)

            sql_text = (parsed.get("sql") or "").strip()
            explanation = (parsed.get("explanation") or "").strip()
            request_id = parsed.get("request_id")

            output_parts = []
            if explanation:
                output_parts.append(explanation)
            if sql_text:
                output_parts.append(f"```sql\n{sql_text}\n```")
            else:
                output_parts.append("No SQL was generated.")
            if request_id:
                output_parts.append(f"(request_id: {request_id})")

            yield to_sse(message_chunk("\n\n".join(output_parts)))
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Error generating SQL suggestion: {str(e)}"))

    elif user_command == "/history":
        yield to_sse(reasoning_step("Fetching conversation history..."))
        try:
            messages = await run_in_thread(client.get_messages, conv_id)

            # Fetch token usage from cache
            token_usage_str = await run_in_thread(
                client.get_conversation_data, conv_id, "token_usage"
            )
            total_tokens = 0
            if token_usage_str:
                try:
                    usage_data = json.loads(token_usage_str)
                    total_tokens = usage_data.get("total_tokens", 0)
                except json.JSONDecodeError:
                    pass

            if messages:
                total_msgs = len(messages)
                user_msgs = sum(
                    1
                    for _, role, content in messages
                    if role in ("user", "human")
                    and not content.startswith("[Tool Result")
                )
                ai_msgs = sum(
                    1 for _, role, _ in messages if role in ("assistant", "ai")
                )
                tool_calls = sum(
                    1
                    for _, role, content in messages
                    if role in ("user", "tool")
                    and (content.startswith("[Tool Result") or role == "tool")
                )

                history_text = "**Conversation Statistics:**\n\n"
                history_text += f"- Total Messages: {total_msgs}\n"
                history_text += f"- User Messages: {user_msgs}\n"
                history_text += f"- Assistant Messages: {ai_msgs}\n"
                history_text += f"- Tool Calls: {tool_calls}\n"
                history_text += f"- Total Tokens Used: {total_tokens:,}\n\n"

                history_text += "**Recent Messages (Last 5):**\n\n"

                # Show last 5 messages
                recent = messages[-5:]
                for _, role, content in recent:
                    truncated = content[:150] + "..." if len(content) > 150 else content
                    # Escape newlines for cleaner display in list
                    clean_content = truncated.replace("\n", " ")
                    history_text += f"- **{role.upper()}:** {clean_content}\n"

                yield to_sse(message_chunk(history_text))
            else:
                yield to_sse(message_chunk("No conversation history found."))
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Error fetching history: {str(e)}"))

    elif user_command == "/context":
        yield to_sse(reasoning_step("Fetching database context..."))
        try:
            current_db = await run_in_thread(client.get_current_database)
            current_schema = await run_in_thread(client.get_current_schema)

            query = f"""
            SELECT TABLE_NAME 
            FROM "{current_db}".INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = '{current_schema}' 
            ORDER BY TABLE_NAME
            """
            result_json = await run_in_thread(client.execute_statement, query)
            rows = json.loads(result_json)
            tables = []
            for row in rows:
                table_name = get_row_value(row, "table_name", "TABLE_NAME")
                if table_name:
                    tables.append(table_name)

            context_text = f"""**Database Context:**

- Database: {current_db}
- Schema: {current_schema}
- Tables: {len(tables) if tables else 0} tables available
"""
            if tables and len(tables) <= 10:
                context_text += "\n\n**Tables:**\n\n" + "\n".join(
                    f"  - {t}" for t in tables[:10]
                )
            elif tables:
                context_text += f"\n**Tables:** {', '.join(tables[:5])} ... and {len(tables)-5} more"

            yield to_sse(message_chunk(context_text))
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Error fetching context: {str(e)}"))

    else:
        # Command not recognized
        yield to_sse(
            message_chunk(
                f"❌ Unknown command: {user_command}\nUse /help to see available commands."
            )
        )


async def process_document_background(
    client: SnowflakeAI,
    stage_path: str,
    filename: str,
    conv_id: str,
    target_database: str,
    target_schema: str,
    embed_images: bool = False,
):
    """Background task to parse document and save results to the working database context.

    Parameters
    ----------
    embed_images : bool, optional
        Whether to generate embeddings for images in the document, by default False
    """
    try:
        # 1. Parse document
        content = await run_in_thread(client.parse_document, stage_path)

        # 2. Ensure results table exists in the working database/schema context
        qualified_table = (
            f'"{target_database}"."{target_schema}"."DOCUMENT_PARSE_RESULTS"'
        )
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {qualified_table} (
            FILE_NAME STRING,
            STAGE_PATH STRING,
            PAGE_NUMBER INTEGER,
            PAGE_CONTENT STRING,
            METADATA VARIANT,
            PARSED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        await run_in_thread(client.execute_statement, create_table_query)

        # 3. Process and Insert Data using single SQL statement
        doc_name = os.path.splitext(filename)[0]

        # Extract stage name and relative file path from stage_path (e.g. @STAGE/path/to/file.pdf)
        clean_path = stage_path.lstrip("@")
        if "/" in clean_path:
            parts = clean_path.split("/", 1)
            # Check if it's a fully qualified stage
            if "." in parts[0]:
                stage_name_extracted = f"@{parts[0]}"
            else:
                stage_name_extracted = f"@{parts[0]}"
            relative_file_path = parts[1]
        else:
            # Fallback if path is weird, though upload ensures @STAGE/file
            stage_name_extracted = f"@{clean_path}"
            relative_file_path = filename

        # Perform parsing and insertion in a single query to avoid client timeout and data transfer
        insert_query = f"""
        INSERT INTO {qualified_table} (FILE_NAME, STAGE_PATH, PAGE_NUMBER, PAGE_CONTENT, METADATA)
        WITH DOC AS (
          SELECT AI_PARSE_DOCUMENT(TO_FILE('{stage_name_extracted}', '{relative_file_path}'), {{'mode': 'LAYOUT', 'page_split': true}}) AS RAW
        )
        SELECT 
            '{filename}', 
            '{stage_path}', 
            INDEX + 1,
            GET(VALUE, 'content')::STRING,
            GET(DOC.RAW, 'metadata')
        FROM DOC, TABLE(FLATTEN(input => COALESCE(GET(DOC.RAW, 'pages'), DOC.RAW)))
        """
        await run_in_thread(client.execute_statement, insert_query)

        # Update cache with page count
        count_query = f"SELECT COUNT(*) as cnt FROM {qualified_table} WHERE FILE_NAME = '{filename}' AND STAGE_PATH = '{stage_path}'"
        count_result_json = await run_in_thread(client.execute_statement, count_query)
        count_rows = json.loads(count_result_json)
        if count_rows:
            num_pages = count_rows[0].get("CNT", count_rows[0].get("cnt", 0))

            await run_in_thread(
                client.set_conversation_data,
                conv_id,
                f"doc_{doc_name}_page_count",
                str(num_pages),
            )

        # Generate embeddings after parsing
        if num_pages > 0:
            try:
                from .document_processor import DocumentProcessor

                doc_proc = DocumentProcessor.instance()

                # Get parsed pages
                select_query = f"SELECT PAGE_NUMBER, PAGE_CONTENT FROM {qualified_table} WHERE FILE_NAME = '{filename}' AND STAGE_PATH = '{stage_path}' ORDER BY PAGE_NUMBER"
                result_json = await run_in_thread(
                    client.execute_statement, select_query
                )
                rows = json.loads(result_json) if result_json else []

                pages = []
                for row in rows:
                    page_num = (
                        row.get("PAGE_NUMBER")
                        if "PAGE_NUMBER" in row
                        else row.get("page_number")
                    )
                    content = (
                        row.get("PAGE_CONTENT")
                        if "PAGE_CONTENT" in row
                        else row.get("page_content")
                    )
                    if page_num and content:
                        pages.append({"page": int(page_num), "content": content})

                if pages:
                    logger.info(
                        "Generating embeddings for %s (embed_images=%s)",
                        filename,
                        embed_images,
                    )
                    await doc_proc._create_document_embeddings_table(client)
                    success = await doc_proc._generate_embeddings_for_document(
                        client=client,
                        file_name=filename,
                        stage_path=stage_path,
                        pages=pages,
                        pdf_bytes=None,
                        embed_images=embed_images,
                    )
                    if success:
                        logger.info(
                            "Embeddings generated successfully for %s", filename
                        )
                    else:
                        logger.warning("Failed to generate embeddings for %s", filename)
            except Exception as emb_err:
                logger.error(
                    "Embedding generation failed for %s: %s", filename, emb_err
                )

    except Exception as e:
        logger.error("Background parsing failed for %s: %s", filename, e)
