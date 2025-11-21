"""Slash command handler for Snowflake AI server."""

from typing import Any, AsyncIterator

from ._snowflake_ai import SnowflakeAI, SnowflakeAgent
from .helpers import to_sse


async def run_in_thread(func, *args, **kwargs):
    """Run a function in a background thread."""
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


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
) -> AsyncIterator[dict[str, Any]]:
    """
    Handle slash commands and yield SSE events.

    Returns an async generator that yields SSE-formatted dicts.
    If the command is not recognized, yields an error message.
    """
    from openbb_ai import message_chunk, reasoning_step

    if user_command == "/help":
        help_text = """### Available Commands

**Model Management:**
- `/models` - List available AI models
- `/model <name>` - Switch to a different model
- `/temperature <0.0-1.0>` - Set response temperature
- `/max_tokens <number>` - Set max response tokens

**Session Information:**
- `/current` - Show current session info (database, model config, usage stats)
- `/history` - Show conversation history summary

**Database Navigation:**
- `/databases` - List all databases
- `/schemas [database]` - List schemas (optionally in specific database)
- `/tables` - List tables in current database.schema
- `/warehouses` - List available warehouses
- `/stages` - List available stages
- `/stage_files <stage_name>` - List files in a stage

**Context Switching:**
- `/use_database <name>` - Switch to a different database
- `/use_schema <name>` - Switch to a different schema
- `/use_warehouse <name>` - Switch to a different warehouse

**Conversation Management:**
- `/clear` - Clear conversation history (all messages)

**SQL Assistance:**
- `/complete <partial_query>` - Complete a partial SQL query

💡 **Tip:** All commands must be typed exactly as shown."""
        yield to_sse(message_chunk(help_text))
        return

    elif user_command == "/models":
        yield to_sse(reasoning_step("Fetching available models...", event_type="INFO"))
        models = await run_in_thread(client.get_available_models)
        yield to_sse(
            message_chunk(
                "### Available AI Models\n\n"
                + "\n".join(f"- `{name}`" for name, _ in models)
            )
        )
        yield to_sse(message_chunk("\n\n💡 Use `/model <name>` to switch models"))
        return

    elif user_command == "/clear":
        try:
            messages_before = await run_in_thread(client.get_messages, conv_id)
            yield to_sse(
                reasoning_step(
                    f"Found {len(messages_before)} messages to delete...",
                    event_type="INFO",
                )
            )
            await run_in_thread(client.clear_conversation, conv_id)
            messages_after = await run_in_thread(client.get_messages, conv_id)
            if len(messages_after) == 0:
                yield to_sse(
                    message_chunk(
                        f"✅ Successfully deleted {len(messages_before)} messages. Conversation cleared!"
                    )
                )
            else:
                yield to_sse(
                    message_chunk(
                        f"❌ ERROR: Failed to clear messages! {len(messages_after)} messages still remain!"
                    )
                )
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Failed to clear conversation: {str(e)}"))
        return

    elif user_command.startswith("/model "):
        model_name = user_command[7:].strip()
        if model_name:
            available_models = await run_in_thread(client.get_available_models)
            if model_name in [name for name, _ in available_models]:
                model_preferences[conv_id] = model_name
                await run_in_thread(
                    client.set_conversation_data,
                    conv_id,
                    "model_preference",
                    model_name,
                )
                yield to_sse(
                    reasoning_step(f"Switched to {model_name}", event_type="INFO")
                )
                yield to_sse(
                    message_chunk(
                        f"✅ Now using **{model_name}** for this conversation"
                    )
                )
            else:
                yield to_sse(
                    message_chunk(f"❌ Model `{model_name}` is not available.")
                )
        else:
            yield to_sse(message_chunk("⚠️ Please specify a model name"))
        return

    elif user_command.startswith("/temperature "):
        temp_str = user_command[13:].strip()
        try:
            temperature = float(temp_str)
            if 0.0 <= temperature <= 1.0:
                temperature_preferences[conv_id] = temperature
                await run_in_thread(
                    client.set_conversation_data,
                    conv_id,
                    "temperature_preference",
                    str(temperature),
                )
                yield to_sse(
                    reasoning_step(
                        f"Set temperature to {temperature}", event_type="INFO"
                    )
                )
                yield to_sse(
                    message_chunk(
                        f"✅ Temperature set to **{temperature}** for this conversation"
                    )
                )
            else:
                yield to_sse(
                    message_chunk("❌ Temperature must be between 0.0 and 1.0.")
                )
        except ValueError:
            yield to_sse(
                message_chunk(
                    "❌ Invalid temperature value. Please provide a number between 0.0 and 1.0."
                )
            )
        return

    elif user_command.startswith("/max_tokens "):
        tokens_str = user_command[12:].strip()
        try:
            max_tokens = int(tokens_str)
            if max_tokens > 0:
                max_tokens_preferences[conv_id] = max_tokens
                await run_in_thread(
                    client.set_conversation_data,
                    conv_id,
                    "max_tokens_preference",
                    str(max_tokens),
                )
                yield to_sse(
                    reasoning_step(f"Set max tokens to {max_tokens}", event_type="INFO")
                )
                yield to_sse(
                    message_chunk(
                        f"✅ Max tokens set to **{max_tokens}** for this conversation"
                    )
                )
            else:
                yield to_sse(message_chunk("❌ Max tokens must be a positive integer."))
        except ValueError:
            yield to_sse(
                message_chunk(
                    "❌ Invalid max tokens value. Please provide a positive integer."
                )
            )
        return

    elif user_command == "/current":
        # Import token_usage from server
        from snowflake_ai.server import token_usage

        # Get database context
        current_database = await run_in_thread(client.get_current_database)
        current_schema = await run_in_thread(client.get_current_schema)

        # Handle empty strings properly
        db_display = current_database if current_database else "(not set)"
        schema_display = current_schema if current_schema else "(not set)"

        # Get model configuration for this conversation
        actual_model = model_preferences.get(conv_id, selected_model)
        actual_temperature = temperature_preferences.get(conv_id, selected_temperature)
        actual_max_tokens = max_tokens_preferences.get(conv_id, selected_max_tokens)

        # Get conversation statistics
        messages = await run_in_thread(client.get_messages, conv_id)
        num_messages = len(messages)

        # Count tool calls (messages that look like tool responses)
        num_tool_calls = sum(
            1
            for _, role, content in messages
            if role == "user" and "[Tool Result" in content
        )

        # Get usage statistics from server's token_usage tracking
        usage = token_usage.get(
            conv_id,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "api_requests": 0,
            },
        )

        api_requests_str = str(usage.get("api_requests", 0))
        prompt_tokens_str = f"{usage.get('prompt_tokens', 0):,}"
        completion_tokens_str = f"{usage.get('completion_tokens', 0):,}"
        total_tokens_str = f"{usage.get('total_tokens', 0):,}"

        output = f"""### Current Session Information

**📍 Database Context:**
- **Database:** `{db_display}`
- **Schema:** `{schema_display}`

**🤖 Model Configuration:**
- **Model:** `{actual_model}`
- **Temperature:** `{actual_temperature}`
- **Max Tokens:** `{actual_max_tokens}`

**📊 Session Statistics:**
- **Messages:** {num_messages:,}
- **Tool Calls:** {num_tool_calls}
- **API Requests:** {api_requests_str}

**💰 Token Usage:**
- **Prompt Tokens:** {prompt_tokens_str}
- **Completion Tokens:** {completion_tokens_str}
- **Total Tokens:** {total_tokens_str}

**🔧 Commands:**
- `/help` - Show all available commands
- `/clear` - Clear conversation history"""

        yield to_sse(message_chunk(output))
        return

    elif user_command == "/databases":
        yield to_sse(reasoning_step("Fetching databases...", event_type="INFO"))
        try:
            databases = await run_in_thread(client.list_databases)
            current_database = await run_in_thread(client.get_current_database)
            db_list = []
            for db in databases:
                marker = " (current)" if db == current_database else ""
                db_list.append(f"- `{db}`{marker}")
            yield to_sse(
                message_chunk("### Accessible Databases\n\n" + "\n".join(db_list))
            )
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Failed to list databases: {str(e)}"))
        return

    elif user_command.startswith("/schemas"):
        yield to_sse(reasoning_step("Fetching schemas...", event_type="INFO"))
        try:
            parts = user_command.split(" ", 1)
            target_db = parts[1].strip() if len(parts) > 1 else None
            schemas = await run_in_thread(client.list_schemas, target_db)
            current_schema = await run_in_thread(client.get_current_schema)
            current_database = await run_in_thread(client.get_current_database)
            schema_list = []
            for s in schemas:
                marker = ""
                if s == current_schema and (
                    target_db is None or target_db == current_database
                ):
                    marker = " (current)"
                schema_list.append(f"- `{s}`{marker}")
            db_name_display = target_db if target_db else current_database
            yield to_sse(
                message_chunk(
                    f"### Schemas in {db_name_display}\n\n" + "\n".join(schema_list)
                )
            )
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Failed to list schemas: {str(e)}"))
        return

    elif user_command == "/warehouses":
        yield to_sse(reasoning_step("Fetching warehouses...", event_type="INFO"))
        try:
            warehouses = await run_in_thread(client.list_warehouses)
            wh_list = [f"- `{wh}`" for wh in warehouses]
            yield to_sse(
                message_chunk("### Accessible Warehouses\n\n" + "\n".join(wh_list))
            )
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Failed to list warehouses: {str(e)}"))
        return

    elif user_command == "/tables":
        yield to_sse(reasoning_step("Fetching tables...", event_type="INFO"))
        try:
            current_database = await run_in_thread(client.get_current_database)
            current_schema = await run_in_thread(client.get_current_schema)
            tables = await run_in_thread(
                client.list_tables_in, current_database, current_schema
            )
            table_list = [f"- `{t}`" for t in tables]
            yield to_sse(
                message_chunk(
                    f"### Tables in {current_database}.{current_schema}\n\n"
                    + "\n".join(table_list)
                )
            )
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Failed to list tables: {str(e)}"))
        return

    elif user_command == "/stages":
        yield to_sse(reasoning_step("Fetching stages...", event_type="INFO"))
        try:
            stages = await run_in_thread(client.list_stages)
            stage_list = [f"- `{s}`" for s in stages]
            yield to_sse(
                message_chunk("### Available Stages\n\n" + "\n".join(stage_list))
            )
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Failed to list stages: {str(e)}"))
        return

    elif user_command.startswith("/stage_files "):
        stage_name = user_command[13:].strip()
        if stage_name:
            yield to_sse(
                reasoning_step(
                    f"Fetching files in stage {stage_name}...",
                    event_type="INFO",
                )
            )
            try:
                files = await run_in_thread(client.list_files_in_stage, stage_name)
                file_list = [f"- `{f}`" for f in files]
                yield to_sse(
                    message_chunk(
                        f"### Files in Stage {stage_name}\n\n" + "\n".join(file_list)
                    )
                )
            except Exception as e:
                yield to_sse(
                    message_chunk(f"❌ Failed to list files in stage: {str(e)}")
                )
        else:
            yield to_sse(message_chunk("❌ Usage: /stage_files <stage_name>"))
        return

    elif user_command.startswith("/use_database "):
        db_name = user_command[14:].strip()
        if db_name:
            try:
                await run_in_thread(client.use_database, db_name)
                yield to_sse(message_chunk(f"✅ Switched to database: **{db_name}**"))
            except Exception as e:
                yield to_sse(message_chunk(f"❌ Failed to switch database: {str(e)}"))
        else:
            yield to_sse(message_chunk("❌ Usage: /use_database <database_name>"))
        return

    elif user_command.startswith("/use_schema "):
        schema_name = user_command[12:].strip()
        if schema_name:
            try:
                await run_in_thread(client.use_schema, schema_name)
                yield to_sse(message_chunk(f"✅ Switched to schema: **{schema_name}**"))
            except Exception as e:
                yield to_sse(message_chunk(f"❌ Failed to switch schema: {str(e)}"))
        else:
            yield to_sse(message_chunk("❌ Usage: /use_schema <schema_name>"))
        return

    elif user_command.startswith("/use_warehouse "):
        warehouse_name = user_command[15:].strip()
        if warehouse_name:
            try:
                await run_in_thread(client.use_warehouse, warehouse_name)
                yield to_sse(
                    message_chunk(f"✅ Switched to warehouse: **{warehouse_name}**")
                )
            except Exception as e:
                yield to_sse(message_chunk(f"❌ Failed to switch warehouse: {str(e)}"))
        else:
            yield to_sse(message_chunk("❌ Usage: /use_warehouse <warehouse_name>"))
        return

    elif user_command.startswith("/complete "):
        from .helpers import cleanup_text

        partial_query = user_command[10:].strip()
        if partial_query:
            yield to_sse(reasoning_step("Completing SQL query...", event_type="INFO"))
            try:
                # Get tool definitions for completion
                tools = [
                    client.get_table_sample_data_tool(),
                    client.get_table_definition_tool(),
                    client.get_multiple_table_definitions_tool(),
                    client.list_databases_tool(),
                    client.list_schemas_tool(),
                    client.list_tables_in_tool(),
                    client.validate_query_tool(),
                    client.execute_query_tool(),
                ]

                prompt = f"Complete this SQL query: {partial_query}\n\nProvide only the completed SQL query."
                completion = await run_in_thread(
                    agent.complete_with_tools,
                    message=prompt,
                    use_history=False,
                    model=selected_model,
                    temperature=selected_temperature,
                    max_tokens=selected_max_tokens,
                    tools=tools,
                    tool_choice=None,
                )
                yield to_sse(
                    message_chunk(
                        f"### Completed Query\n\n```sql\n{cleanup_text(completion)}\n```"
                    )
                )
            except Exception as e:
                yield to_sse(message_chunk(f"❌ Failed to complete query: {str(e)}"))
        else:
            yield to_sse(message_chunk("❌ Usage: /complete <partial_query>"))
        return

    elif user_command == "/history":
        yield to_sse(
            reasoning_step("Fetching conversation history...", event_type="INFO")
        )
        try:
            messages = await run_in_thread(client.get_messages, conv_id)

            # Summarize the conversation
            output = "### Conversation History Summary\n\n"
            output += f"**Total Messages:** {len(messages)}\n"

            # Count message types
            human_msgs = 0
            assistant_msgs = 0
            tool_results = []
            tables_accessed = set()
            queries_executed = []

            for msg_id, role, content in messages:
                if role in ["human", "user"]:
                    if "[Tool Result" in content:
                        # Extract tool name and details
                        import re

                        match = re.search(r"\[Tool Result from (\w+)\]", content)
                        if match:
                            tool_name = match.group(1)
                            tool_results.append(tool_name)

                            # Track specific operations
                            if "get_table" in tool_name:
                                table_match = re.search(
                                    r"table[_\s]+name[\"']?\s*:\s*[\"']?([A-Z0-9_\.]+)",
                                    content,
                                    re.IGNORECASE,
                                )
                                if table_match:
                                    tables_accessed.add(table_match.group(1))
                            elif tool_name == "execute_query":
                                query_match = re.search(
                                    r"SELECT.*?(?:FROM|WHERE|GROUP|ORDER|LIMIT|;)",
                                    content[:500],
                                    re.IGNORECASE | re.DOTALL,
                                )
                                if query_match:
                                    queries_executed.append(
                                        query_match.group(0)[:100] + "..."
                                    )
                    else:
                        human_msgs += 1
                elif role in ["assistant", "ai"]:
                    assistant_msgs += 1

            output += f"**Message Breakdown:**\n"
            output += f"- Human messages: {human_msgs}\n"
            output += f"- Assistant responses: {assistant_msgs}\n"
            output += f"- Tool results stored: {len(tool_results)}\n\n"

            if tool_results:
                output += f"**Tools Used:**\n"
                tool_counts = {}
                for tool in tool_results:
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1
                for tool, count in sorted(tool_counts.items()):
                    output += f"- `{tool}`: {count} time(s)\n"
                output += "\n"

            # Show recent conversation snippets
            output += "**Recent Conversation (last 10 exchanges):**\n"
            recent_messages = messages[-20:]  # Get last 20 to show exchanges

            for i, (msg_id, role, content) in enumerate(recent_messages):
                if role in ["human", "user"] and "[Tool Result" not in content:
                    snippet = content[:150].replace("\n", " ")
                    if len(content) > 150:
                        snippet += "..."
                    output += f"\n👤 **User:** {snippet}  \n"

                elif role in ["assistant", "ai"]:
                    snippet = content[:150].replace("\n", " ")
                    if len(content) > 150:
                        snippet += "..."
                    output += f"🤖 **Assistant:** {snippet}  \n"

            yield to_sse(message_chunk(output))
        except Exception as e:
            yield to_sse(message_chunk(f"❌ Failed to fetch history: {str(e)}"))
        return

    elif user_command == "/context":
        # New command to show what's currently in the AI's context window
        yield to_sse(
            reasoning_step("Checking current context window...", event_type="INFO")
        )

        messages = await run_in_thread(client.get_messages, conv_id)
        total_messages = len(messages)

        # Simulate what would be in context (last 20 messages)
        CONTEXT_WINDOW = 20
        context_messages = (
            messages[-CONTEXT_WINDOW:] if len(messages) > CONTEXT_WINDOW else messages
        )

        output = f"""### Current Context Window

**📊 Context Statistics:**
- **Total conversation history:** {total_messages} messages
- **Currently in AI context:** {len(context_messages)} messages (most recent)
- **Context window size:** {CONTEXT_WINDOW} messages

**📝 What's in current context:**
"""

        for msg_id, role, content in context_messages:
            if role in ["human", "user"]:
                if "[Tool Result" in content:
                    import re

                    match = re.search(r"\[Tool Result from (\w+)\]", content)
                    tool_name = match.group(1) if match else "unknown"
                    output += f"- 🔧 Tool result: `{tool_name}`\n"
                else:
                    snippet = (
                        content[:50].replace("\n", " ") + "..."
                        if len(content) > 50
                        else content
                    )
                    output += f"- 👤 User: {snippet}\n"
            elif role in ["assistant", "ai"]:
                snippet = (
                    content[:50].replace("\n", " ") + "..."
                    if len(content) > 50
                    else content
                )
                output += f"- 🤖 Assistant: {snippet}\n"

        output += """

**💡 Context Management:**
- Recent messages and tool results are automatically included
- When you reference earlier data, more context is loaded
- Use specific references like "the EXTRACTS table" or "the query we ran earlier"
- The AI intelligently expands context based on your questions"""

        yield to_sse(message_chunk(output))
        return

    else:
        # Unknown slash command
        cmd = user_command.split()[0]
        yield to_sse(
            message_chunk(
                f"❌ Unknown command: `{cmd}`\n\n"
                f"Type `/help` to see available commands."
            )
        )
        return
