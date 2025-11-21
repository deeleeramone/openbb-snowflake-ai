# Snowflake AI - Python Bindings

Python bindings (via PyO3/Rust) for Snowflake AI_COMPLETE with HTTP server and SSE support.

## Installation

```bash
cd /Users/darrenlee/github/OpenBB/desktop/snowflake
pip install maturin
maturin develop --release
```

## Usage

### Direct Python API

```python
from snowflake_ai import SnowflakeAI

# Initialize
client = SnowflakeAI(
    user="your_user",
    password="your_password",
    account="your_account",
    role="your_role",
    warehouse="your_warehouse",
    database="your_database",
    schema="your_schema",
)

# Simple completion
response = client.complete(
    prompt="What is Snowflake?",
    model="snowflake-arctic",
    temperature=0.7,
    max_tokens=2048,
)
print(response)

# Chat with history
conversation = [
    ("user", "What is Python?"),
    ("assistant", "Python is a programming language..."),
    ("user", "What are its main uses?"),
]

response = client.chat(
    messages=conversation,
    model="snowflake-arctic",
)
print(response)
```

### HTTP Server

```bash
# Set environment variables
export SNOWFLAKE_USER=your_user
export SNOWFLAKE_PASSWORD=your_password
export SNOWFLAKE_ACCOUNT=your_account
export SNOWFLAKE_ROLE=your_role
export SNOWFLAKE_WAREHOUSE=your_warehouse
export SNOWFLAKE_DATABASE=your_database
export SNOWFLAKE_SCHEMA=your_schema

# Start server
python -m openbb_snowflake_ai.server
```

Server runs on `http://localhost:8000` with OpenAI-compatible endpoints.

## API Endpoint

**POST `/v1/chat/completions`**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "model": "snowflake-arctic",
    "stream": true
  }'
```
