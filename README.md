# VAKLM - Versatile AI Knowledge Language Model

A Python client for interacting with OpenAI-compatible API endpoints with reasoning support.

## Installation

```bash
pip install vaklm
```

## Usage

### Basic Usage

```python
from vaklm import vaklm, VAKLMException

# Non-streaming response with reasoning
try:
    content, reasoning = vaklm(
        endpoint="http://localhost:11434/v1/chat/completions",
        model_name="llama3.2:latest",
        user_prompt="Explain quantum computing in simple terms",
        system_prompt="You are a helpful AI assistant",
        api_key="YOUR_API_KEY",
        temperature=0.7
    )
    print("Content:", content)
    print("Reasoning:", reasoning)
except VAKLMException as e:
    print(f"Error: {str(e)}")
```

### Streaming Usage

```python
from vaklm import vaklm

print("\nStreaming example:")
try:
    for content, reasoning in vaklm(
        endpoint="http://localhost:11434/v1/chat/completions",
        model_name="llama3.2:latest",
        user_prompt="Write a short story about a cat.",
        system_prompt="You are a creative writer.",
        api_key="YOUR_API_KEY",
        stream=True,
        temperature=0.7
    ):
        print(content, end='', flush=True)
        if reasoning:
            print(f"\n[Reasoning: {reasoning}]")
except VAKLMException as e:
    print(f"Error: {str(e)}")
```

## Features

- Supports both streaming and non-streaming responses
- Includes reasoning content in responses
- Automatic retry logic for failed requests
- Configurable temperature and max tokens
- System prompt support for context setting
- Comprehensive error handling

## Configuration

The `vaklm` function accepts the following parameters:

- `endpoint`: API endpoint URL (required)
- `model_name`: Model identifier (required)
- `user_prompt`: User's input message (required)
- `system_prompt`: Optional system context message
- `api_key`: API key for authentication
- `stream`: Whether to stream the response (default: False)
- `temperature`: Sampling temperature (0-2, default: 1.0)
- `max_tokens`: Maximum tokens to generate
- `timeout`: Request timeout in seconds (default: 30)
- `max_retries`: Maximum retry attempts (default: 3)
- `retry_delay`: Base delay between retries (default: 1.0)

## Error Handling

The client raises `VAKLMException` for general errors, with specific subclasses:
- `APIError`: For API-specific errors
- `StreamingError`: For streaming-specific errors

Always wrap calls in try/except blocks to handle potential errors.
