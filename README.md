# vaklm

Easy interaction with OpenAI-compatible LLM endpoints.

No more `client.chat.completions.create` verbosity!

## Installation

```bash
pip install vaklm
```

## Usage

```python
from vaklm import vaklm, VAKLMException

print("Non-streaming example:")
try:
    response = vaklm(
        endpoint="http://localhost:11434/v1/chat/completions",
        model_name="llama3.2:latest",
        user_prompt="Write a short story about a cat.",
        system_prompt="You are a creative writer.",
        api_key="YOUR_API_KEY",
        temperature=0.7
    )
    print(response)
except VAKLMException as e:
    print(f"Error: {str(e)}")
```

```python
from vaklm import vaklm

print("\nStreaming example:")
try:
    for chunk in vaklm(
        endpoint="http://localhost:11434/v1/chat/completions",
        model_name="llama3.2:latest",
        user_prompt="Write a short story about a cat.",
        system_prompt="You are a creative writer.",
        api_key="YOUR_API_KEY",
        stream=True,
        temperature=0.7
    ):
        print(chunk, end='', flush=True)
except VAKLMException as e:
    print(f"Error: {str(e)}")
```
