from typing import Optional, Generator, Dict, Any, Union
import requests
import json
from dataclasses import dataclass
from enum import Enum
import logging
from contextlib import contextmanager
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}

class VAKLMException(Exception):
    """Base exception for VAKLM client errors"""
    pass

class APIError(VAKLMException):
    """Raised when the API returns an error"""
    pass

class StreamingError(VAKLMException):
    """Raised when there's an error during streaming"""
    pass

def vaklm(
    endpoint: str,
    model_name: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    api_key: str = "",
    stream: bool = False,
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    timeout: int = 30,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    **kwargs
) -> Union[str, Generator[tuple[str, str], None, None]]:
    """
    Make a request to an OpenAI-compatible endpoint.
    
    Args:
        endpoint: The API endpoint URL
        model_name: Name of the model to use
        user_prompt: The user's input prompt
        system_prompt: Optional system prompt to set context
        api_key: API key for authentication
        stream: Whether to stream the response
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries (will be multiplied by attempt number)
        **kwargs: Additional parameters to pass to the API
        
    Returns:
        Either a string response or a Generator of string chunks if streaming
        
    Raises:
        VAKLMException: For general errors
        APIError: For API-specific errors
        StreamingError: For streaming-specific errors
    """
    session = requests.Session()
    session.headers.update({
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    })

    messages = []
    if system_prompt:
        messages.append(Message(Role.SYSTEM, system_prompt))
    messages.append(Message(Role.USER, user_prompt))

    data = {
        'model': model_name,
        'messages': [msg.to_dict() for msg in messages],
        'stream': stream,
        'temperature': temperature,
        **kwargs
    }
    
    if max_tokens:
        data['max_tokens'] = max_tokens

    @contextmanager
    def handle_request_errors():
        try:
            yield
        except requests.exceptions.Timeout:
            raise VAKLMException("Request timed out")
        except requests.exceptions.RequestException as e:
            raise APIError(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise VAKLMException(f"Failed to decode JSON response: {str(e)}")
        except Exception as e:
            raise VAKLMException(f"Unexpected error: {str(e)}")

    def handle_streaming_response(response: requests.Response) -> Generator[tuple[str, str], None, None]:
        """Handle streaming response from the API"""
        reasoning_content = ""
        for line in response.iter_lines():
            if line:
                try:
                    if line.strip() == b"data: [DONE]":
                        break

                    if line.startswith(b"data: "):
                        json_str = line[6:].decode('utf-8')
                        data = json.loads(json_str)
                        delta = data.get('choices', [{}])[0].get('delta', {})
                        content = delta.get('content', '')
                        if delta.get('reasoning_content'):
                            reasoning_content += delta['reasoning_content']
                        yield (content, reasoning_content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode streaming response: {str(e)}")
                except Exception as e:
                    raise StreamingError(f"Error processing stream: {str(e)}")

    def handle_complete_response(response: requests.Response) -> tuple[str, str]:
        """Handle complete (non-streaming) response from the API"""
        data = response.json()
        choice = data.get('choices', [{}])[0]
        message = choice.get('message', {})
        reasoning = choice.get('reasoning', '')
        return (message.get('content', ''), reasoning)

    # Make request with retry logic
    endpoint = endpoint.rstrip('/')
    for attempt in range(max_retries):
        try:
            with handle_request_errors():
                response = session.post(
                    f"{endpoint}",
                    json=data,
                    stream=stream,
                    timeout=timeout
                )
                response.raise_for_status()

                if stream:
                    return handle_streaming_response(response)
                return handle_complete_response(response)

        except VAKLMException as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
            time.sleep(retry_delay * (attempt + 1))
        finally:
            if not stream:
                session.close()

if __name__ == '__main__':
    # Example usage
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
