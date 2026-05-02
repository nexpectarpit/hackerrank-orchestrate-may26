from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from groq import APIStatusError
from groq import Groq
from groq import RateLimitError as GroqRateLimitError

from config import settings
from utils.errors import InvalidResponseError, LLMError, RateLimitError
from utils.logger import get_logger

log = get_logger(__name__)


class LLMClient:
    """Wrapper around the Groq SDK with retry and JSON parsing logic."""

    def __init__(self) -> None:
        if not settings.GROQ_API_KEY:
            log.warning("GROQ_API_KEY is not set. LLM calls will fail.")
            self.client = None
        else:
            self.client = Groq(api_key=settings.GROQ_API_KEY)

    def generate(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> str:
        """Call the Groq LLM with retries and exponential backoff.

        Args:
            model: Override the default model. Use settings.GROQ_FAST_MODEL
                   for lightweight tasks to save tokens.
        """
        if not self.client:
            raise LLMError("Groq client not initialized (missing API key).")

        use_model = model or settings.GROQ_MODEL
        temp = temperature if temperature is not None else settings.GROQ_TEMPERATURE
        max_retries = settings.GROQ_MAX_RETRIES
        delay = 2.0

        response_format = {"type": "json_object"} if json_mode else None

        for attempt in range(max_retries):
            try:
                # Add delay to avoid rate limits on free tier
                if attempt == 0 and settings.API_CALL_DELAY > 0:
                    time.sleep(settings.API_CALL_DELAY)

                response = self.client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    temperature=temp,
                    response_format=response_format,
                )
                content = response.choices[0].message.content
                if not content:
                    raise LLMError("Empty response from LLM")
                return content

            except GroqRateLimitError as e:
                if attempt < max_retries - 1:
                    log.warning("Rate limit hit, retrying in %d seconds...", delay)
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise RateLimitError(
                        f"Rate limit exceeded after {max_retries} attempts: {e}"
                    ) from e
            except APIStatusError as e:
                raise LLMError(f"Groq API error: {e}") from e
            except Exception as e:
                raise LLMError(f"Unexpected error calling LLM: {e}") from e

        raise LLMError("Exhausted retries calling LLM")

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate and parse a JSON response from the LLM."""
        content = self.generate(messages, json_mode=True, temperature=temperature, model=model)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise InvalidResponseError(f"Failed to parse JSON response: {content}") from e


# Singleton
_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


if __name__ == "__main__":
    # Test execution
    client = get_client()
    try:
        res = client.generate_json(
            [{"role": "user", "content": "Return a JSON object with key 'test' and value 'success'."}]
        )
        print("Response:", res)
    except Exception as e:
        print("Error:", e)
