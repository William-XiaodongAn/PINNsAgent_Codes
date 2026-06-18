# utils/llm_client.py

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    import requests

from typing import Dict, List

class LLMClient:
    """LLM client, prioritizes official SDK, falls back to HTTP requests"""

    def __init__(self, api_key: str = None, base_url: str = "https://api.openai.com/v1", model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        # Token usage, accumulated across every call. We only track input / output tokens.
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}

        if OPENAI_AVAILABLE and api_key:
            # Use official SDK
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self.use_sdk = True
        else:
            # Fall back to HTTP requests
            self.client = None
            self.use_sdk = False
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else ""
            }

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Call LLM chat completion API"""

        if self.use_sdk:
            return self._chat_completion_sdk(messages, temperature, max_tokens)
        else:
            return self._chat_completion_http(messages, temperature, max_tokens)

    def _record_usage(self, input_tokens, output_tokens):
        """Record input/output token usage for one call and add it to the running totals."""
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        self.last_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        self.usage["input_tokens"] += input_tokens
        self.usage["output_tokens"] += output_tokens

    def get_usage(self) -> Dict[str, int]:
        """Cumulative input/output token usage across all calls so far (returns a copy)."""
        return dict(self.usage)

    def _chat_completion_sdk(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Call using official SDK"""
        try:
            # Newer models (o-series / gpt-5 ...) reject `max_tokens` and require
            # `max_completion_tokens`. Try the classic name first, fall back on error.
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as e:
                if "max_completion_tokens" in str(e):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=max_tokens
                    )
                else:
                    raise

            usage = getattr(response, "usage", None)
            if usage is not None:
                self._record_usage(
                    getattr(usage, "prompt_tokens", 0),
                    getattr(usage, "completion_tokens", 0),
                )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI SDK call failed: {e}")
            return ""

    def _chat_completion_http(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Call using HTTP requests"""
        url = f"{self.base_url}/chat/completions"

        def _post(token_key):
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                token_key: max_tokens
            }
            return requests.post(url, headers=self.headers, json=data, timeout=30)

        try:
            # Try `max_tokens`, fall back to `max_completion_tokens` if the model rejects it.
            response = _post("max_tokens")
            if response.status_code == 400 and "max_completion_tokens" in response.text:
                response = _post("max_completion_tokens")
            response.raise_for_status()

            result = response.json()
            usage = result.get("usage") or {}
            self._record_usage(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"HTTP API call failed: {e}")
            return ""
