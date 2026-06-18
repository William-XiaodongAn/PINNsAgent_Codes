# utils/llm_client.py

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    import requests

from typing import Dict, List

# Different providers name the token-usage fields differently. We try them in order
# so the same client works for OpenAI/Ollama (OpenAI format), Gemini and Claude.
INPUT_TOKEN_FIELDS = ("prompt_tokens", "input_tokens", "prompt_token_count", "prompt_eval_count")
OUTPUT_TOKEN_FIELDS = ("completion_tokens", "output_tokens", "candidates_token_count", "eval_count")


def _read_token_field(usage, names):
    """Read the first present field from a usage object or dict; 0 if none found."""
    if usage is None:
        return 0
    for n in names:
        if isinstance(usage, dict):
            val = usage.get(n)
        else:
            val = getattr(usage, n, None)
        if val is not None:
            return val
    return 0


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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            # OpenAI-format responses carry `.usage`; Gemini-native carry `.usage_metadata`.
            usage = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
            self._record_usage(
                _read_token_field(usage, INPUT_TOKEN_FIELDS),
                _read_token_field(usage, OUTPUT_TOKEN_FIELDS),
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI SDK call failed: {e}")
            return ""

    def _chat_completion_http(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Call using HTTP requests"""
        url = f"{self.base_url}/chat/completions"
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }

        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            usage = result.get("usage") or result.get("usage_metadata") or result.get("usageMetadata")
            self._record_usage(
                _read_token_field(usage, INPUT_TOKEN_FIELDS),
                _read_token_field(usage, OUTPUT_TOKEN_FIELDS),
            )
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"HTTP API call failed: {e}")
            return ""
