# utils/llm_client.py

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    import requests

from typing import Dict, List, Any

class LLMClient:
    """LLM client, prioritizes official SDK, falls back to HTTP requests"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.openai.com/v1", model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        # Token usage accounting, accumulated across every chat_completion call.
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "num_calls": 0}
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

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

    def _record_usage(self, prompt_tokens, completion_tokens, total_tokens):
        """Record token usage for one call and add it to the running totals."""
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        total_tokens = int(total_tokens or (prompt_tokens + completion_tokens))
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += total_tokens
        self.usage["num_calls"] += 1

    def get_usage(self) -> Dict[str, int]:
        """Cumulative token usage across all calls so far (returns a copy)."""
        return dict(self.usage)
    
    def _chat_completion_sdk(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Call using official SDK"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._record_usage(
                    getattr(usage, "prompt_tokens", 0),
                    getattr(usage, "completion_tokens", 0),
                    getattr(usage, "total_tokens", 0),
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
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            usage = result.get("usage") or {}
            self._record_usage(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("total_tokens", 0),
            )
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"HTTP API call failed: {e}")
            return ""