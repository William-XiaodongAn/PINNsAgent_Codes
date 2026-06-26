# utils/llm_client.py

import os
from typing import Dict, List

# Different providers name the token-usage fields differently. We try them in order
# so the same accounting works for OpenAI/Ollama (OpenAI format), Gemini and Claude.
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


def _resolve_key(env_name):
    """Provider API key from the process env, else the parent cardiac-agent/.env
    (gitignored) — mirrors how config_loader resolves OPENAI_API_KEY."""
    val = os.environ.get(env_name)
    if val:
        return val
    try:
        from utils.config_loader import _read_dotenv_key
        return _read_dotenv_key(env_name)
    except Exception:
        return None


def _infer_provider(model_name):
    """Fallback provider routing by model-name pattern (matches baselines/main.py
    get_model_provider) when no explicit provider is supplied."""
    m = (model_name or "").lower()
    if "qwen/qwen" in m:
        return "featherless"
    if "gpt" in m and "cloud" not in m:
        return "openai"
    if "claude" in m:
        return "anthropic"
    if "gemini" in m:
        return "gemini"
    if "cloud" in m:
        return "ollama_cloud"
    return "ollama_local"


class LLMClient:
    """Multi-provider LLM client.

    Routes by ``provider`` to each vendor's native SDK (OpenAI / Gemini / Anthropic /
    Ollama / Featherless), matching the parent cardiac-agent routing in
    baselines/main.py. The provider normally comes from the parent
    experiment_config.json ``llms`` entry; if omitted it is inferred from the model
    name. A single ``chat_completion()`` interface and cumulative token accounting
    keep the planner provider-agnostic. Per-provider SDKs are imported lazily, so a
    missing SDK only fails the providers that need it.
    """

    def __init__(self, api_key: str = None, base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-3.5-turbo", provider: str = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.provider = (provider or _infer_provider(model)).lower()

        # Token usage, accumulated across every call. We only track input / output tokens.
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}

    # ---- public API ------------------------------------------------------
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7,
                        max_tokens: int = 1000) -> str:
        """Send a chat request to the configured provider and return the text. On any
        error returns '' (the planner treats empty output as a failed generation)."""
        try:
            if self.provider in ("openai", "featherless", "openai_compatible"):
                content, it, ot = self._call_openai(messages, temperature)
            elif self.provider == "gemini":
                content, it, ot = self._call_gemini(messages)
            elif self.provider == "anthropic":
                content, it, ot = self._call_anthropic(messages, max_tokens)
            elif self.provider in ("ollama_cloud", "ollama_local"):
                content, it, ot = self._call_ollama(messages, local=self.provider == "ollama_local")
            else:
                content, it, ot = self._call_openai(messages, temperature)
            self._record_usage(it, ot)
            return content
        except Exception as e:
            print(f"LLM call failed (provider={self.provider}, model={self.model}): {e}")
            return ""

    def get_usage(self) -> Dict[str, int]:
        """Cumulative input/output token usage across all calls so far (returns a copy)."""
        return dict(self.usage)

    # ---- usage bookkeeping ----------------------------------------------
    def _record_usage(self, input_tokens, output_tokens):
        """Record input/output token usage for one call and add it to the running totals."""
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        self.last_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        self.usage["input_tokens"] += input_tokens
        self.usage["output_tokens"] += output_tokens

    @staticmethod
    def _split_messages(messages):
        """Split OpenAI-style role messages into (system_prompt, user_prompt) for the
        providers (Gemini/Anthropic) that take the system instruction separately."""
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        user = "\n".join(m["content"] for m in messages if m.get("role") != "system")
        return system, user

    # ---- provider calls (native SDKs, lazily imported) -------------------
    def _call_openai(self, messages, temperature):
        from openai import OpenAI
        if self.provider == "featherless":
            client = OpenAI(base_url="https://api.featherless.ai/v1",
                            api_key=self.api_key or _resolve_key("FEATHERLESS_AI_API_KEY"))
        else:
            client = OpenAI(api_key=self.api_key or _resolve_key("OPENAI_API_KEY"),
                            base_url=self.base_url)
        resp = client.chat.completions.create(model=self.model, messages=messages,
                                              temperature=temperature)
        u = getattr(resp, "usage", None)
        return (resp.choices[0].message.content,
                _read_token_field(u, INPUT_TOKEN_FIELDS),
                _read_token_field(u, OUTPUT_TOKEN_FIELDS))

    def _call_gemini(self, messages):
        from google import genai
        from google.genai import types
        system, user = self._split_messages(messages)
        client = genai.Client(api_key=self.api_key or _resolve_key("GEMINI_API_KEY"))
        cfg = types.GenerateContentConfig(system_instruction=system) if system else None
        resp = client.models.generate_content(model=self.model, contents=user, config=cfg)
        u = getattr(resp, "usage_metadata", None)
        return (resp.text,
                _read_token_field(u, INPUT_TOKEN_FIELDS),
                _read_token_field(u, OUTPUT_TOKEN_FIELDS))

    def _call_anthropic(self, messages, max_tokens):
        import anthropic
        system, user = self._split_messages(messages)
        client = anthropic.Anthropic(api_key=self.api_key or _resolve_key("ANTHROPIC_API_KEY"))
        msg = client.messages.create(model=self.model, max_tokens=max_tokens or 4096,
                                     system=system, messages=[{"role": "user", "content": user}])
        u = getattr(msg, "usage", None)
        return (msg.content[0].text,
                _read_token_field(u, INPUT_TOKEN_FIELDS),
                _read_token_field(u, OUTPUT_TOKEN_FIELDS))

    def _call_ollama(self, messages, local=False):
        import ollama
        if local:
            client = ollama.Client()
        else:
            client = ollama.Client(host="https://ollama.com",
                                   headers={"Authorization": f"Bearer {_resolve_key('OLLAMA_API_KEY')}"})
        resp = client.chat(model=self.model, messages=messages)
        # ollama returns a ChatResponse (mapping-like) with top-level *_eval_count fields.
        return (resp["message"]["content"],
                _read_token_field(resp, INPUT_TOKEN_FIELDS),
                _read_token_field(resp, OUTPUT_TOKEN_FIELDS))
