# utils/config_loader.py

import yaml
import os
from typing import Dict, Any, Optional


def _parse_env_value(env_path: str, key: str) -> Optional[str]:
    """Read a single KEY=value (also 'KEY = value' / 'export KEY=value') from a
    .env file. Returns None if the key is absent."""
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if k.startswith("export "):
                    k = k[len("export "):].strip()
                if k == key:
                    return v.strip().strip('"').strip("'") or None
    except OSError:
        return None
    return None


def _read_dotenv_key(key: str, start_dir: str = None, max_up: int = 6) -> Optional[str]:
    """Walk up from start_dir looking for a `.env` that defines `key`. Finds the
    parent cardiac-agent/.env (which is gitignored, so the secret stays local)."""
    d = start_dir or os.path.dirname(os.path.abspath(__file__))
    for _ in range(max_up):
        val = _parse_env_value(os.path.join(d, ".env"), key)
        if val:
            return val
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


class ConfigLoader:
    """Configuration file loader"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "../configs/default_config.yaml")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get_fixed_params(self) -> Dict[str, Any]:
        """Get fixed parameters"""
        return self.config.get("fixed_params", {})
    
    def get_pde_list(self, dimension: str) -> list:
        """Get PDE list for specified dimension"""
        return self.config.get("pde_lists", {}).get(dimension, [])
    
    def get_search_space(self) -> Dict[str, list]:
        """Get search space"""
        return self.config.get("search_space", {})
    
    def get_llm_config(self) -> Dict[str, str]:
        """Get LLM configuration.

        The API key is resolved without ever committing a real key to the yaml:
            1. OPENAI_API_KEY environment variable, else
            2. OPENAI_API_KEY from the parent cardiac-agent/.env (gitignored), else
            3. the (normally empty) yaml value.
        """
        cfg = dict(self.config.get("llm_config", {}))
        api_key = os.environ.get("OPENAI_API_KEY") or _read_dotenv_key("OPENAI_API_KEY")
        if api_key:
            cfg["api_key"] = api_key
        return cfg
    
    def update_fixed_params(self, **kwargs):
        """Update fixed parameters"""
        self.config["fixed_params"].update(kwargs)