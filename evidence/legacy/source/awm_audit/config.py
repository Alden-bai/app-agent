from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .common import ROOT, digest


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str
    timeout: int = 120
    retries: int = 2

    def public(self):
        return {"base_url": self.base_url, "model": self.model,
                "timeout": self.timeout, "transport_retries": self.retries}

    @property
    def fingerprint(self):
        return digest(self.public())

    def redact(self, text):
        return str(text).replace(self.api_key, "[REDACTED]") if self.api_key else str(text)


def load_config(env_file=None, *, base_url=None, model=None, api_key=None):
    from dotenv import dotenv_values

    file_values = dotenv_values(env_file or ROOT / ".env", interpolate=False)
    # Resolve the entire group at each tier; a process alias beats a file native key.
    def choose(explicit, *names, default=""):
        if explicit is not None:
            return explicit
        for source in (os.environ, file_values):
            for name in names:
                if source.get(name):
                    return source[name]
        return default

    return LLMConfig(
        base_url=choose(base_url, "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL"),
        model=choose(model, "AWM_SYN_OVERRIDE_MODEL", "DEEPSEEK_MODEL"),
        api_key=choose(api_key, "OPENAI_API_KEY", "DEEPSEEK_API_KEY"),
    )


def validate_config(config):
    missing = [name for name in ("base_url", "model", "api_key") if not getattr(config, name)]
    if missing:
        raise ValueError("LLM configuration missing: " + ", ".join(missing))
    url = urlsplit(config.base_url)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
        raise ValueError("LLM base URL must be HTTP(S) without embedded credentials")
