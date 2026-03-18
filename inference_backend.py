"""Shared OpenAI-compatible inference backend helpers.

This module keeps backend selection lightweight while centralizing
provider defaults for Ollama and future OpenAI-compatible backends
such as vLLM.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI


DEFAULT_PROVIDER = "ollama"
DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
    "vllm": "http://localhost:8000",
}
DEFAULT_API_KEYS = {
    "ollama": "ollama",
    "vllm": "EMPTY",
}


@dataclass(frozen=True)
class BackendConfig:
    provider: str
    model: str
    base_url: str
    api_key: str

    @property
    def openai_base_url(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/v1") else f"{base}/v1"


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in DEFAULT_BASE_URLS:
        raise ValueError(f"Unsupported provider '{provider}'. Expected one of: {', '.join(DEFAULT_BASE_URLS)}")
    return normalized


def add_backend_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add shared inference/backend CLI args to a parser."""
    parser.add_argument(
        "--provider",
        choices=sorted(DEFAULT_BASE_URLS.keys()),
        default=None,
        help="Inference backend provider (default: ollama, or BACKEND_PROVIDER if set).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for the OpenAI-compatible inference endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the OpenAI-compatible inference endpoint.",
    )
    return parser


def load_backend_config_from_env() -> Dict[str, Optional[str]]:
    """Load shared backend config overrides from environment variables."""
    return {
        "provider": os.getenv("BACKEND_PROVIDER"),
        "base_url": os.getenv("BACKEND_BASE_URL"),
        "api_key": os.getenv("BACKEND_API_KEY"),
        "model": os.getenv("BACKEND_MODEL"),
    }


def resolve_backend_config(
    *,
    model: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BackendConfig:
    """Resolve a complete backend config from CLI args plus env defaults."""
    env_config = load_backend_config_from_env()

    resolved_provider = _normalize_provider(provider or env_config["provider"] or DEFAULT_PROVIDER)
    resolved_model = model or env_config["model"]
    if not resolved_model:
        raise ValueError("Model name is required.")

    resolved_base_url = (base_url or env_config["base_url"] or DEFAULT_BASE_URLS[resolved_provider]).rstrip("/")
    resolved_api_key = api_key or env_config["api_key"] or DEFAULT_API_KEYS[resolved_provider]

    return BackendConfig(
        provider=resolved_provider,
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
    )


def create_openai_client(config: BackendConfig) -> OpenAI:
    """Create an OpenAI client targeting an OpenAI-compatible backend."""
    return OpenAI(base_url=config.openai_base_url, api_key=config.api_key)


def run_chat_completion(
    client: OpenAI,
    config: BackendConfig,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    *,
    tool_choice: Optional[str] = None,
):
    """Run a chat completion against the configured backend."""
    return client.chat.completions.create(
        model=config.model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice if tools else None,
    )
