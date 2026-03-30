"""
Model client adapter.

Isolates all LLM provider logic so the harness never touches provider APIs directly.
Swap backends by changing configs/models.yaml — nothing else changes.

Supported backends:
  - ollama   : local Ollama (current Eureka path)
  - openai   : OpenAI / any OpenAI-compatible HTTP endpoint
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI


@dataclass
class ToolCall:
    """Normalised tool call extracted from any provider response."""
    function_name: str
    arguments: dict[str, Any]
    call_source: str  # 'native' | 'fallback'


@dataclass
class ModelConfig:
    """Everything needed to instantiate a client for one model."""
    name: str                          # model identifier sent to the API
    backend: str                       # 'ollama' | 'openai'
    base_url: str
    api_key: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            name=d["name"],
            backend=d["backend"],
            base_url=d["base_url"],
            api_key=d.get("api_key", "none"),
            extra=d.get("extra", {}),
        )


_BACKEND_DEFAULTS = {
    "ollama": "http://localhost:11434/v1",
    "openai": "https://api.openai.com/v1",
}

_BACKEND_API_KEY_ENV_VARS = {
    "ollama": ("OLLAMA_API_KEY", "LLM_API_KEY"),
    "openai": ("OPENAI_API_KEY", "LLM_API_KEY"),
}


class ModelClient:
    """
    Thin wrapper around the OpenAI-compatible chat completions API.

    Both Ollama and vLLM expose the same /v1/chat/completions endpoint,
    so the same client works for both — only base_url and api_key differ.

    Usage:
        cfg = ModelConfig(name="qwen2.5:7b", backend="ollama",
                          base_url="http://localhost:11434/v1")
        client = ModelClient(cfg)
        tool_call = client.get_tool_call(messages, tools)
    """

    def __init__(self, config: ModelConfig, allow_fallback: bool = False):
        self.config = config
        self.allow_fallback = allow_fallback
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict], tools: list[dict]) -> Any:
        """Raw chat completion — returns the OpenAI response object."""
        return self._client.chat.completions.create(
            model=self.config.name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

    def get_tool_call(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> Optional[ToolCall]:
        """
        Call the model and extract a ToolCall, or None if the model
        produced no tool call.

        Falls back to JSON-in-text parsing when allow_fallback=True and
        the model doesn't emit native tool_calls (useful for models that
        don't yet support structured outputs on their endpoint).
        """
        response = self.chat(messages, tools)
        message = response.choices[0].message

        # 1. Native tool call (preferred path)
        if getattr(message, "tool_calls", None):
            tc = message.tool_calls[0]
            return ToolCall(
                function_name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
                call_source="native",
            )

        # 2. Fallback: model emitted JSON-in-text
        if self.allow_fallback and getattr(message, "content", None):
            parsed = _parse_fallback_json(message.content)
            if parsed:
                return ToolCall(
                    function_name=str(parsed["tool"]),
                    arguments=parsed["args"],
                    call_source="fallback",
                )

        return None

    def probe_tool_support(self) -> bool:
        """
        Quick check: does this model/endpoint support native tool calls?
        Returns True if yes, False if the endpoint rejects the request.
        """
        try:
            self._client.chat.completions.create(
                model=self.config.name,
                messages=[{"role": "user", "content": "ping"}],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "ping",
                        "description": "ping",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }],
                tool_choice="auto",
            )
            return True
        except Exception as exc:
            if "does not support tools" in str(exc).lower():
                return False
            return True  # other errors are not tool-support failures


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_fallback_json(text: str) -> Optional[dict]:
    """
    Parse {"tool": "<name>", "args": {...}} from a text response.
    Strips markdown code fences if present.
    """
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None
    try:
        payload = json.loads(candidate)
    except Exception:
        return None
    if isinstance(payload, dict) and "tool" in payload and isinstance(payload.get("args"), dict):
        return payload
    return None


def client_from_yaml(model_name: str, configs_path: str = "configs/models.yaml") -> ModelClient:
    """
    Load a ModelClient by model name from configs/models.yaml.

    configs/models.yaml format:
        models:
          - name: qwen2.5:7b
            backend: ollama
            base_url: http://localhost:11434/v1
          - name: openai-compatible-local
            backend: openai
            base_url: http://localhost:8000/v1
            api_key: token-abc123
    """
    return ModelClient(resolve_model_config(model_name, configs_path=configs_path))


def resolve_model_config(
    model_name: str,
    *,
    backend: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    configs_path: str = "configs/models.yaml",
    default_backend: str = "ollama",
) -> ModelConfig:
    """
    Resolve model/provider configuration from configs/models.yaml first,
    then apply any explicit CLI overrides.

    If the model is not present in YAML, fall back to backend defaults.
    """
    yaml_entry = _find_model_entry(model_name, configs_path)

    resolved_backend = backend or (
        yaml_entry["backend"] if yaml_entry else default_backend
    )
    if resolved_backend not in _BACKEND_DEFAULTS:
        raise ValueError(
            f"Unsupported backend '{resolved_backend}'. "
            f"Expected one of {sorted(_BACKEND_DEFAULTS)}."
        )

    if base_url is not None:
        resolved_base_url = _normalize_base_url(base_url)
    elif yaml_entry and "base_url" in yaml_entry:
        resolved_base_url = _resolve_base_url_value(
            str(yaml_entry["base_url"]),
            backend=resolved_backend,
        )
    else:
        resolved_base_url = _normalize_base_url(_BACKEND_DEFAULTS[resolved_backend])

    if api_key is not None:
        resolved_api_key = api_key
    elif yaml_entry and "api_key" in yaml_entry:
        resolved_api_key = _resolve_api_key_value(str(yaml_entry["api_key"]))
    else:
        resolved_api_key = _default_api_key_for_backend(resolved_backend)

    extra = dict(yaml_entry.get("extra", {})) if yaml_entry else {}
    return ModelConfig(
        name=model_name,
        backend=resolved_backend,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        extra=extra,
    )


def _find_model_entry(model_name: str, configs_path: str) -> Optional[dict[str, Any]]:
    import yaml

    path = Path(configs_path)
    if not path.exists():
        return None

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    for entry in data.get("models", []):
        if entry.get("name") == model_name:
            return entry
    return None


def _resolve_api_key_value(value: str) -> str:
    if value in {"", "none"}:
        return "none"
    if value.startswith("$"):
        return os.environ.get(value[1:], "none")
    return os.environ.get(value, value)


def _resolve_base_url_value(value: str, *, backend: str) -> str:
    if value.startswith("$"):
        value = os.environ.get(value[1:], "")
    if not value:
        return _normalize_base_url(_BACKEND_DEFAULTS[backend])
    return _normalize_base_url(value)


def _normalize_base_url(value: str) -> str:
    normalized = value.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized += "/v1"
    return normalized


def _default_api_key_for_backend(backend: str) -> str:
    for env_var in _BACKEND_API_KEY_ENV_VARS.get(backend, ("LLM_API_KEY",)):
        value = os.environ.get(env_var)
        if value:
            return value
    return "none"


def provider_runtime_note(config: ModelConfig, *, allow_fallback: bool = False) -> Optional[str]:
    """
    Return a short provider-specific note for CLI output when one is useful.
    """
    if config.backend != "openai":
        return None

    note = (
        "OpenAI-compatible endpoints vary in native tool-calling support depending "
        "on the served model and runtime."
    )
    if not allow_fallback:
        note += " Use --allow-fallback if the endpoint returns JSON-in-text instead of tool_calls."
    return note
