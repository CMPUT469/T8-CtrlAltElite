"""
Model client adapter.

Isolates all LLM provider logic so the harness never touches provider APIs directly.
Swap backends by changing configs/models.yaml — nothing else changes.

Supported backends:
  - ollama   : local Ollama (current)
  - vllm     : vLLM OpenAI-compatible endpoint (Eureka target)
  - openai   : OpenAI / any OpenAI-compatible SaaS
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    backend: str                       # 'ollama' | 'vllm' | 'openai'
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

        # 1. Native tool call (preferred path for both Ollama and vLLM)
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
          - name: meta-llama/Llama-3.1-8B-Instruct
            backend: vllm
            base_url: http://eureka-node-01:8000/v1
            api_key: token-abc123
    """
    import yaml  # optional dep; only needed when using this helper
    with open(configs_path) as f:
        data = yaml.safe_load(f)
    for entry in data.get("models", []):
        if entry["name"] == model_name:
            return ModelClient(ModelConfig.from_dict(entry))
    raise ValueError(
        f"Model '{model_name}' not found in {configs_path}. "
        f"Available: {[m['name'] for m in data.get('models', [])]}"
    )
