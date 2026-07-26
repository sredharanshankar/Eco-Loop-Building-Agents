"""Thin wrapper over an OpenAI-compatible endpoint (Ollama by default).

Kept deliberately small and defensive: hard timeout, bounded retries, and it always
returns a structured result the supervisor can branch on, rather than throwing deep in
the control loop. Any modern OSS model served behind an OpenAI-compatible API
(Ollama, vLLM, llama.cpp server, LM Studio, TGI) works by changing base_url/model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


@dataclass
class LLMResult:
    ok: bool
    message: Any = None            # the assistant message object (may hold tool_calls)
    latency_s: float = 0.0
    error: str = ""
    raw_usage: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    def __init__(self, cfg: dict):
        a = cfg["agent"]
        self.model = a["model"]
        self.temperature = a["temperature"]
        self.timeout_s = a["timeout_s"]
        self.max_retries = a["max_retries"]
        self.num_ctx = a.get("num_ctx", 4096)
        self.max_tokens = a.get("max_tokens", 220)
        self.client = OpenAI(
            base_url=a["base_url"],
            api_key=a.get("api_key", "ollama"),
            timeout=self.timeout_s,
            max_retries=0,   # we manage retries ourselves for clearer telemetry
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str = "auto") -> LLMResult:
        last_err = ""
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    # Ollama: keep the model resident (avoid 40s reloads on low-RAM machines)
                    # and honour num_ctx. Both are ignored by non-Ollama endpoints.
                    "extra_body": {"keep_alive": -1, "options": {"num_ctx": self.num_ctx}},
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice
                resp = self.client.chat.completions.create(**kwargs)
                dt = time.perf_counter() - t0
                usage = {}
                if getattr(resp, "usage", None):
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                    }
                return LLMResult(True, resp.choices[0].message, dt, "", usage)
            except Exception as exc:  # noqa: BLE001 - loop must never die here
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(0.6 * (attempt + 1))
        return LLMResult(False, None, self.timeout_s, last_err)

    def health(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
