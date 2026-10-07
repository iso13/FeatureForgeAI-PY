"""Pluggable LLM access.

TS equivalent: the ``new OpenAI(...)`` instances created at import time in
``failureClassifier.ts``, ``developerBrief.ts`` and ``generators/llm/llmClient.ts``.

Here every AI feature takes an :class:`LLMClient`. Without a provider configured
the :class:`NullLLM` is used and callers fall back to heuristics, so a run never
needs network access or an API key.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Protocol

from featureforge.core.settings import Settings

JSONDict = dict[str, Any]

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMClient(Protocol):
    name: str

    def complete_json(
        self, prompt: str, *, max_tokens: int = 400, temperature: float = 0.1
    ) -> JSONDict | None:
        """Return the parsed JSON object, or None when unavailable or unparsable."""
        ...


def parse_json_object(text: str) -> JSONDict | None:
    cleaned = _FENCE.sub("", text.strip()).strip()
    if not cleaned.startswith("{"):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        cleaned = cleaned[start : end + 1]
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class NullLLM:
    name = "none"

    def complete_json(
        self, prompt: str, *, max_tokens: int = 400, temperature: float = 0.1
    ) -> JSONDict | None:
        return None


class OpenAILLM:
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self._api_key = api_key
        self.model = model
        self._client: Any = None

    def complete_json(
        self, prompt: str, *, max_tokens: int = 400, temperature: float = 0.1
    ) -> JSONDict | None:
        try:
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key)
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
        except Exception:  # network, auth, missing extra — degrade to heuristics
            return None
        return parse_json_object(content)


class OllamaLLM:
    name = "ollama"

    def __init__(self, host: str, model: str = "llama3.1", timeout: float = 60.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete_json(
        self, prompt: str, *, max_tokens: int = 400, temperature: float = 0.1
    ) -> JSONDict | None:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except Exception:
            return None
        return parse_json_object(str(body.get("response", "")))


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return OpenAILLM(settings.openai_api_key, settings.resolved_llm_model)
    if settings.llm_provider == "ollama":
        return OllamaLLM(settings.ollama_host, settings.resolved_llm_model)
    return NullLLM()
