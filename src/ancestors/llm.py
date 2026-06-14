"""Anthropic client wrapper with disk caching.

A thin layer over the Anthropic SDK. Knows nothing about agents, genealogy,
or tools — just request-in / response-out, with a deterministic disk cache
keyed on the canonical JSON of the request parameters. Caching is the
quality-of-life win during development; second runs of identical prompts
are free.

Patterned on menu_agent's `CachedOpenAI`, ported to Anthropic-native shapes
so we get the native tool_use / tool_result content blocks downstream.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import diskcache
from anthropic import Anthropic
from anthropic.types import Message

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0


class LlmConfigError(RuntimeError):
    """Raised when the API key cannot be located."""


def _keys_dir() -> Path:
    """Where API keys live on disk. Mirrors menu_agent's pattern."""
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "ancestors" / "keys"


def _resolve_api_key() -> str:
    """Find the Anthropic API key.

    Priority: ANTHROPIC_API_KEY env var, then ~/.config/ancestors/keys/anthropic.
    Failing loudly is correct — a silently-unauthenticated client is worse
    than a startup error.
    """
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env:
        return env.strip()

    key_file = _keys_dir() / "anthropic"
    if key_file.exists():
        return key_file.read_text().strip()

    raise LlmConfigError(
        "No Anthropic API key found. Set ANTHROPIC_API_KEY or write the key to "
        f"{key_file}"
    )


def _default_cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "ancestors" / "llm"


class CachedAnthropic:
    """Anthropic Messages API with deterministic disk cache.

    The cache key includes every parameter that could affect the response,
    so changing model / temperature / system / tools / messages all bypass
    a stale cached entry naturally.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        model: str = DEFAULT_MODEL,
        verbose: bool = False,
    ) -> None:
        self.client = Anthropic(api_key=api_key or _resolve_api_key())
        self.model = model
        self.use_cache = use_cache
        self.verbose = verbose
        if use_cache:
            cache_dir = cache_dir or _default_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache: diskcache.Cache | None = diskcache.Cache(str(cache_dir))
        else:
            self.cache = None

    def messages(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        model: str | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> Message:
        """Call messages.create with caching. Returns a typed Message.

        Cached entries round-trip through Pydantic so the on-disk format is
        JSON and the in-memory object is the SDK's native type.
        """
        request: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system is not None:
            request["system"] = system
        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        if self.cache is not None:
            key = _cache_key(request)
            cached = self.cache.get(key)
            if cached is not None:
                if self.verbose:
                    log.debug("LLM cache hit: %s", key[:12])
                return Message.model_validate_json(cached)

        response = self.client.messages.create(**request)

        if self.cache is not None:
            self.cache[key] = response.model_dump_json()
            if self.verbose:
                log.debug("LLM cache miss, stored: %s", key[:12])

        return response


def _cache_key(request: dict[str, Any]) -> str:
    """Canonical SHA256 of the request. Same input ⇒ same key, every run."""
    payload = json.dumps(request, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
