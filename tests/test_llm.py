"""Tests for the disk-cached Anthropic client.

We don't call the real API. We swap the underlying SDK client with a fake
that produces a real `Message`, then verify that caching round-trips through
JSON and that distinct requests produce distinct cache keys.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from anthropic.types import Message, TextBlock, Usage

from ancestors.llm import CachedAnthropic


def _fake_message(text: str = "hello") -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text=text)],
        model="claude-opus-4-7",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(
            input_tokens=1,
            output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            server_tool_use=None,
            service_tier="standard",
        ),
    )


@pytest.fixture
def client(tmp_path: Path) -> CachedAnthropic:
    c = CachedAnthropic(api_key="fake", cache_dir=tmp_path / "cache", use_cache=True)
    c.client = MagicMock()
    c.client.messages.create.return_value = _fake_message("first")
    return c


def test_first_call_invokes_api_and_caches(client):
    msg = client.messages(messages=[{"role": "user", "content": "hi"}])
    assert msg.content[0].text == "first"
    assert client.client.messages.create.call_count == 1


def test_second_identical_call_hits_cache(client):
    args = {"messages": [{"role": "user", "content": "hi"}]}
    client.messages(**args)
    client.client.messages.create.return_value = _fake_message("second")
    msg = client.messages(**args)
    assert msg.content[0].text == "first"  # cached, not the new response
    assert client.client.messages.create.call_count == 1


def test_changing_messages_misses_cache(client):
    client.messages(messages=[{"role": "user", "content": "hi"}])
    client.client.messages.create.return_value = _fake_message("second")
    msg = client.messages(messages=[{"role": "user", "content": "different"}])
    assert msg.content[0].text == "second"
    assert client.client.messages.create.call_count == 2


def test_changing_tools_misses_cache(client):
    base = {"messages": [{"role": "user", "content": "hi"}]}
    client.messages(**base)
    client.client.messages.create.return_value = _fake_message("with-tools")
    msg = client.messages(**base, tools=[{"name": "x", "description": "y", "input_schema": {}}])
    assert msg.content[0].text == "with-tools"
    assert client.client.messages.create.call_count == 2


def test_no_cache_mode_always_calls_api(tmp_path: Path):
    c = CachedAnthropic(api_key="fake", use_cache=False)
    c.client = MagicMock()
    c.client.messages.create.return_value = _fake_message()
    c.messages(messages=[{"role": "user", "content": "hi"}])
    c.messages(messages=[{"role": "user", "content": "hi"}])
    assert c.client.messages.create.call_count == 2
