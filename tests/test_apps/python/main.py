#!/usr/bin/env python3
"""Smoke tests for liter-llm published package.

Validates the published package works against real LLM APIs.
Requires API keys in environment variables or .env file at repo root.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Load .env from repo root
env_file = Path(__file__).resolve().parent.parent.parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from liter_llm import LlmClient


class SmokeTest:
    """Simple test runner that tracks pass/fail/skip counts."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def run(self, name: str, coro: object) -> None:
        """Run a single async test case."""
        try:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(coro)
            finally:
                loop.close()
            if result is None:
                self.skipped += 1
            else:
                self.passed += 1
        except Exception:
            self.failed += 1

    def summary(self) -> int:
        """Print summary and return exit code."""
        self.passed + self.failed + self.skipped
        return 1 if self.failed > 0 else 0


async def test_chat_openai() -> str | None:
    """Chat completion against OpenAI gpt-4o-mini."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    client = LlmClient(api_key=key)
    r = await client.chat(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
    )
    assert r.choices, "no choices in response"
    assert r.choices[0].message.content, "empty content"
    assert r.usage, "no usage data"
    assert r.usage.total_tokens > 0, "zero tokens"
    return "ok"


async def test_chat_anthropic() -> str | None:
    """Chat completion against Anthropic claude-3-5-haiku."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    client = LlmClient(api_key=key)
    r = await client.chat(
        model="anthropic/claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
    )
    assert r.choices, "no choices"
    assert r.choices[0].message.content, "empty content"
    return "ok"


async def test_chat_gemini() -> str | None:
    """Chat completion against Google gemini-2.0-flash."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    client = LlmClient(api_key=key)
    r = await client.chat(
        model="google/gemini-2.0-flash",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
    )
    assert r.choices, "no choices"
    assert r.choices[0].message.content, "empty content"
    return "ok"


async def test_streaming_openai() -> str | None:
    """Streaming chat completion against OpenAI."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    client = LlmClient(api_key=key)
    chunks: list[object] = []
    async for chunk in await client.chat_stream(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        max_tokens=50,
    ):
        chunks.append(chunk)
    assert len(chunks) > 0, "no chunks received"
    return "ok"


async def test_embed_openai() -> str | None:
    """Embeddings request against OpenAI text-embedding-3-small."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    client = LlmClient(api_key=key)
    r = await client.embed(
        model="openai/text-embedding-3-small",
        input=["Hello, world!"],
    )
    assert r.data, "no embeddings"
    assert len(r.data[0].embedding) > 0, "empty embedding vector"
    return "ok"


async def test_list_models_openai() -> str | None:
    """List models against OpenAI."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    client = LlmClient(api_key=key)
    r = await client.list_models()
    assert r.data, "no models returned"
    assert len(r.data) > 0, "empty models list"
    return "ok"


async def test_provider_routing() -> str | None:
    """Test provider routing: same interface, different providers."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not openai_key or not anthropic_key:
        return None

    messages = [{"role": "user", "content": "Say hi."}]

    client_openai = LlmClient(api_key=openai_key)
    r1 = await client_openai.chat(model="openai/gpt-4o-mini", messages=messages, max_tokens=5)
    assert r1.choices, "OpenAI failed"

    client_anthropic = LlmClient(api_key=anthropic_key)
    r2 = await client_anthropic.chat(
        model="anthropic/claude-3-5-haiku-20241022",
        messages=messages,
        max_tokens=5,
    )
    assert r2.choices, "Anthropic failed"
    return "ok"


async def test_cache_memory() -> str | None:
    """Test in-memory caching: identical requests return cached response."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    client = LlmClient(api_key=key, cache={"max_entries": 10, "ttl_seconds": 60})
    messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    r1 = await client.chat(model="openai/gpt-4o-mini", messages=messages, max_tokens=5)
    r2 = await client.chat(model="openai/gpt-4o-mini", messages=messages, max_tokens=5)
    assert r1.choices, "first request failed"
    assert r2.choices, "second request failed"
    assert r1.choices[0].message.content == r2.choices[0].message.content, "cache miss - responses differ"
    return "ok"


def main() -> int:
    """Run all smoke tests and return exit code."""
    suite = SmokeTest()

    suite.run("OpenAI gpt-4o-mini", test_chat_openai())
    suite.run("Anthropic claude-3-5-haiku", test_chat_anthropic())
    suite.run("Google gemini-2.0-flash", test_chat_gemini())

    suite.run("OpenAI streaming", test_streaming_openai())

    suite.run("OpenAI text-embedding-3-small", test_embed_openai())

    suite.run("OpenAI list models", test_list_models_openai())

    suite.run("Multi-provider routing", test_provider_routing())

    suite.run("In-memory cache hit", test_cache_memory())

    return suite.summary()


if __name__ == "__main__":
    sys.exit(main())
