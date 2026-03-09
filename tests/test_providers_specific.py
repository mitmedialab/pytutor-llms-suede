from types import SimpleNamespace
from typing import Any, Mapping, cast

import pytest
from google.genai import types as genai_types
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from pydantic import BaseModel

from release import Provider

from release.providers.anthropic import (
    AnthropicProvider,
    SUPPORTED_IMAGE_MEDIA_TYPES,
    _to_messages as anthropic_to_messages,
)
from release.providers.google import (
    GoogleProvider,
    _config_from_metadata,
    _to_messages as google_to_messages,
)
from release.providers.openai import (
    OpenAIProvider,
    delta_content_from_chunk as openai_delta,
)
from release.providers.openrouter import OpenRouterProvider


class DemoModel(BaseModel):
    value: str


def _text_request(model: str, **kwargs):
    return {
        "messages": [{"role": "user", "content": "hello"}],
        "model": model,
        **kwargs,
    }


def _pydantic_request(model: str):
    from release.providers.base import Provider

    return Provider.PydanticStream.Request(
        messages=[{"role": "user", "content": "hello"}],
        model=model,
        type=DemoModel,
    )


def test_openai_delta_content_from_chunk() -> None:
    chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="abc"))]
    )
    assert openai_delta(cast(ChatCompletionChunk, chunk)) == "abc"
    assert openai_delta(cast(ChatCompletionChunk, SimpleNamespace(choices=[]))) is None


def test_google_config_from_metadata() -> None:
    metadata = GoogleProvider.ModelMetadata(thinking_level="high")
    config = _config_from_metadata(metadata)
    assert config == {
        "thinking_config": {"thinking_level": genai_types.ThinkingLevel.HIGH}
    }


def test_google_to_messages_handles_system_and_images() -> None:
    from release.providers.base import Provider

    request = Provider.TextStream.Request(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    }
                ],
            },
        ],
    )

    contents, config = google_to_messages(request)

    assert len(contents) == 2
    assert config is not None
    assert config.get("system_instruction") == "sys"


def test_anthropic_to_messages_filters_unsupported_data_images() -> None:
    from release.providers.base import Provider

    request = Provider.TextStream.Request(
        model="claude-3-5-sonnet",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/svg+xml;base64,PHN2Zz4="},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                    },
                ],
            }
        ],
        model_metadata=[AnthropicProvider.ModelMetadata(max_tokens=100)],
    )

    system_prompt, messages = anthropic_to_messages(request)

    assert system_prompt == ""
    assert len(messages) == 1
    first_message = cast(Mapping[str, Any], messages[0])
    blocks = cast(list[Mapping[str, Any]], first_message.get("content", []))
    image_blocks = [block for block in blocks if block.get("type") == "image"]
    assert len(image_blocks) == 1
    source = cast(Mapping[str, Any], image_blocks[0].get("source", {}))
    media_type = source.get("media_type")
    assert isinstance(media_type, str)
    assert media_type in SUPPORTED_IMAGE_MEDIA_TYPES


@pytest.mark.asyncio
async def test_provider_model_acceptance_and_stream_start(monkeypatch) -> None:
    from release.providers import anthropic as anthropic_mod
    from release.providers import google as google_mod
    from release.providers import openai as openai_mod
    from release.providers import openrouter as openrouter_mod
    from release.providers.base import Provider

    async def raw_chunk_producer(_request):
        async def iterator():
            yield "chunk"

        return iterator()

    monkeypatch.setattr(openai_mod, "produce_raw_chunks", raw_chunk_producer)
    monkeypatch.setattr(openrouter_mod, "produce_raw_chunks", raw_chunk_producer)
    monkeypatch.setattr(google_mod, "produce_raw_chunks", raw_chunk_producer)
    monkeypatch.setattr(anthropic_mod, "produce_raw_chunks", raw_chunk_producer)

    monkeypatch.setattr(openai_mod, "delta_content_from_chunk", lambda c: c)
    monkeypatch.setattr(openrouter_mod, "delta_content_from_chunk", lambda c: c)
    monkeypatch.setattr(google_mod, "delta_content_from_chunk", lambda c: c)
    monkeypatch.setattr(anthropic_mod, "delta_content_from_chunk", lambda c: c)

    cases = [
        (OpenAIProvider(), "gpt-4o-mini", True),
        (OpenAIProvider(), "gemini-2.0-flash", False),
        (GoogleProvider(), "gemini-2.0-flash", True),
        (GoogleProvider(), "claude-3-5-sonnet", False),
        (AnthropicProvider(), "claude-3-5-sonnet", True),
        (AnthropicProvider(), "gpt-4o-mini", False),
        (OpenRouterProvider(), "openrouter/meta-llama/llama-3", True),
        (OpenRouterProvider(), "gpt-4o-mini", False),
    ]

    for provider, model, accepted in cases:
        request = Provider.TextStream.Request(
            messages=[{"role": "user", "content": "hello"}],
            model=model,
        )
        starter = await provider.try_prepare_text_stream(request)

        if not accepted:
            assert starter is None
            continue

        assert starter is not None
        stream = await starter()
        events = await Provider.TextStream.Collect(stream)
        assert len(events) == 1
        assert events[0].type == "chunk"
        assert events[0].payload.delta == "chunk"


@pytest.mark.asyncio
async def test_provider_pydantic_stream_acceptance(monkeypatch) -> None:
    from release.providers import openai as openai_mod

    async def produce_models(_request):
        async def iterator():
            yield DemoModel(value="a")

        return iterator()

    monkeypatch.setattr(openai_mod, "produce_pydantic_models", produce_models)

    provider = OpenAIProvider()
    starter = await provider.try_prepare_pydantic_stream(
        _pydantic_request("gpt-4o-mini")
    )
    assert starter is not None

    stream = await starter()
    events = await Provider.PydanticStream.Collect(stream)

    assert len(events) == 1
    assert events[0].type == "chunk" and events[0].payload.current == DemoModel(
        value="a"
    )
