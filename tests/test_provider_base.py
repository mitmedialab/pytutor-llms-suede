from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from release.common import Chunk
from release.providers.base import Event, Provider


class FakeProvider(Provider):
    def __init__(self, accepted_prefix: str):
        self.accepted_prefix = accepted_prefix

    async def try_prepare_text_stream(self, request):
        if not request.model.startswith(self.accepted_prefix):
            return None

        async def stream():
            async def iterator():
                yield Event.Text(
                    type="chunk",
                    payload=Chunk.Text(delta="ok", accumulated="ok", i=0),
                )

            return iterator()

        return stream

    async def try_prepare_pydantic_stream(self, request):
        if not request.model.startswith(self.accepted_prefix):
            return None

        async def stream():
            async def iterator():
                fields = list(request.type.model_fields.keys())
                payload = {fields[0]: "x"} if fields else {}
                yield Event.Pydantic(
                    type="chunk",
                    payload=Chunk.Pydantic(
                        current=request.type.model_validate(payload),
                        previous=None,
                        i=0,
                    ),
                )

            return iterator()

        return stream


@dataclass(frozen=True, kw_only=True)
class MetaA:
    value: int


def _text_request(**kwargs):
    model = kwargs.pop("model", "gpt-test")
    return Provider.TextStream.Request(messages=[], model=model, **kwargs)


class DemoModel(BaseModel):
    value: str


def _pydantic_request(**kwargs):
    model = kwargs.pop("model", "gpt-test")
    return Provider.PydanticStream.Request(
        messages=[], model=model, type=DemoModel, **kwargs
    )


async def _collect(async_iterable):
    out = []
    async for item in async_iterable:
        out.append(item)
    return out


@pytest.mark.asyncio
async def test_model_metadata_finds_matching_type() -> None:
    request = _text_request(model_metadata=[MetaA(value=3)])
    metadata = Provider.model_metadata(request, MetaA)
    assert metadata == MetaA(value=3)


@pytest.mark.asyncio
async def test_text_select_uses_first_matching_provider() -> None:
    stream = await Provider.TextStream.Select(
        _text_request(model="gemini-2.0"),
        FakeProvider("gpt"),
        FakeProvider("gemini"),
    )

    items = await _collect(stream)

    assert len(items) == 1
    assert items[0].payload.delta == "ok"


@pytest.mark.asyncio
async def test_select_uses_fallback_model_when_primary_rejected() -> None:
    stream = await Provider.TextStream.Select(
        _text_request(model="unknown-model", fallback_model="gpt-fallback"),
        FakeProvider("gpt"),
    )

    items = await _collect(stream)

    assert len(items) == 1
    assert items[0].payload.accumulated == "ok"


@pytest.mark.asyncio
async def test_text_stream_from_chunks_filters_none_and_accumulates() -> None:
    chunks = ["a", None, "b"]

    async def producer(_request):
        async def iterator():
            for chunk in chunks:
                yield chunk

        return iterator()

    def delta(chunk):
        return chunk

    stream = Provider.TextStream.FromChunks(
        _text_request(),
        raw_chunk_producer=producer,
        delta_content_from_chunk=delta,
    )

    items = await _collect(stream)

    assert [item.payload.delta for item in items] == ["a", "b"]
    assert [item.payload.accumulated for item in items] == ["a", "ab"]
    assert [item.payload.i for item in items] == [0, 1]


@pytest.mark.asyncio
async def test_text_stream_from_chunks_interrupts() -> None:
    chunks = ["a", "b", "c"]
    seen = {"count": 0}

    async def producer(_request):
        async def iterator():
            for chunk in chunks:
                yield chunk

        return iterator()

    def should_interrupt() -> bool:
        seen["count"] += 1
        return seen["count"] > 1

    stream = Provider.TextStream.FromChunks(
        _text_request(interrupt=should_interrupt),
        raw_chunk_producer=producer,
        delta_content_from_chunk=lambda c: c,
    )

    items = await _collect(stream)

    assert len(items) == 1
    assert items[0].payload.delta == "a"


@pytest.mark.asyncio
async def test_stream_from_chunks_emits_error_event_on_exception() -> None:
    async def producer(_request):
        raise RuntimeError("boom")

    stream = Provider.TextStream.FromChunks(
        _text_request(),
        raw_chunk_producer=producer,
        delta_content_from_chunk=lambda _c: "x",
    )

    items = await _collect(stream)

    assert len(items) == 1
    assert items[0].type == "error"
    assert isinstance(items[0].payload, RuntimeError)


@pytest.mark.asyncio
async def test_pydantic_stream_tracks_previous_model() -> None:
    models = [DemoModel(value="a"), DemoModel(value="b")]

    async def producer(_request):
        async def iterator():
            for model in models:
                yield model

        return iterator()

    stream = Provider.PydanticStream.FromModels(
        _pydantic_request(),
        model_producer=producer,
    )

    items = await _collect(stream)

    assert len(items) == 2
    assert items[0].payload.previous is None
    assert items[1].payload.previous == DemoModel(value="a")


@pytest.mark.asyncio
async def test_schema_stream_select_builds_model_and_streams() -> None:
    schema = '{"title":"Ticket","type":"object","properties":{"id":{"type":"string"}},"required":["id"]}'
    stream = await Provider.SchemaStream.Select(
        Provider.SchemaStream.Request(messages=[], model="gpt-test", schema=schema),
        FakeProvider("gpt"),
    )

    items = await _collect(stream)

    assert len(items) == 1
    assert items[0].payload.current.model_dump() == {"id": "x"}


@pytest.mark.asyncio
async def test_text_stream_collect_collects_all_events() -> None:
    stream = await Provider.TextStream.Select(
        _text_request(model="gpt-test"),
        FakeProvider("gpt"),
    )

    items = await Provider.TextStream.Collect(stream)

    assert len(items) == 1
    assert items[0].type == "chunk"
    assert items[0].payload.accumulated == "ok"


@pytest.mark.asyncio
async def test_pydantic_stream_collect_collects_all_events() -> None:
    stream = await Provider.PydanticStream.Select(
        _pydantic_request(model="gpt-test"),
        FakeProvider("gpt"),
    )

    items = await Provider.PydanticStream.Collect(stream)

    assert len(items) == 1
    assert items[0].type == "chunk"
    assert items[0].payload.current.model_dump() == {"value": "x"}


@pytest.mark.asyncio
async def test_schema_stream_collect_collects_all_events() -> None:
    schema = '{"title":"Ticket","type":"object","properties":{"id":{"type":"string"}},"required":["id"]}'
    stream = await Provider.SchemaStream.Select(
        Provider.SchemaStream.Request(messages=[], model="gpt-test", schema=schema),
        FakeProvider("gpt"),
    )

    items = await Provider.SchemaStream.Collect(stream)

    assert len(items) == 1
    assert items[0].type == "chunk"
    assert items[0].payload.current.model_dump() == {"id": "x"}
