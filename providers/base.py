from ..common import Msg, OnText, TextChunk

from dataclasses import dataclass, field
from pydantic import BaseModel

from abc import ABC, abstractmethod
from typing import (
    Awaitable,
    Callable,
    AsyncIterator,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    TypeVar,
)

if TYPE_CHECKING:
    from .anthropic import AnthropicModelMetadata
    from .google import GoogleModelMetadata
    from .openai import OpenAIModelMetadata
    from .openrouter import OpenRouterModelMetadata

    type ModelMetadata = (
        OpenAIModelMetadata
        | AnthropicModelMetadata
        | GoogleModelMetadata
        | OpenRouterModelMetadata
    )
else:
    type ModelMetadata = object

type TextStream = AsyncIterator[str]
type GetTextStream = Callable[[], Awaitable[TextStream]]
type Interrupt = Callable[[], bool]

ChunkT = TypeVar("ChunkT")
MetadataT = TypeVar("MetadataT")
type ChunkProducer[T] = Callable[
    ["Provider.TextStream.Request"], Awaitable[AsyncIterator[T]]
]
type ContentFromChunk[T] = Callable[[T], str | None]


class Base:
    @dataclass(frozen=True, kw_only=True)
    class Callbacks:
        error: Optional[Callable[[Exception], None]] = None

    @dataclass(frozen=True, kw_only=True)
    class Request:
        messages: List[Msg]
        model: str
        model_metadata: frozenset[ModelMetadata] = field(default_factory=frozenset)
        interrupt: Optional[Interrupt] = None


class Provider(ABC):
    @classmethod
    def model_metadata(
        cls,
        request: Base.Request,
        metadata_type: type[MetadataT],
    ) -> MetadataT | None:
        return next(
            (
                metadata
                for metadata in request.model_metadata
                if isinstance(metadata, metadata_type)
            ),
            None,
        )

    class TextStream:
        @dataclass(frozen=True, kw_only=True)
        class Request(Base.Request):
            @dataclass(frozen=True, kw_only=True)
            class Callbacks(Base.Callbacks):
                chunk: OnText
                done: Optional[OnText] = None

            on: Callbacks

        @classmethod
        async def FromChunks(
            cls,
            request: Request,
            *,
            chunk_producer: ChunkProducer[ChunkT],
            content_from_chunk: ContentFromChunk[ChunkT],
        ) -> TextStream:
            i = 0
            accumulated = ""

            try:
                async for chunk in await chunk_producer(request):
                    if request.interrupt and request.interrupt():
                        break

                    content = content_from_chunk(chunk)
                    if content is None:
                        continue

                    accumulated += content
                    yield request.on.chunk(
                        TextChunk(delta=content, accumulated=accumulated, i=i)
                    )
                    i += 1
            except Exception as e:
                if request.on.error:
                    request.on.error(e)
            finally:
                if request.on.done:
                    yield request.on.done(
                        TextChunk(accumulated=accumulated, i=i, delta="")
                    )

    class StructuredStream:
        class Request(Base.Request):
            json_schema: str

    class ModelStream:
        class Request(Base.Request):
            messages: List[Msg]
            type: type[BaseModel]

    @abstractmethod
    async def try_prepare_text_stream(
        self,
        request: "Provider.TextStream.Request",
    ) -> Optional[GetTextStream]:
        raise NotImplementedError


async def select_text_stream(
    providers: Sequence[Provider],
    request: "Provider.TextStream.Request",
) -> TextStream:
    for provider in providers:
        starter = await provider.try_prepare_text_stream(request)
        if starter is not None:
            return await starter()  # only one stream actually starts
    raise ValueError(f"No provider accepted model={request.model}")
