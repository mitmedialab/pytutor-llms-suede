from release.common import Msg, OnText, TextChunk

from pydantic import BaseModel

from abc import ABC, abstractmethod
from typing import (
    Awaitable,
    Callable,
    AsyncIterator,
    List,
    Optional,
    Sequence,
    TypeVar,
)

type TextStream = AsyncIterator[str]
type GetTextStream = Callable[[], Awaitable[TextStream]]
type Interrupt = Callable[[], bool]

ChunkT = TypeVar("ChunkT")
type ChunkProducer[T] = Callable[
    ["Provider.TextStream.Request"], Awaitable[AsyncIterator[T]]
]
type ContentFromChunk[T] = Callable[[T], str | None]


class Base:
    class Callbacks(BaseModel):
        error: Optional[Callable[[Exception], None]] = None

    class Request(BaseModel):
        messages: List[Msg]
        model: str
        interrupt: Optional[Interrupt]


class Provider(ABC):
    class TextStream:
        class Request(Base.Request):
            class Callbacks(Base.Callbacks):
                chunk: OnText
                done: Optional[OnText] = None

            on: Callbacks

        @classmethod
        async def Stream(
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
