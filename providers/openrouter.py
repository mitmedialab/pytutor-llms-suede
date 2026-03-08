from .base import GetTextStream, Provider
from .openai import content_from_chunk

from openai import AsyncOpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from dataclasses import dataclass
import os


client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


@dataclass(frozen=True, kw_only=True)
class OpenRouterModelMetadata:
    pass


async def produce_chunks(request: "Provider.TextStream.Request"):
    return await client.chat.completions.create(
        model=request.model,
        messages=request.messages,
        stream=True,
    )


class OpenRouterProvider(Provider):
    async def try_prepare_text_stream(self, request) -> GetTextStream | None:
        if not request.model.startswith(("qwen", "moonshot", "openrouter/")):
            return None

        async def stream():
            return Provider.TextStream.FromChunks(
                request,
                chunk_producer=produce_chunks,
                content_from_chunk=content_from_chunk,
            )

        return stream
