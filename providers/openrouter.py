from .base import GetPydanticStream, GetTextStream, Provider
from .openai import delta_content_from_chunk

from dotenv import load_dotenv
import instructor
from openai import AsyncOpenAI, api_key
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from dataclasses import dataclass
import os
from pydantic import BaseModel

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
instructor_client = instructor.from_openai(client)


@dataclass(frozen=True, kw_only=True)
class OpenRouterModelMetadata:
    pass


async def produce_raw_chunks(request: "Provider.TextStream.Request"):
    return await client.chat.completions.create(
        model=request.model,
        messages=request.messages,
        stream=True,
    )


async def produce_pydantic_models[ModelT: BaseModel](
    request: "Provider.PydanticStream.Request[ModelT]",
):
    return instructor_client.chat.completions.create_partial(
        response_model=request.type,
        model=request.model,
        messages=request.messages,
    )


class OpenRouterProvider(Provider):
    async def try_prepare_text_stream(self, request) -> GetTextStream | None:
        if not request.model.startswith(("qwen", "moonshot", "openrouter/")):
            return None

        async def stream():
            return Provider.TextStream.FromChunks(
                request,
                raw_chunk_producer=produce_raw_chunks,
                delta_content_from_chunk=delta_content_from_chunk,
            )

        return stream

    async def try_prepare_pydantic_stream[ModelT: BaseModel](
        self,
        request: "Provider.PydanticStream.Request[ModelT]",
    ) -> GetPydanticStream[ModelT] | None:
        if not request.model.startswith(("qwen", "moonshot", "openrouter/")):
            return None

        async def stream():
            return Provider.PydanticStream.FromModels(
                request,
                model_producer=produce_pydantic_models,
            )

        return stream
