from release.providers.base import Provider, GetTextStream

from openai import AsyncOpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


openai_client = AsyncOpenAI()


def content_from_openai_chunk(chunk: ChatCompletionChunk) -> str | None:
    if not getattr(chunk, "choices", None):
        return None

    if len(chunk.choices) == 0:
        return None

    content = chunk.choices[0].delta.content

    if content is None:
        return None

    return content


async def produce_chunks(request: "Provider.TextStream.Request"):
    return await openai_client.chat.completions.create(
        model=request.model,
        messages=request.messages,
        stream=True,
    )


class OpenAIProvider(Provider):
    async def try_prepare_text_stream(self, request) -> GetTextStream | None:
        if not request.model.startswith("gpt"):
            return None

        async def start():
            return Provider.TextStream.Stream(
                request,
                chunk_producer=produce_chunks,
                content_from_chunk=content_from_openai_chunk,
            )

        return start
