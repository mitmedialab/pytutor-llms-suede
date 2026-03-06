from format import sse

from openai import OpenAIError
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

import json
import time
import asyncio
import logging
from typing import AsyncGenerator, cast, overload
from typing import Optional, List


logger = logging.getLogger(__name__)


@overload
async def stream(
    messages: List[ChatCompletionMessageParam],
    response_json_schema: str,
    *,
    model: str = default_model,
) -> AsyncIterable[BaseModel]: ...


@overload
async def stream(
    messages: List[ChatCompletionMessageParam], *, model: str = default_model
) -> AsyncStream[ChatCompletionChunk]: ...


async def stream(
    messages: List[ChatCompletionMessageParam],
    response_json_schema: str | None = None,
    *,
    model: str,
):
    return (
        await openai_client.structured_client.create_partial(
            build_from_json_schema(response_json_schema),
            model=model,
            stream=True,
            messages=messages,
        )
        if response_json_schema is not None
        else await openai_client.client.chat.completions.create(
            model=model, messages=messages, stream=True
        )
    )


async def stream_as_event_source(
    messages: List[ChatCompletionMessageParam],
    model: str = "gpt-4o",
    *,
    response_json_schema: Optional[str] = None,
):
    try:
        entry_time = time.time()

        def delta_time():
            return time.time() - entry_time

        time_to_first_msg = -1.0

        if messages is None or len(messages) == 0:
            yield sse("(Internal Error: No input found to generate response.)", "error")
            yield sse("done", event="done")
            return

        has_response_schema = response_json_schema is not None

        if not model.startswith(
            ("gpt", "gemini", "learnlm", "claude", "qwen", "moonshot")
        ):
            logger.warning(f"Unsupported model '{model}', defaulting to gpt-4o")
            model = "gpt-4o"

        if model.startswith("gpt"):
            logger.info(f"LLM: {model} model selected")
            message_stream = (
                stream(messages, response_json_schema, model=model)
                if has_response_schema
                else stream(messages, model=model)
            )

        elif model.startswith("gemini") or model.startswith("learnlm"):
            logger.info(f"LLM: {model} model selected")
            gemini_parts = [convert_raw_to_gemini_parts(msg) for msg in messages[1:]]
            system_prompt = messages[0].get("content", "")
            message_stream = stream_gemini(system_prompt, gemini_parts, model=model)

        elif model.startswith("claude"):
            logger.info(f"LLM: {model} model selected")
            converted_claude = [
                convert_to_claude_format(i) if i["role"] == "user" else i
                for i in messages[1:]
            ]
            system_prompt = messages[0].get("content", "")
            message_stream = stream_claude(system_prompt, converted_claude, model=model)

        elif model.startswith("qwen") or model.startswith("moonshot"):
            logger.info(f"LLM: {model} model selected")
            message_stream = stream_openrouter(messages, model=model)

        ChatMessage.Is_Canceled[response_id] = False

        with log_operation_time(f"Chat {response_id}: Stream Messages"):
            if has_response_schema:
                full_answer = ""
                async for partial in await message_stream:
                    time_to_first_msg = (
                        time_to_first_msg if time_to_first_msg > 0 else delta_time()
                    )
                    full_answer = partial.model_dump_json()
                    yield sse(full_answer)
                    if ChatMessage.Is_Canceled.get(response_id, False):
                        break
            else:
                collected_chunks = []

                if (
                    model.startswith("gpt")
                    or model.startswith("qwen")
                    or model.startswith("moonshot")
                ):
                    async for chunk in await message_stream:
                        time_to_first_msg = (
                            time_to_first_msg if time_to_first_msg > 0 else delta_time()
                        )
                        c = cast(ChatCompletionChunk, chunk)

                        if not getattr(c, "choices", None):
                            continue

                        content = c.choices[0].delta.content
                        if content is not None:
                            collected_chunks.append(content)
                            yield sse(content)
                        if ChatMessage.Is_Canceled.get(response_id, False):
                            break
                    full_answer = "".join(collected_chunks)
                elif model.startswith("gemini") or model.startswith("learnlm"):
                    async for chunk in await message_stream:
                        time_to_first_msg = (
                            time_to_first_msg if time_to_first_msg > 0 else delta_time()
                        )
                        content = chunk.text
                        if content is not None:
                            collected_chunks.append(content)
                            yield sse(content)
                        if ChatMessage.Is_Canceled.get(response_id, False):
                            break
                    full_answer = "".join(collected_chunks)

                elif model.startswith("claude"):
                    async for chunk in await message_stream:
                        time_to_first_msg = (
                            time_to_first_msg if time_to_first_msg > 0 else delta_time()
                        )
                        for part in chunk:
                            if isinstance(part[1], TextDelta):
                                content = part[1].text
                                collected_chunks.append(content)
                                yield sse(content)
                        if ChatMessage.Is_Canceled.get(response_id, False):
                            break
                    full_answer = "".join(collected_chunks)

            ChatMessage.Drop_Intermediate_Values_For_Response(response_id)

            yield sse("done", event="done")

            ChatMessage.Is_Canceled.pop(response_id)

            total_generation_time = delta_time()
    except OpenAIError as e:
        yield sse(f"{type(e).__name__}: {str(e)}", "error")
    except Exception as e:
        yield sse(f"Internal Error: {str(e)}", "error")
        raise e
