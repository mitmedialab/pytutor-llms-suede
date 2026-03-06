from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from typing import Callable, NamedTuple

type Msg = ChatCompletionMessageParam


class TextChunk(NamedTuple):
    delta: str
    accumulated: str
    i: int


type OnText = Callable[[TextChunk], str]
