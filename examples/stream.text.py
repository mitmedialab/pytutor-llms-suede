from release import Provider
from release.providers.openai import OpenAIProvider

import asyncio


openai = OpenAIProvider()


async def main() -> None:
    stream = await Provider.TextStream.Select(
        Provider.TextStream.Request(
            model_metadata=[OpenAIProvider.ModelMetadata()],
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {
                    "role": "user",
                    "content": "In one sentence, explain what streaming responses are.",
                },
            ],
        ),
        openai,
    )

    async for item in stream:
        if item.type == "chunk":
            chunk, _ = item
            print(chunk.delta, end="", flush=True)
        else:
            print(f"\n[Error: {item.payload}]")


if __name__ == "__main__":
    asyncio.run(main())
