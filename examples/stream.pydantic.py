import asyncio

from pydantic import BaseModel

from release import Provider
from release.providers.openai import OpenAIProvider


class LessonSummary(BaseModel):
    topic: str
    key_points: list[str]
    difficulty: str


openai = OpenAIProvider()


async def main() -> None:
    stream = await Provider.PydanticStream.Select(
        Provider.PydanticStream.Request(
            model_metadata=[OpenAIProvider.ModelMetadata()],
            model="gpt-4o-mini",
            type=LessonSummary,
            messages=[
                {
                    "role": "system",
                    "content": "You produce compact educational summaries.",
                },
                {
                    "role": "user",
                    "content": (
                        "Return a lesson summary about latency benchmarking "
                        "with topic, 3 key_points, and difficulty."
                    ),
                },
            ],
        ),
        openai,
    )

    final_model: LessonSummary | None = None

    async for event in stream:
        if event.type == "error":
            print(f"\n[Error: {event.payload}]")
            return

        final_model = event.payload.current
        print(f"partial[{event.payload.i}]: {final_model.model_dump()}")

    if final_model is not None:
        print("\nfinal:")
        print(final_model.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
