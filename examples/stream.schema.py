import asyncio
import json

from release import Provider
from release.providers.openai import OpenAIProvider


openai = OpenAIProvider()


SCHEMA = {
    "title": "RecipeCard",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "prep_minutes": {"type": "integer"},
        "ingredients": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["name", "prep_minutes", "ingredients"],
}


async def main() -> None:
    stream = await Provider.SchemaStream.Select(
        Provider.SchemaStream.Request(
            model_metadata=[OpenAIProvider.ModelMetadata()],
            model="gpt-4o-mini",
            schema=json.dumps(SCHEMA),
            messages=[
                {"role": "system", "content": "Return valid structured data."},
                {
                    "role": "user",
                    "content": "Create a simple recipe card for overnight oats.",
                },
            ],
        ),
        openai,
    )

    final_model = None

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
