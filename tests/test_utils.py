from pydantic import BaseModel

from release.common import Chunk
from release.utils import build_from_json_schema, compute_delta


def test_build_from_json_schema_returns_titled_model() -> None:
    schema = """
    {
      "title": "Person",
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
      },
      "required": ["name", "age"]
    }
    """

    model_type = build_from_json_schema(schema)
    parsed = model_type.model_validate({"name": "Ada", "age": 42})

    assert model_type.__name__ == "Person"
    assert parsed.model_dump() == {"name": "Ada", "age": 42}


def test_compute_delta_handles_nested_dicts_and_lists() -> None:
    previous = {
        "title": "v1",
        "metadata": {"count": 1, "tags": ["a"]},
        "items": [1, 2],
    }
    current = {
        "title": "v2",
        "metadata": {"count": 2, "tags": ["a"]},
        "items": [1, 2, 3],
        "new": True,
    }

    delta = compute_delta(current, previous)

    assert delta == {
        "title": "v2",
        "metadata": {"count": 2},
        "items": [1, 2, 3],
        "new": True,
    }


def test_compute_delta_returns_none_when_unchanged() -> None:
    data = {"a": 1, "nested": {"b": [1, 2, 3]}}
    assert compute_delta(data, data) is None


def test_chunk_pydantic_compute_delta_first_and_subsequent() -> None:
    class Demo(BaseModel):
        x: int
        y: str

    first = Chunk.Pydantic(current=Demo(x=1, y="a"), previous=None, i=0)
    second = Chunk.Pydantic(
        current=Demo(x=2, y="a"),
        previous=Demo(x=1, y="a"),
        i=1,
    )

    assert first.compute_delta() == {"x": 1, "y": "a"}
    assert second.compute_delta() == {"x": 2}
