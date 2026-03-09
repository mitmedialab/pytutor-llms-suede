from datamodel_code_generator import InputFileType, generate
from datamodel_code_generator.enums import DataModelType
from pydantic import BaseModel

import json


def build_from_json_schema(json_schema: str) -> type[BaseModel]:
    obj = json.loads(json_schema)
    title = obj.get("title")

    generated = generate(
        json_schema,
        input_file_type=InputFileType.JsonSchema,
        input_filename="example.json",
        output_model_type=DataModelType.PydanticV2BaseModel,
    )
    if not isinstance(generated, str) or not generated.strip():
        raise ValueError("Failed to generate pydantic model code from schema")

    namespace: dict[str, object] = {}
    exec(generated, namespace)

    if isinstance(title, str):
        model = namespace.get(title)
        if isinstance(model, type) and issubclass(model, BaseModel):
            return model

    fallback_models = [
        value
        for value in namespace.values()
        if isinstance(value, type)
        and issubclass(value, BaseModel)
        and value is not BaseModel
    ]
    if fallback_models:
        return fallback_models[0]

    raise ValueError("No pydantic model class found in generated code")


def compute_delta(current: object, previous: object) -> object | None:
    if isinstance(current, dict) and isinstance(previous, dict):
        delta: dict[str, object] = {}
        for key, current_value in current.items():
            if key not in previous:
                delta[key] = current_value
                continue
            nested_delta = compute_delta(current_value, previous[key])
            if nested_delta is not None:
                delta[key] = nested_delta
        return delta or None

    if isinstance(current, list) and isinstance(previous, list):
        return None if current == previous else current

    return None if current == previous else current
