"""SOTOPIA's structured request with one Surface5 content repair."""

from __future__ import annotations

import os
from typing import Any

import gin
from pydantic import BaseModel
from sotopia.generation_utils import PydanticOutputParser
from sotopia.generation_utils import generate as sotopia_generation
from sotopia.utils import format_docstring

_CONTENT_REPAIR_TEMPLATE = """
    The action below failed content validation. Correct only the invalid content. For speech, extract the spoken dialogue and omit stage directions, while keeping every spoken word unchanged. Return a JSON action matching the schema; do not merely remove formatting symbols.
    Original string: {ill_formed_output}

    Format instructions: {format_instructions}

    Please only generate the JSON:
    """


async def generate_structured_action[Result: BaseModel](
    *,
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: PydanticOutputParser[Result],
    temperature: float | None,
    context: dict[str, Any] | None = None,
) -> Result:
    """Keep the first SOTOPIA request unchanged; repair only a parse failure.

    The parser owns action validation and repair instructions. API failures and
    cancellation propagate, and a failed repair never triggers another call.
    """
    bindings = gin.get_bindings(sotopia_generation.agenerate)
    repair_model = (
        bindings.get("bad_output_process_model")
        or sotopia_generation.DEFAULT_BAD_OUTPUT_PROCESS_MODEL
    )
    values = dict(input_values)
    if "format_instructions" not in values:
        values["format_instructions"] = output_parser.get_format_instructions()
    content = format_docstring(template)
    for key, value in values.items():
        content = content.replace(f"{{{key}}}", str(value))

    if model_name.startswith("custom"):
        base_url = model_name.split("@")[1]
        api_key = os.environ.get("CUSTOM_API_KEY", "EMPTY")
        model_name = model_name.split("@")[0].replace("custom/", "openai/")
    else:
        base_url = None
        api_key = None
        supported_params = sotopia_generation.get_supported_openai_params(
            model=model_name
        )
        assert supported_params is not None
        assert "response_format" in supported_params, (
            "response_format is not supported in this model"
        )
        assert sotopia_generation.supports_response_schema(model=model_name), (
            "response_schema is not supported in this model"
        )

    completion_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "response_format": sotopia_generation._build_json_schema_response_format(
            output_parser.pydantic_object
        ),
        "drop_params": True,
        "base_url": base_url,
        "api_key": api_key,
    }
    if temperature is not None:
        completion_kwargs["temperature"] = temperature
    response = await sotopia_generation.acompletion(**completion_kwargs)
    result = sotopia_generation._strip_thinking_tags(
        response.choices[0].message.content
    )
    try:
        parsed = output_parser.parse(result, context=context)
    except Exception:  # noqa: BLE001 - Match SOTOPIA's one parse-error repair.
        repaired = await _repair_action(
            result, output_parser, repair_model, base_url=base_url
        )
        parsed = output_parser.parse(repaired, context=context)
    return parsed


async def _repair_action[Result: BaseModel](
    result: str | None,
    output_parser: PydanticOutputParser[Result],
    model_name: str,
    *,
    base_url: str | None,
) -> str:
    # SOTOPIA's formatter rejects None before making a repair request.
    if result is None:
        raise ValueError("Response content is None")
    content = _CONTENT_REPAIR_TEMPLATE.format(
        ill_formed_output=result,
        format_instructions=output_parser.get_format_instructions(),
    )
    api_key = None
    if model_name.startswith("custom"):
        if "@" in model_name:
            base_url = model_name.split("@", 1)[1]
        model_name = model_name.split("@", 1)[0].replace("custom/", "openai/")
        api_key = os.environ.get("CUSTOM_API_KEY", "EMPTY")
    elif base_url is not None:
        api_key = os.environ.get("CUSTOM_API_KEY", "EMPTY")

    response = await sotopia_generation.acompletion(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        response_format=sotopia_generation._build_json_schema_response_format(
            output_parser.pydantic_object
        ),
        drop_params=True,
        base_url=base_url,
        api_key=api_key,
    )
    repaired = sotopia_generation._strip_thinking_tags(
        response.choices[0].message.content
    )
    if repaired is None:
        raise ValueError("Response content is None")
    return repaired
