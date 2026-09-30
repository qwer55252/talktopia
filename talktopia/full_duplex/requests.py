"""One SOTOPIA-compatible structured request, without LLM output repair."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from pydantic import BaseModel
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from sotopia.generation_utils import PydanticOutputParser
from sotopia.generation_utils import generate as sotopia_generation
from sotopia.utils import format_docstring


async def generate_structured_action[Result: BaseModel](
    *,
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: PydanticOutputParser[Result],
    temperature: float | None,
    context: dict[str, Any] | None = None,
    responses: list[str | None] | None = None,
) -> Result:
    """Parse locally; propagate invalid output without another model call."""
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
        # This function owns the single transport retry.
        "num_retries": 0,
        "max_retries": 0,
        "base_url": base_url,
        "api_key": api_key,
    }
    if temperature is not None:
        completion_kwargs["temperature"] = temperature
    for attempt in range(2):
        try:
            response = await sotopia_generation.acompletion(**completion_kwargs)
            break
        except (
            APIConnectionError,
            InternalServerError,
            RateLimitError,
            ServiceUnavailableError,
            Timeout,
        ):
            if attempt == 1:
                raise
            await asyncio.sleep(0.25)
    if responses is not None:
        responses.append(response.choices[0].message.content)
    result = sotopia_generation._strip_thinking_tags(
        response.choices[0].message.content
    )
    if result is None:
        raise ValueError("Response content is None")
    return output_parser.parse(result, context=context)
