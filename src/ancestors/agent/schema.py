"""Convert the dispatch registry into Anthropic-format tool definitions.

Anthropic's Messages API expects each tool as:

    {"name": str, "description": str, "input_schema": JSONSchema}

Pydantic gives us a JSON schema for free via `model_json_schema()`. We strip
the Pydantic-specific bits ('title', '$defs' resolved inline) and surface the
result. The agent passes this list as the `tools` parameter — and *only* the
tools listed here are exposed to the model.

This is the single point where dispatcher safety meets the model surface: a
tool that isn't in the registry can't be in this export, can't be in the
`tools` parameter, can't be named by the model's tool_use blocks.
"""

from __future__ import annotations

from typing import Any

from ancestors.dispatch import Tool


def export_tools_for_anthropic(tools: dict[str, Tool]) -> list[dict[str, Any]]:
    """Return tool definitions in Anthropic's API shape."""
    return [_tool_def(tools[name]) for name in sorted(tools)]


def _tool_def(tool: Tool) -> dict[str, Any]:
    raw_schema = tool.args_model.model_json_schema()
    input_schema = _normalize_schema(raw_schema)
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": input_schema,
    }


def _normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip Pydantic title fields; inline $defs (enums); keep the shape Anthropic expects.

    Anthropic's tool input_schema is plain JSONSchema and does not need title
    annotations. We also resolve $defs into inline $ref-free shapes so the
    schema is self-contained — simpler for the model to read, no resolution
    pass needed by the API.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and node["$ref"].startswith("#/$defs/"):
                key = node["$ref"].split("/")[-1]
                return resolve(defs.get(key, {}))
            return {
                k: resolve(v)
                for k, v in node.items()
                if k not in {"title", "$defs"}
            }
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    resolved = resolve(schema)
    # The top-level schema must declare type=object for Anthropic.
    if "type" not in resolved:
        resolved["type"] = "object"
    return resolved
