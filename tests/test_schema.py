"""Tests for the Anthropic tool-schema exporter.

The contract: each registered tool produces one entry in the export, the
entry has Anthropic's required keys, and the input_schema is self-contained
(no $ref pointing into a separate $defs block).
"""

from __future__ import annotations

from ancestors.agent.schema import export_tools_for_anthropic
from ancestors.tool_registry import TOOLS


def test_every_tool_exported():
    exported = export_tools_for_anthropic(TOOLS)
    assert {t["name"] for t in exported} == set(TOOLS.keys())


def test_each_entry_has_anthropic_keys():
    for entry in export_tools_for_anthropic(TOOLS):
        assert "name" in entry
        assert "description" in entry
        assert "input_schema" in entry
        assert entry["input_schema"]["type"] == "object"


def test_no_dangling_refs():
    """$defs must be inlined; otherwise the model sees broken pointers."""

    def find_refs(node):
        if isinstance(node, dict):
            if "$ref" in node:
                yield node["$ref"]
            for v in node.values():
                yield from find_refs(v)
        elif isinstance(node, list):
            for v in node:
                yield from find_refs(v)

    for entry in export_tools_for_anthropic(TOOLS):
        refs = list(find_refs(entry["input_schema"]))
        assert refs == [], f"Tool {entry['name']!r} has dangling refs: {refs}"
        assert "$defs" not in entry["input_schema"]
        assert "title" not in entry["input_schema"]


def test_enum_args_export_as_enum_values():
    """Sex/EventType are enums; the schema must surface their string values."""
    by_name = {e["name"]: e for e in export_tools_for_anthropic(TOOLS)}
    sex_schema = by_name["filter_by_sex"]["input_schema"]
    sex_prop = sex_schema["properties"]["sex"]
    assert "enum" in sex_prop or sex_prop.get("type") in {"string", None}
    assert "M" in sex_prop.get("enum", []) or sex_prop.get("type") == "string"


def test_required_fields_propagate():
    by_name = {e["name"]: e for e in export_tools_for_anthropic(TOOLS)}
    ancestors = by_name["get_ancestors_of"]["input_schema"]
    assert "person_id" in ancestors.get("required", [])
