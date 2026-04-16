"""Domain models for codegraph."""

import json
from enum import StrEnum

from pydantic import BaseModel, model_validator


class NodeType(StrEnum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class EdgeType(StrEnum):
    CONTAINS = "contains"
    IMPORTS = "imports"
    INHERITS = "inherits"
    CALLS = "calls"
    DECORATES = "decorates"


class GraphNode(BaseModel):
    node_id: str
    node_type: NodeType
    name: str
    file_path: str
    line_start: int
    line_end: int | None = None
    docstring: str | None = None
    decorators: list[str] = []
    parameters: list[str] = []
    bases: list[str] = []
    module_path: str = ""

    @model_validator(mode="before")
    @classmethod
    def _parse_json_lists(cls, data: dict) -> dict:
        data = dict(data)
        for field in ("decorators", "parameters", "bases"):
            if isinstance(data.get(field), str):
                data[field] = json.loads(data[field] or "[]")
        return data


class GraphEdge(BaseModel):
    source: str
    target: str
    edge_type: EdgeType
    file_path: str | None = None
    line: int | None = None


class RepoInfo(BaseModel):
    repo_id: str
    repo_path: str
    repo_url: str | None = None
    name: str
    indexed_at: str
    file_count: int
    node_count: int
    edge_count: int
