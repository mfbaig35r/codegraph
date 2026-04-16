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
    SEMANTIC_SIMILARITY = "semantic_similarity"


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
    summary: str | None = None
    cluster_id: str | None = None

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
    weight: float | None = None


class RepoInfo(BaseModel):
    repo_id: str
    repo_path: str
    repo_url: str | None = None
    name: str
    indexed_at: str
    file_count: int
    node_count: int
    edge_count: int


class ClusterInfo(BaseModel):
    cluster_id: str
    label: str
    description: str | None = None
    member_ids: list[str] = []
    member_count: int = 0

    @model_validator(mode="before")
    @classmethod
    def _parse_member_ids(cls, data: dict) -> dict:
        data = dict(data)
        if isinstance(data.get("member_ids"), str):
            data["member_ids"] = json.loads(data["member_ids"] or "[]")
        if "member_count" not in data or data["member_count"] == 0:
            data["member_count"] = len(data.get("member_ids", []))
        return data


class ImpactReport(BaseModel):
    target_node_id: str
    affected_nodes: list[str]
    risk_level: str
    explanation: str
    impact_chain: list[dict] = []


class NarrativeSection(BaseModel):
    title: str
    summary: str
    key_nodes: list[str] = []
    relationships: str = ""


class CostSummary(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    estimated_cost_usd: float = 0.0
