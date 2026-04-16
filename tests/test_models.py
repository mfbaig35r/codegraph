"""Tests for codegraph.models."""


from codegraph.models import EdgeType, GraphEdge, GraphNode, NodeType, RepoInfo


def test_node_type_values():
    assert NodeType.MODULE == "module"
    assert NodeType.CLASS == "class"
    assert NodeType.FUNCTION == "function"
    assert NodeType.METHOD == "method"


def test_edge_type_values():
    assert EdgeType.CONTAINS == "contains"
    assert EdgeType.IMPORTS == "imports"
    assert EdgeType.INHERITS == "inherits"
    assert EdgeType.CALLS == "calls"
    assert EdgeType.DECORATES == "decorates"


def test_graph_node_basic():
    node = GraphNode(
        node_id="pkg.module.func",
        node_type=NodeType.FUNCTION,
        name="func",
        file_path="src/pkg/module.py",
        line_start=10,
    )
    assert node.node_id == "pkg.module.func"
    assert node.decorators == []
    assert node.parameters == []
    assert node.bases == []


def test_graph_node_json_list_parsing():
    """JSON strings in list fields should be parsed."""
    data = {
        "node_id": "pkg.Cls",
        "node_type": "class",
        "name": "Cls",
        "file_path": "pkg.py",
        "line_start": 1,
        "decorators": '["dataclass"]',
        "parameters": "[]",
        "bases": '["Base"]',
    }
    node = GraphNode.model_validate(data)
    assert node.decorators == ["dataclass"]
    assert node.bases == ["Base"]
    assert node.parameters == []


def test_graph_node_roundtrip():
    node = GraphNode(
        node_id="pkg.Cls.method",
        node_type=NodeType.METHOD,
        name="method",
        file_path="pkg.py",
        line_start=5,
        line_end=10,
        decorators=["staticmethod"],
        parameters=["self", "x: int"],
    )
    dumped = node.model_dump()
    restored = GraphNode.model_validate(dumped)
    assert restored == node


def test_graph_edge():
    edge = GraphEdge(
        source="a.b",
        target="c.d",
        edge_type=EdgeType.CALLS,
        line=42,
    )
    assert edge.source == "a.b"
    assert edge.edge_type == "calls"


def test_repo_info():
    repo = RepoInfo(
        repo_id="abc123",
        repo_path="/tmp/repo",
        name="repo",
        indexed_at="2026-01-01T00:00:00",
        file_count=10,
        node_count=50,
        edge_count=100,
    )
    assert repo.repo_url is None
    assert repo.file_count == 10
