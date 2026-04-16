"""Tests for codegraph.store."""

from pathlib import Path

import pytest

from codegraph.models import EdgeType, GraphEdge, GraphNode, NodeType, RepoInfo
from codegraph.store import GraphStore


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    return GraphStore(tmp_path / "test.db")


def _sample_repo() -> tuple[RepoInfo, list[GraphNode], list[GraphEdge]]:
    repo = RepoInfo(
        repo_id="test_repo",
        repo_path="/tmp/test",
        name="test",
        indexed_at="2026-01-01T00:00:00",
        file_count=2,
        node_count=5,
        edge_count=4,
    )
    nodes = [
        GraphNode(
            node_id="mod_a", node_type=NodeType.MODULE, name="mod_a",
            file_path="mod_a.py", line_start=1, module_path="mod_a",
        ),
        GraphNode(
            node_id="mod_a.func_a", node_type=NodeType.FUNCTION, name="func_a",
            file_path="mod_a.py", line_start=5, module_path="mod_a",
        ),
        GraphNode(
            node_id="mod_a.MyClass", node_type=NodeType.CLASS, name="MyClass",
            file_path="mod_a.py", line_start=10, bases=["Base"], module_path="mod_a",
        ),
        GraphNode(
            node_id="mod_a.MyClass.method", node_type=NodeType.METHOD, name="method",
            file_path="mod_a.py", line_start=12, module_path="mod_a",
        ),
        GraphNode(
            node_id="mod_b", node_type=NodeType.MODULE, name="mod_b",
            file_path="mod_b.py", line_start=1, module_path="mod_b",
        ),
    ]
    edges = [
        GraphEdge(source="mod_a", target="mod_a.func_a", edge_type=EdgeType.CONTAINS),
        GraphEdge(source="mod_a", target="mod_a.MyClass", edge_type=EdgeType.CONTAINS),
        GraphEdge(
            source="mod_a.MyClass", target="mod_a.MyClass.method",
            edge_type=EdgeType.CONTAINS,
        ),
        GraphEdge(
            source="mod_a.MyClass.method", target="mod_a.func_a",
            edge_type=EdgeType.CALLS, line=15,
        ),
    ]
    return repo, nodes, edges


def test_save_and_get_repo(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    result = store.get_repo("test_repo")
    assert result is not None
    assert result.name == "test"
    assert result.node_count == 5


def test_list_repos(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    repos = store.list_repos()
    assert len(repos) == 1
    assert repos[0].repo_id == "test_repo"


def test_delete_repo(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    assert store.delete_repo("test_repo") is True
    assert store.get_repo("test_repo") is None
    assert store.delete_repo("test_repo") is False


def test_query_nodes_all(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    result = store.query_nodes("test_repo")
    assert len(result) == 5


def test_query_nodes_by_type(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    classes = store.query_nodes("test_repo", node_type="class")
    assert len(classes) == 1
    assert classes[0].name == "MyClass"


def test_query_nodes_by_name(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    result = store.query_nodes("test_repo", name="func")
    assert len(result) == 1
    assert result[0].name == "func_a"


def test_get_edges_both(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    result = store.get_edges("test_repo", "mod_a.MyClass")
    assert len(result["incoming"]) >= 1  # contains from mod_a
    assert len(result["outgoing"]) >= 1  # contains to method


def test_get_edges_by_type(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    result = store.get_edges("test_repo", "mod_a.MyClass.method", edge_type="calls")
    assert len(result["outgoing"]) == 1
    assert result["outgoing"][0].target == "mod_a.func_a"


def test_get_subgraph(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    sub_nodes, sub_edges = store.get_subgraph("test_repo", "mod_a.MyClass", depth=1)
    node_ids = {n.node_id for n in sub_nodes}
    assert "mod_a.MyClass" in node_ids
    assert "mod_a.MyClass.method" in node_ids
    assert "mod_a" in node_ids


def test_get_subgraph_missing_node(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    sub_nodes, sub_edges = store.get_subgraph("test_repo", "nonexistent")
    assert sub_nodes == []
    assert sub_edges == []


def test_find_path(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    path, path_edges = store.find_path("test_repo", "mod_a", "mod_a.func_a")
    assert len(path) >= 2
    assert path[0] == "mod_a"
    assert path[-1] == "mod_a.func_a"


def test_find_path_no_connection(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    path, path_edges = store.find_path("test_repo", "mod_a", "mod_b")
    # mod_b is disconnected, so no path
    assert path == []


def test_export_graph_d3(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    result = store.export_graph("test_repo", format="d3")
    assert "nodes" in result
    assert "links" in result
    assert "metadata" in result
    assert result["metadata"]["node_count"] == 5
    assert result["metadata"]["edge_count"] == 4

    # D3 format should have id, name, type, file, group, size
    d3_node = result["nodes"][0]
    assert "id" in d3_node
    assert "type" in d3_node
    assert "group" in d3_node
    assert "size" in d3_node


def test_export_graph_full(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    result = store.export_graph("test_repo", format="full")
    d3_node = result["nodes"][0]
    assert "docstring" in d3_node
    assert "decorators" in d3_node
    assert "parameters" in d3_node


def test_get_stats(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    stats = store.get_stats("test_repo")
    assert stats["name"] == "test"
    assert stats["nodes_by_type"]["module"] == 2
    assert stats["nodes_by_type"]["class"] == 1
    assert stats["nodes_by_type"]["function"] == 1
    assert stats["nodes_by_type"]["method"] == 1
    assert stats["edges_by_type"]["contains"] == 3
    assert stats["edges_by_type"]["calls"] == 1
    assert len(stats["top_connected"]) > 0
    assert stats["avg_connections"] > 0


def test_get_subgraph_skips_stub_nodes(store: GraphStore) -> None:
    """Edges to undefined targets (e.g. builtins) should not crash get_subgraph."""
    repo, nodes, edges = _sample_repo()
    # Add an edge pointing to a stub (no matching node definition)
    edges.append(GraphEdge(
        source="mod_a.MyClass", target="builtin_stub",
        edge_type=EdgeType.INHERITS,
    ))
    store.save_repo(repo, nodes, edges)

    sub_nodes, sub_edges = store.get_subgraph("test_repo", "mod_a.MyClass", depth=1)
    node_ids = {n.node_id for n in sub_nodes}
    assert "mod_a.MyClass" in node_ids
    # builtin_stub should be skipped, not cause a validation error
    assert "builtin_stub" not in node_ids


def test_export_graph_skips_stub_nodes(store: GraphStore) -> None:
    """Stub nodes should not appear in the export."""
    repo, nodes, edges = _sample_repo()
    edges.append(GraphEdge(
        source="mod_a", target="external_lib",
        edge_type=EdgeType.IMPORTS,
    ))
    store.save_repo(repo, nodes, edges)

    result = store.export_graph("test_repo")
    node_ids = {n["id"] for n in result["nodes"]}
    assert "external_lib" not in node_ids


def test_save_repo_replaces_existing(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    # Save again with different data
    repo2 = RepoInfo(
        repo_id="test_repo", repo_path="/tmp/test", name="test_v2",
        indexed_at="2026-02-01", file_count=1, node_count=1, edge_count=0,
    )
    store.save_repo(repo2, nodes[:1], [])

    result = store.get_repo("test_repo")
    assert result is not None
    assert result.name == "test_v2"
    assert store.query_nodes("test_repo") == [nodes[0]]
