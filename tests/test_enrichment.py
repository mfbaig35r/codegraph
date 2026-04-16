"""Tests for codegraph.enrichment — mocked LLM calls."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codegraph.models import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    RepoInfo,
)
from codegraph.store import GraphStore


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    return GraphStore(tmp_path / "test.db")


def _sample_repo_with_nodes() -> tuple[RepoInfo, list[GraphNode], list[GraphEdge]]:
    repo = RepoInfo(
        repo_id="test_repo",
        repo_path="/tmp/test",
        name="test",
        indexed_at="2026-01-01T00:00:00",
        file_count=1,
        node_count=4,
        edge_count=3,
    )
    nodes = [
        GraphNode(
            node_id="mod", node_type=NodeType.MODULE, name="mod",
            file_path="mod.py", line_start=1, module_path="mod",
        ),
        GraphNode(
            node_id="mod.func_a", node_type=NodeType.FUNCTION, name="func_a",
            file_path="mod.py", line_start=5, module_path="mod",
            docstring="Does thing A",
        ),
        GraphNode(
            node_id="mod.func_b", node_type=NodeType.FUNCTION, name="func_b",
            file_path="mod.py", line_start=10, module_path="mod",
        ),
        GraphNode(
            node_id="mod.MyClass", node_type=NodeType.CLASS, name="MyClass",
            file_path="mod.py", line_start=15, module_path="mod",
        ),
    ]
    edges = [
        GraphEdge(
            source="mod", target="mod.func_a", edge_type=EdgeType.CONTAINS,
        ),
        GraphEdge(
            source="mod", target="mod.func_b", edge_type=EdgeType.CONTAINS,
        ),
        GraphEdge(
            source="mod.func_b", target="mod.func_a", edge_type=EdgeType.CALLS,
        ),
    ]
    return repo, nodes, edges


def test_enrich_repo_generates_summaries(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo_with_nodes()
    store.save_repo(repo, nodes, edges)

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.reset_cost_tracking.return_value = None
    mock_llm.get_cost_summary.return_value = MagicMock(
        input_tokens=100, output_tokens=50,
        embedding_tokens=0, estimated_cost_usd=0.001,
        model_dump=lambda: {
            "input_tokens": 100, "output_tokens": 50,
            "embedding_tokens": 0, "estimated_cost_usd": 0.001,
        },
    )
    # batch_complete is called twice: once for summaries, once for clusters
    mock_llm.batch_complete.side_effect = [
        # Call 1: summaries
        [
            {
                "content": json.dumps([
                    {"node_id": "mod.func_a", "summary": "Does thing A"},
                    {"node_id": "mod.func_b", "summary": "Calls func_a"},
                    {"node_id": "mod.MyClass", "summary": "A sample class"},
                ])
            }
        ],
        # Call 2: cluster labeling
        [
            {
                "content": json.dumps({
                    "label": "Core Module",
                    "description": "Main module functions",
                })
            }
        ],
    ]
    mock_llm.embed.return_value = []

    with patch("codegraph.enrichment.get_llm_client", return_value=mock_llm):
        from codegraph.enrichment import _impl_enrich_repo
        result = _impl_enrich_repo("test_repo", store)

    assert result["status"] == "enriched"
    assert result["summaries_generated"] == 3

    # Verify summaries are stored
    nodes_after = store.query_nodes("test_repo", name="func_a")
    assert nodes_after[0].summary == "Does thing A"


def test_enrich_repo_idempotent(store: GraphStore) -> None:
    """Nodes with existing summaries should be skipped."""
    repo, nodes, edges = _sample_repo_with_nodes()
    store.save_repo(repo, nodes, edges)
    # Pre-set a summary
    store.update_node_summaries("test_repo", [("mod.func_a", "Existing summary")])

    # Should only get 2 nodes without summaries
    without = store.get_nodes_without_summary("test_repo")
    assert len(without) == 2  # func_b and MyClass (module excluded)


def test_get_clusters_empty(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo_with_nodes()
    store.save_repo(repo, nodes, edges)

    from codegraph.enrichment import _impl_get_clusters
    result = _impl_get_clusters("test_repo", store)
    assert result["count"] == 0


def test_enrich_repo_no_api_key(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo_with_nodes()
    store.save_repo(repo, nodes, edges)

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = False

    with patch("codegraph.enrichment.get_llm_client", return_value=mock_llm):
        from codegraph.enrichment import _impl_enrich_repo
        result = _impl_enrich_repo("test_repo", store)

    assert "error" in result
    assert "OPENAI_API_KEY" in result["error"]
