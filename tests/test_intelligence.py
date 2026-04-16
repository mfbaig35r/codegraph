"""Tests for codegraph.intelligence — mocked LLM calls."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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
            summary="Does thing A",
        ),
        GraphNode(
            node_id="mod.func_b", node_type=NodeType.FUNCTION, name="func_b",
            file_path="mod.py", line_start=10, module_path="mod",
            summary="Calls func_a",
        ),
        GraphNode(
            node_id="mod.MyClass", node_type=NodeType.CLASS, name="MyClass",
            file_path="mod.py", line_start=15, module_path="mod",
        ),
    ]
    edges = [
        GraphEdge(source="mod", target="mod.func_a", edge_type=EdgeType.CONTAINS),
        GraphEdge(source="mod", target="mod.func_b", edge_type=EdgeType.CONTAINS),
        GraphEdge(source="mod.func_b", target="mod.func_a", edge_type=EdgeType.CALLS),
    ]
    return repo, nodes, edges


def test_ask_returns_answer(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.reset_cost_tracking.return_value = None
    mock_llm.get_cost_summary.return_value = MagicMock(
        model_dump=lambda: {
            "input_tokens": 100, "output_tokens": 50,
            "embedding_tokens": 0, "estimated_cost_usd": 0.001,
        },
    )
    # LLM gives a direct answer (no tool calls)
    mock_llm.complete.return_value = {
        "content": "func_a does thing A and is called by func_b."
    }

    with patch("codegraph.intelligence.get_llm_client", return_value=mock_llm):
        from codegraph.intelligence import _impl_ask
        result = _impl_ask("test_repo", "what does func_a do?", store)

    assert "answer" in result
    assert "func_a" in result["answer"]
    assert result["repo_id"] == "test_repo"


def test_ask_with_tool_calls(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.reset_cost_tracking.return_value = None
    mock_llm.get_cost_summary.return_value = MagicMock(
        model_dump=lambda: {
            "input_tokens": 200, "output_tokens": 100,
            "embedding_tokens": 0, "estimated_cost_usd": 0.002,
        },
    )

    # First call: LLM wants to use a tool
    # Second call: LLM gives final answer
    mock_llm.complete.side_effect = [
        {
            "content": "",
            "tool_calls": [{
                "id": "tc_1",
                "function": {
                    "name": "query_nodes",
                    "arguments": json.dumps({"name": "func_a"}),
                },
            }],
        },
        {"content": "func_a is a function that does thing A."},
    ]

    with patch("codegraph.intelligence.get_llm_client", return_value=mock_llm):
        from codegraph.intelligence import _impl_ask
        result = _impl_ask("test_repo", "tell me about func_a", store)

    assert result["answer"] == "func_a is a function that does thing A."
    assert "mod.func_a" in result["sources"]


def test_ask_no_api_key(store: GraphStore) -> None:
    mock_llm = MagicMock()
    mock_llm.is_available.return_value = False

    with patch("codegraph.intelligence.get_llm_client", return_value=mock_llm):
        from codegraph.intelligence import _impl_ask
        result = _impl_ask("test_repo", "hello", store)

    assert "error" in result


def test_analyze_impact(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.reset_cost_tracking.return_value = None
    mock_llm.get_cost_summary.return_value = MagicMock(
        model_dump=lambda: {
            "input_tokens": 100, "output_tokens": 50,
            "embedding_tokens": 0, "estimated_cost_usd": 0.001,
        },
    )
    mock_llm.complete.return_value = {
        "content": json.dumps({
            "risk_level": "medium",
            "explanation": "func_b directly calls func_a.",
        }),
    }

    with patch("codegraph.intelligence.get_llm_client", return_value=mock_llm):
        from codegraph.intelligence import _impl_analyze_impact
        result = _impl_analyze_impact("test_repo", "mod.func_a", store)

    assert result["affected_count"] >= 1
    assert result["risk_level"] == "medium"
    assert "mod.func_b" in result["affected_nodes"]


def test_analyze_impact_no_callers(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.reset_cost_tracking.return_value = None
    mock_llm.get_cost_summary.return_value = MagicMock(
        model_dump=lambda: {
            "input_tokens": 0, "output_tokens": 0,
            "embedding_tokens": 0, "estimated_cost_usd": 0.0,
        },
    )

    with patch("codegraph.intelligence.get_llm_client", return_value=mock_llm):
        from codegraph.intelligence import _impl_analyze_impact
        # func_b has no callers
        result = _impl_analyze_impact("test_repo", "mod.func_b", store)

    assert result["affected_count"] == 0
    assert result["risk_level"] == "low"


def test_narrate(store: GraphStore) -> None:
    repo, nodes, edges = _sample_repo()
    store.save_repo(repo, nodes, edges)

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.reset_cost_tracking.return_value = None
    mock_llm.get_cost_summary.return_value = MagicMock(
        model_dump=lambda: {
            "input_tokens": 300, "output_tokens": 200,
            "embedding_tokens": 0, "estimated_cost_usd": 0.003,
        },
    )
    mock_llm.complete.return_value = {
        "content": json.dumps({
            "title": "Test Project Overview",
            "sections": [
                {
                    "title": "Core Module",
                    "summary": "The mod module contains two functions.",
                    "key_nodes": ["mod.func_a", "mod.func_b"],
                    "relationships": "func_b calls func_a",
                }
            ],
        }),
    }

    with patch("codegraph.intelligence.get_llm_client", return_value=mock_llm):
        from codegraph.intelligence import _impl_narrate
        result = _impl_narrate("test_repo", store)

    assert result["title"] == "Test Project Overview"
    assert len(result["sections"]) == 1
    assert "Core Module" in result["sections"][0]["title"]
