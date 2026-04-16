"""Tests for codegraph.server."""

from pathlib import Path
from unittest.mock import patch

import codegraph.server as server_module
from codegraph.store import GraphStore

FIXTURES = Path(__file__).parent / "fixtures"


def test_impl_index_repo_local(tmp_path: Path) -> None:
    test_store = GraphStore(tmp_path / "test.db")
    with patch.object(server_module, "store", test_store), \
         patch.object(server_module, "DATA_DIR", tmp_path):
        result = server_module._impl_index_repo(str(FIXTURES))

    assert result["status"] == "indexed"
    assert result["node_count"] > 0
    assert result["edge_count"] > 0
    assert result["file_count"] > 0


def test_impl_index_repo_bad_path() -> None:
    result = server_module._impl_index_repo("/nonexistent/path")
    assert "error" in result


def test_impl_list_repos(tmp_path: Path) -> None:
    test_store = GraphStore(tmp_path / "test.db")
    with patch.object(server_module, "store", test_store), \
         patch.object(server_module, "DATA_DIR", tmp_path):
        server_module._impl_index_repo(str(FIXTURES))
        result = server_module._impl_list_repos()

    assert result["count"] == 1


def test_impl_delete_repo(tmp_path: Path) -> None:
    test_store = GraphStore(tmp_path / "test.db")
    with patch.object(server_module, "store", test_store), \
         patch.object(server_module, "DATA_DIR", tmp_path):
        idx = server_module._impl_index_repo(str(FIXTURES))
        repo_id = idx["repo_id"]

        result = server_module._impl_delete_repo(repo_id)
        assert result["success"] is True

        result = server_module._impl_delete_repo(repo_id)
        assert result["success"] is False


def test_impl_query_nodes(tmp_path: Path) -> None:
    test_store = GraphStore(tmp_path / "test.db")
    with patch.object(server_module, "store", test_store), \
         patch.object(server_module, "DATA_DIR", tmp_path):
        idx = server_module._impl_index_repo(str(FIXTURES))
        repo_id = idx["repo_id"]

        result = server_module._impl_query_nodes(repo_id, node_type="class")
        assert result["count"] > 0
        assert all(n["node_type"] == "class" for n in result["nodes"])


def test_impl_get_stats(tmp_path: Path) -> None:
    test_store = GraphStore(tmp_path / "test.db")
    with patch.object(server_module, "store", test_store), \
         patch.object(server_module, "DATA_DIR", tmp_path):
        idx = server_module._impl_index_repo(str(FIXTURES))
        repo_id = idx["repo_id"]

        stats = server_module._impl_get_stats(repo_id)
        assert stats["node_count"] > 0
        assert "nodes_by_type" in stats
        assert "edges_by_type" in stats
        assert "top_connected" in stats


def test_impl_export_graph(tmp_path: Path) -> None:
    test_store = GraphStore(tmp_path / "test.db")
    with patch.object(server_module, "store", test_store), \
         patch.object(server_module, "DATA_DIR", tmp_path):
        idx = server_module._impl_index_repo(str(FIXTURES))
        repo_id = idx["repo_id"]

        result = server_module._impl_export_graph(repo_id)
        assert "nodes" in result
        assert "links" in result
        assert "metadata" in result
        assert len(result["nodes"]) > 0


def test_impl_export_graph_not_found(tmp_path: Path) -> None:
    test_store = GraphStore(tmp_path / "test.db")
    with patch.object(server_module, "store", test_store):
        result = server_module._impl_export_graph("nonexistent")
        assert "error" in result


def test_impl_find_path(tmp_path: Path) -> None:
    test_store = GraphStore(tmp_path / "test.db")
    with patch.object(server_module, "store", test_store), \
         patch.object(server_module, "DATA_DIR", tmp_path):
        idx = server_module._impl_index_repo(str(FIXTURES))
        repo_id = idx["repo_id"]

        # Find path from module to one of its functions
        result = server_module._impl_find_path(
            repo_id, "simple_module", "simple_module.helper"
        )
        assert result["found"] is True
        assert len(result["path"]) >= 2
