"""Tests for codegraph.cli — argument parsing and command dispatch."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from codegraph.store import GraphStore

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    return GraphStore(tmp_path / "test.db")


def test_cmd_index(store: GraphStore, tmp_path: Path) -> None:
    import codegraph.server as server_module

    with patch.object(server_module, "store", store), \
         patch.object(server_module, "DATA_DIR", tmp_path):
        result = server_module._impl_index_repo(str(FIXTURES))

    assert result["status"] == "indexed"
    assert result["node_count"] > 0


def test_cmd_stats(store: GraphStore) -> None:
    import codegraph.server as server_module

    with patch.object(server_module, "store", store), \
         patch.object(server_module, "DATA_DIR", FIXTURES.parent):
        idx = server_module._impl_index_repo(str(FIXTURES))
        repo_id = idx["repo_id"]
        stats = server_module._impl_get_stats(repo_id)

    assert stats["name"] == "fixtures"
    assert stats["node_count"] > 0
    assert "nodes_by_type" in stats


def test_cmd_export_to_file(store: GraphStore, tmp_path: Path) -> None:
    import codegraph.server as server_module

    with patch.object(server_module, "store", store), \
         patch.object(server_module, "DATA_DIR", FIXTURES.parent):
        idx = server_module._impl_index_repo(str(FIXTURES))
        repo_id = idx["repo_id"]
        result = server_module._impl_export_graph(repo_id, format="d3")

    output_file = tmp_path / "test_export.json"
    with open(output_file, "w") as f:
        json.dump(result, f)

    loaded = json.loads(output_file.read_text())
    assert "nodes" in loaded
    assert "links" in loaded
    assert len(loaded["nodes"]) > 0


def test_cmd_export_full_format(store: GraphStore) -> None:
    import codegraph.server as server_module

    with patch.object(server_module, "store", store), \
         patch.object(server_module, "DATA_DIR", FIXTURES.parent):
        idx = server_module._impl_index_repo(str(FIXTURES))
        repo_id = idx["repo_id"]
        result = server_module._impl_export_graph(repo_id, format="full")

    # Full format should include docstring, decorators, etc.
    node = result["nodes"][0]
    assert "docstring" in node
    assert "decorators" in node
    assert "summary" in node
    assert "cluster_id" in node


def test_cmd_index_bad_path() -> None:
    import codegraph.server as server_module

    result = server_module._impl_index_repo("/nonexistent/path")
    assert "error" in result


def test_cli_main_serve_is_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no subcommand, cli should call the MCP server main."""
    monkeypatch.setattr("sys.argv", ["codegraph", "serve"])

    with patch("codegraph.server.main") as mock_serve:
        from codegraph.cli import cli_main
        cli_main()
        mock_serve.assert_called_once()


def test_cli_index_subcommand(
    store: GraphStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The index subcommand should call _impl_index_repo."""
    import codegraph.server as server_module

    monkeypatch.setattr("sys.argv", ["codegraph", "index", str(FIXTURES)])
    with patch.object(server_module, "store", store), \
         patch.object(server_module, "DATA_DIR", FIXTURES.parent):
        from codegraph.cli import cli_main
        cli_main()
