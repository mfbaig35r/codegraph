"""codegraph FastMCP server — 14 tools for querying Python code relationship graphs."""

import os
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import FastMCP

from .models import RepoInfo
from .parser import parse_repository
from .repo import resolve_repo
from .store import GraphStore

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = (
    Path(os.environ.get("CODEGRAPH_DIR", "~/.codegraph"))
    .expanduser()
    .resolve()
)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Singletons ────────────────────────────────────────────────────────────────

store = GraphStore(DATA_DIR / "codegraph.db")

# ── Server ────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "codegraph",
    instructions=(
        "Parse Python repositories and build a queryable graph of code relationships. "
        "Use index_repo to parse a local path or git URL. "
        "Use query_nodes to find classes, functions, methods by type/name/file. "
        "Use get_edges to see callers, callees, imports, inheritance for a node. "
        "Use get_subgraph to get everything connected to a node within N hops. "
        "Use export_graph to get D3.js-compatible JSON for visualization. "
        "Use get_stats for a high-level codebase summary. "
        "Use find_path to discover how two symbols are connected. "
        "Use enrich_repo to add LLM-generated summaries, cluster labels, and semantic edges. "
        "Use get_clusters to see labeled code communities. "
        "Use ask to answer natural language questions about the codebase. "
        "Use analyze_impact to see what would break if you changed a node. "
        "Use narrate to get a guided tour of the codebase."
    ),
)


# ── Implementations ──────────────────────────────────────────────────────────


def _impl_index_repo(source: str) -> dict:
    try:
        repo_id, repo_path = resolve_repo(source, DATA_DIR / "clones")
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc)}

    try:
        nodes, edges = parse_repository(repo_path)
    except Exception as exc:
        return {"error": f"Failed to parse repository: {exc}"}

    py_files = list(Path(repo_path).rglob("*.py"))

    repo_info = RepoInfo(
        repo_id=repo_id,
        repo_path=repo_path,
        repo_url=source if source.startswith("http") or source.startswith("git@") else None,
        name=Path(repo_path).name,
        indexed_at=datetime.now(timezone.utc).isoformat(),
        file_count=len(py_files),
        node_count=len(nodes),
        edge_count=len(edges),
    )
    store.save_repo(repo_info, nodes, edges)

    return {
        "repo_id": repo_id,
        "name": repo_info.name,
        "status": "indexed",
        "file_count": repo_info.file_count,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "indexed_at": repo_info.indexed_at,
    }


def _impl_list_repos() -> dict:
    repos = store.list_repos()
    return {
        "count": len(repos),
        "repos": [r.model_dump() for r in repos],
    }


def _impl_delete_repo(repo_id: str) -> dict:
    if store.delete_repo(repo_id):
        return {"success": True, "repo_id": repo_id}
    return {"success": False, "error": f"Repo not found: {repo_id}"}


def _impl_query_nodes(
    repo_id: str,
    node_type: str | None = None,
    name: str | None = None,
    file_path: str | None = None,
    limit: int = 50,
) -> dict:
    nodes = store.query_nodes(
        repo_id, node_type=node_type, name=name,
        file_path=file_path, limit=limit,
    )
    return {
        "repo_id": repo_id,
        "count": len(nodes),
        "nodes": [n.model_dump() for n in nodes],
    }


def _impl_get_edges(
    repo_id: str,
    node_id: str,
    direction: str = "both",
    edge_type: str | None = None,
) -> dict:
    result = store.get_edges(repo_id, node_id, direction=direction, edge_type=edge_type)
    return {
        "repo_id": repo_id,
        "node_id": node_id,
        "incoming": [e.model_dump() for e in result["incoming"]],
        "outgoing": [e.model_dump() for e in result["outgoing"]],
    }


def _impl_get_subgraph(
    repo_id: str,
    node_id: str,
    depth: int = 2,
    edge_types: list[str] | None = None,
) -> dict:
    nodes, edges = store.get_subgraph(
        repo_id, node_id, depth=depth, edge_types=edge_types,
    )
    return {
        "repo_id": repo_id,
        "center": node_id,
        "depth": depth,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [n.model_dump() for n in nodes],
        "edges": [e.model_dump() for e in edges],
    }


def _impl_export_graph(repo_id: str, format: str = "d3") -> dict:
    repo = store.get_repo(repo_id)
    if not repo:
        return {"error": f"Repo not found: {repo_id}"}
    return store.export_graph(repo_id, format=format)


def _impl_get_stats(repo_id: str) -> dict:
    return store.get_stats(repo_id)


def _impl_find_path(repo_id: str, source_id: str, target_id: str) -> dict:
    path, edges = store.find_path(repo_id, source_id, target_id)
    if not path:
        return {
            "found": False,
            "repo_id": repo_id,
            "source_id": source_id,
            "target_id": target_id,
            "path": [],
            "edges": [],
        }
    return {
        "found": True,
        "repo_id": repo_id,
        "source_id": source_id,
        "target_id": target_id,
        "path": path,
        "edges": [e.model_dump() for e in edges],
    }


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
def index_repo(source: str) -> dict:
    """
    Index a Python repository to build a code relationship graph.

    Parses all .py files, extracts classes, functions, methods, imports,
    call relationships, and inheritance hierarchies.

    Args:
        source: Local directory path or git clone URL.

    Returns:
        repo_id, name, status, file_count, node_count, edge_count, indexed_at.
    """
    return _impl_index_repo(source)


@mcp.tool()
def list_repos() -> dict:
    """
    List all indexed repositories.

    Returns:
        count, repos — each with repo_id, name, repo_path, indexed_at, counts.
    """
    return _impl_list_repos()


@mcp.tool()
def delete_repo(repo_id: str) -> dict:
    """
    Delete an indexed repository and all its graph data.

    Args:
        repo_id: The repository ID returned by index_repo.

    Returns:
        success, repo_id — or error.
    """
    return _impl_delete_repo(repo_id)


@mcp.tool()
def query_nodes(
    repo_id: str,
    node_type: str | None = None,
    name: str | None = None,
    file_path: str | None = None,
    limit: int = 50,
) -> dict:
    """
    Find nodes in the code graph by type, name, or file path.

    Args:
        repo_id:    Repository to search.
        node_type:  Filter by type: module, class, function, method.
        name:       Substring match on node name.
        file_path:  Substring match on file path.
        limit:      Max results (default 50).

    Returns:
        repo_id, count, nodes — each with node_id, node_type, name,
        file_path, line_start, docstring, decorators, parameters, bases.
    """
    return _impl_query_nodes(repo_id, node_type, name, file_path, limit)


@mcp.tool()
def get_edges(
    repo_id: str,
    node_id: str,
    direction: str = "both",
    edge_type: str | None = None,
) -> dict:
    """
    Get incoming and/or outgoing edges for a node.

    Args:
        repo_id:    Repository to search.
        node_id:    Fully qualified node ID (e.g. "pkg.module.ClassName.method").
        direction:  "incoming", "outgoing", or "both" (default).
        edge_type:  Filter by edge type: contains, imports, inherits, calls, decorates.

    Returns:
        repo_id, node_id, incoming edges, outgoing edges.
    """
    return _impl_get_edges(repo_id, node_id, direction, edge_type)


@mcp.tool()
def get_subgraph(
    repo_id: str,
    node_id: str,
    depth: int = 2,
    edge_types: list[str] | None = None,
) -> dict:
    """
    Get everything connected to a node within N hops.

    Args:
        repo_id:     Repository to search.
        node_id:     Center node.
        depth:       How many hops to traverse (default 2).
        edge_types:  Only follow these edge types (default: all).

    Returns:
        center, depth, node_count, edge_count, nodes, edges.
    """
    return _impl_get_subgraph(repo_id, node_id, depth, edge_types)


@mcp.tool()
def export_graph(repo_id: str, format: str = "d3") -> dict:
    """
    Export the full code graph as JSON for visualization.

    Args:
        repo_id:  Repository to export.
        format:   "d3" (default, compact) or "full" (includes docstrings, params, etc).

    Returns:
        D3.js force-directed graph format: nodes[], links[], metadata.
    """
    return _impl_export_graph(repo_id, format)


@mcp.tool()
def get_stats(repo_id: str) -> dict:
    """
    Get a high-level summary of a repository's code graph.

    Args:
        repo_id: Repository to summarize.

    Returns:
        repo_id, name, file_count, node_count, edge_count,
        nodes_by_type, edges_by_type, top_connected, avg_connections.
    """
    return _impl_get_stats(repo_id)


@mcp.tool()
def find_path(repo_id: str, source_id: str, target_id: str) -> dict:
    """
    Find the shortest path between two nodes in the code graph.

    Tries directed path first, then undirected if no directed path exists.

    Args:
        repo_id:    Repository to search.
        source_id:  Starting node ID.
        target_id:  Destination node ID.

    Returns:
        found, path (list of node IDs), edges along the path.
    """
    return _impl_find_path(repo_id, source_id, target_id)


# ── LLM-powered tools ────────────────────────────────────────────────────────


@mcp.tool()
def enrich_repo(repo_id: str) -> dict:
    """
    Enrich a repository with LLM-generated summaries, cluster labels, and
    semantic similarity edges. Requires OPENAI_API_KEY.

    Generates:
    - One-line summaries for every function, class, and method
    - Community clusters with human-readable labels
    - Semantic similarity edges between related nodes

    Idempotent: skips nodes that already have summaries.

    Args:
        repo_id: Repository to enrich.

    Returns:
        repo_id, status, summaries_generated, clusters_found,
        semantic_edges_added, cost.
    """
    from .enrichment import _impl_enrich_repo
    return _impl_enrich_repo(repo_id, store)


@mcp.tool()
def get_clusters(repo_id: str) -> dict:
    """
    Get labeled code clusters (communities) for a repository.

    Run enrich_repo first to generate clusters.

    Args:
        repo_id: Repository to query.

    Returns:
        repo_id, count, clusters — each with cluster_id, label, description,
        member_ids, member_count.
    """
    from .enrichment import _impl_get_clusters
    return _impl_get_clusters(repo_id, store)


@mcp.tool()
def ask(repo_id: str, question: str) -> dict:
    """
    Ask a natural language question about the codebase.

    The LLM queries the code graph internally to find relevant nodes and
    relationships, then synthesizes an answer. Requires OPENAI_API_KEY.

    Args:
        repo_id:   Repository to query.
        question:  Natural language question (e.g. "what handles persistence?").

    Returns:
        repo_id, question, answer, sources (node IDs consulted), cost.
    """
    from .intelligence import _impl_ask
    return _impl_ask(repo_id, question, store)


@mcp.tool()
def analyze_impact(repo_id: str, node_id: str) -> dict:
    """
    Analyze what would be affected by changing a node.

    Walks the call graph to find all callers (direct and transitive),
    then uses the LLM to assess risk. Requires OPENAI_API_KEY.

    Args:
        repo_id:  Repository to analyze.
        node_id:  The node you're considering changing.

    Returns:
        repo_id, node_id, affected_count, risk_level, explanation,
        affected_nodes, impact_chain, cost.
    """
    from .intelligence import _impl_analyze_impact
    return _impl_analyze_impact(repo_id, node_id, store)


@mcp.tool()
def narrate(repo_id: str) -> dict:
    """
    Generate a guided tour of the codebase.

    Analyzes the graph structure, clusters, and node summaries to produce
    a structured walkthrough of the project. Requires OPENAI_API_KEY.
    Best results when run after enrich_repo.

    Args:
        repo_id: Repository to narrate.

    Returns:
        repo_id, title, sections (each with title, summary, key_nodes,
        relationships), cost.
    """
    from .intelligence import _impl_narrate
    return _impl_narrate(repo_id, store)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    mcp.run()
