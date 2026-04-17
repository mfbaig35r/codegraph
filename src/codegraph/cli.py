"""CLI for batch operations: index, enrich, export."""

import argparse
import json
import sys

from .server import _impl_export_graph, _impl_get_stats, _impl_index_repo, store


def cmd_index(args: argparse.Namespace) -> None:
    """Index a repository."""
    result = _impl_index_repo(args.source)
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print(
        f"Indexed {result['name']}: "
        f"{result['node_count']} nodes, {result['edge_count']} edges"
    )
    print(f"repo_id: {result['repo_id']}")


def cmd_enrich(args: argparse.Namespace) -> None:
    """Enrich a repository with LLM features."""
    from .enrichment import _impl_enrich_repo

    # If source is a path, index first
    repo_id = args.repo_id
    if "/" in repo_id or repo_id.startswith("."):
        idx = _impl_index_repo(repo_id)
        if "error" in idx:
            print(f"Error indexing: {idx['error']}", file=sys.stderr)
            sys.exit(1)
        repo_id = idx["repo_id"]
        print(
            f"Indexed {idx['name']}: "
            f"{idx['node_count']} nodes, {idx['edge_count']} edges"
        )

    result = _impl_enrich_repo(repo_id, store)
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print(
        f"Enriched: {result['summaries_generated']} summaries, "
        f"{result['clusters_found']} clusters, "
        f"{result['semantic_edges_added']} semantic edges"
    )
    cost = result.get("cost", {})
    if cost.get("estimated_cost_usd"):
        print(f"Cost: ${cost['estimated_cost_usd']:.4f}")


def cmd_export(args: argparse.Namespace) -> None:
    """Export graph as JSON."""
    result = _impl_export_graph(args.repo_id, format=args.format)
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Exported to {args.output}")
    else:
        json.dump(result, sys.stdout, indent=2)
        print()


def cmd_stats(args: argparse.Namespace) -> None:
    """Show repository stats."""
    result = _impl_get_stats(args.repo_id)
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"Repository: {result['name']}")
    print(f"Files: {result['file_count']}")
    print(f"Nodes: {result['node_count']}")
    print(f"Edges: {result['edge_count']}")
    print("\nNodes by type:")
    for t, c in result.get("nodes_by_type", {}).items():
        print(f"  {t}: {c}")
    print("\nEdges by type:")
    for t, c in result.get("edges_by_type", {}).items():
        print(f"  {t}: {c}")
    top = result.get("top_connected", [])[:5]
    if top:
        print("\nMost connected:")
        for t in top:
            print(f"  {t['node_id']} ({t['connections']})")


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        prog="codegraph",
        description="Python code relationship graph tool",
    )
    sub = parser.add_subparsers(dest="command")

    # index
    p_index = sub.add_parser("index", help="Index a Python repository")
    p_index.add_argument("source", help="Local path or git URL")

    # enrich
    p_enrich = sub.add_parser(
        "enrich", help="Index + enrich with LLM (summaries, clusters, semantic edges)"
    )
    p_enrich.add_argument(
        "repo_id", help="repo_id from index, or a path to auto-index first"
    )

    # export
    p_export = sub.add_parser("export", help="Export graph as JSON")
    p_export.add_argument("repo_id", help="Repository ID")
    p_export.add_argument("-o", "--output", help="Output file (default: stdout)")
    p_export.add_argument(
        "-f", "--format", default="full", choices=["d3", "full"],
        help="Export format (default: full)",
    )

    # stats
    p_stats = sub.add_parser("stats", help="Show repository stats")
    p_stats.add_argument("repo_id", help="Repository ID")

    # serve (default — MCP server)
    sub.add_parser("serve", help="Start the MCP server (default)")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "enrich":
        cmd_enrich(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "serve" or args.command is None:
        from .server import main
        main()
    else:
        parser.print_help()
