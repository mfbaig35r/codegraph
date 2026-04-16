# codegraph

A FastMCP server that parses Python repositories and builds a queryable graph of code relationships (imports, calls, inheritance, containment). Export as D3.js-compatible JSON for visualization.

## Install

```bash
pip install codegraph
```

## Add to Claude Code

```bash
claude mcp add codegraph -- uvx codegraph
```

## Tools

| Tool | Purpose |
|------|---------|
| `index_repo` | Parse a local path or git URL into a code graph |
| `list_repos` | List all indexed repositories |
| `delete_repo` | Remove a repository's graph data |
| `query_nodes` | Find nodes by type, name, or file path |
| `get_edges` | Get incoming/outgoing edges for a node |
| `get_subgraph` | Get everything within N hops of a node |
| `export_graph` | Export full graph as D3.js JSON |
| `get_stats` | Codebase summary with counts and top-connected nodes |
| `find_path` | Shortest path between two nodes |
