# codegraph

A FastMCP server that parses Python repositories into queryable code relationship graphs. Uses AST analysis to extract functions, classes, methods, imports, calls, and inheritance — then optionally enriches the graph with LLM-generated summaries, cluster labels, semantic similarity edges, natural language queries, impact analysis, and codebase narratives.

## Install

```bash
pip install codegraph
```

For LLM features (summaries, clusters, semantic edges, NL queries):

```bash
pip install "codegraph[llm]"
```

## Add to Claude Code

```bash
claude mcp add --scope user codegraph -- python3 -m codegraph
```

With LLM features, add your API key to `~/.codegraph/.env`:

```
OPENAI_API_KEY=sk-your-key-here
```

## CLI

```bash
codegraph index /path/to/repo          # Index a repository
codegraph enrich /path/to/repo         # Index + enrich (summaries, clusters, semantic edges)
codegraph stats <repo_id>              # Show codebase summary
codegraph export <repo_id> -o out.json # Export graph as D3.js JSON
codegraph                              # Start MCP server (default)
codegraph -v enrich /path/to/repo      # Verbose mode (debug logging)
```

## MCP Tools (14)

### Graph Building

| Tool | Description |
|------|-------------|
| `index_repo(source)` | Parse a local path or git URL into a code graph |
| `list_repos()` | List all indexed repositories |
| `delete_repo(repo_id)` | Remove a repository and all its data |

### Querying

| Tool | Description |
|------|-------------|
| `query_nodes(repo_id, node_type?, name?, file_path?, limit?)` | Find nodes by type, name, or file |
| `get_edges(repo_id, node_id, direction?, edge_type?)` | Get incoming/outgoing edges for a node |
| `get_subgraph(repo_id, node_id, depth?, edge_types?)` | Get everything within N hops of a node |
| `find_path(repo_id, source_id, target_id)` | Shortest path between two nodes |
| `get_stats(repo_id)` | Codebase summary: counts by type, top connected nodes |
| `export_graph(repo_id, format?)` | Export full graph as D3.js-compatible JSON |

### LLM-Powered (requires `OPENAI_API_KEY`)

| Tool | Description |
|------|-------------|
| `enrich_repo(repo_id, similarity_threshold?, force?)` | Generate summaries + clusters + semantic edges |
| `get_clusters(repo_id)` | Return labeled code communities |
| `ask(repo_id, question)` | Natural language questions about the codebase |
| `analyze_impact(repo_id, node_id)` | "What breaks if I change this?" |
| `narrate(repo_id)` | Generate a guided tour of the codebase |

## Graph Model

### Node Types

| Type | What | Example ID |
|------|------|-----------|
| `module` | Python file | `src.pkg.module` |
| `class` | Class definition | `src.pkg.module.MyClass` |
| `function` | Top-level function | `src.pkg.module.helper` |
| `method` | Method inside a class | `src.pkg.module.MyClass.process` |

### Edge Types

| Type | Meaning |
|------|---------|
| `contains` | Module contains class, class contains method |
| `imports` | Module imports from another module |
| `calls` | Function/method calls another function/method |
| `inherits` | Class extends a base class |
| `decorates` | Decorator applied to a function/class |
| `semantic_similarity` | LLM-detected semantic relationship (enrichment) |

## LLM Features

All LLM features use **gpt-5.4-mini** for reasoning and **text-embedding-3-small** for embeddings. Every LLM tool returns a `cost` field with token counts and estimated USD.

### Summaries

`enrich_repo` generates a one-line plain-English summary for every function, class, and method. Summaries appear in `query_nodes` responses and `export_graph(format="full")` output.

### Clusters

Community detection (Louvain algorithm) groups related code entities. The LLM labels each cluster with a human-readable name and description. View with `get_clusters`.

### Semantic Edges

Node signatures (name + docstring + parameters) are embedded and compared via cosine similarity. Pairs above the threshold (default 0.8) get a `semantic_similarity` edge.

### Natural Language Queries

`ask` uses tool-use: the LLM internally calls `query_nodes`, `get_edges`, `get_subgraph`, and `get_stats` to find relevant information, then synthesizes an answer.

### Impact Analysis

`analyze_impact` walks the call graph to find all callers (direct and transitive) of a node, then asks the LLM to assess risk level and explain what could break.

### Codebase Narrative

`narrate` generates a structured walkthrough of the project — sections covering each component, how they relate, and key entry points.

## Visualization

Export a graph and load it in [codegraph-ui](https://github.com/mfbaig35r/codegraph-ui) — a cyberpunk-themed Next.js frontend with interactive force-directed visualization, filtering, search, and detail panels.

```bash
codegraph export <repo_id> -o graph.json
# Then upload graph.json in codegraph-ui
```

## Configuration

| Env var | Default | Description |
|---|---|---|
| `CODEGRAPH_DIR` | `~/.codegraph` | Data directory (database, clones, embeddings) |
| `OPENAI_API_KEY` | — | Required for LLM features (also reads from `~/.codegraph/.env`) |

## Development

```bash
git clone https://github.com/mfbaig35r/codegraph.git
cd codegraph
pip install -e ".[dev,llm]"

ruff check src/ tests/                              # Lint
mypy src/codegraph/ --ignore-missing-imports         # Type check
pytest tests/ -v                                     # Tests (70 tests)
```
