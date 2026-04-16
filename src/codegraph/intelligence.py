"""LLM-powered reasoning: natural language queries, impact analysis, narratives."""

import json

import networkx as nx

from .llm import get_llm_client
from .store import GraphStore

# ── Inner tool definitions for NL queries ────────────────────────────────────

_INNER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_nodes",
            "description": "Find nodes in the code graph by type, name, or file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_type": {
                        "type": "string",
                        "enum": ["module", "class", "function", "method"],
                        "description": "Filter by node type.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Substring match on node name.",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Substring match on file path.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 20).",
                        "default": 20,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_edges",
            "description": (
                "Get incoming and outgoing edges for a node. "
                "Shows callers, callees, imports, inheritance, containment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Fully qualified node ID.",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["incoming", "outgoing", "both"],
                        "default": "both",
                    },
                    "edge_type": {
                        "type": "string",
                        "enum": [
                            "contains", "imports", "calls",
                            "inherits", "decorates",
                        ],
                        "description": "Filter by edge type.",
                    },
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_subgraph",
            "description": "Get all nodes and edges within N hops of a center node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Center node ID.",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "How many hops to traverse (default 2).",
                        "default": 2,
                    },
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stats",
            "description": "Get high-level codebase summary: counts by type, top connected nodes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _execute_inner_tool(
    tool_name: str,
    arguments: dict,
    store: GraphStore,
    repo_id: str,
) -> str:
    """Execute an inner tool and return JSON string result."""
    if tool_name == "query_nodes":
        nodes = store.query_nodes(
            repo_id,
            node_type=arguments.get("node_type"),
            name=arguments.get("name"),
            file_path=arguments.get("file_path"),
            limit=arguments.get("limit", 20),
        )
        return json.dumps([
            {
                "node_id": n.node_id, "name": n.name,
                "type": n.node_type, "file": n.file_path,
                "summary": n.summary, "docstring": n.docstring,
            }
            for n in nodes
        ])[:4000]

    if tool_name == "get_edges":
        result = store.get_edges(
            repo_id,
            arguments["node_id"],
            direction=arguments.get("direction", "both"),
            edge_type=arguments.get("edge_type"),
        )
        edges = []
        for e in result["incoming"]:
            edges.append({
                "source": e.source, "target": e.target,
                "type": e.edge_type, "direction": "incoming",
            })
        for e in result["outgoing"]:
            edges.append({
                "source": e.source, "target": e.target,
                "type": e.edge_type, "direction": "outgoing",
            })
        return json.dumps(edges)[:4000]

    if tool_name == "get_subgraph":
        sub_nodes, sub_edges = store.get_subgraph(
            repo_id,
            arguments["node_id"],
            depth=arguments.get("depth", 2),
        )
        return json.dumps({
            "node_count": len(sub_nodes),
            "edge_count": len(sub_edges),
            "nodes": [
                {"id": n.node_id, "name": n.name, "type": n.node_type,
                 "summary": n.summary}
                for n in sub_nodes
            ][:30],
        })[:4000]

    if tool_name == "get_stats":
        return json.dumps(store.get_stats(repo_id))[:4000]

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ── Natural Language Queries ─────────────────────────────────────────────────


def _impl_ask(repo_id: str, question: str, store: GraphStore) -> dict:
    """Answer a natural language question about the codebase."""
    llm = get_llm_client()
    if not llm.is_available():
        return {"error": "OPENAI_API_KEY not configured."}

    repo = store.get_repo(repo_id)
    if not repo:
        return {"error": f"Repo not found: {repo_id}"}

    llm.reset_cost_tracking()

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                f"You are a code analysis assistant for the Python project '{repo.name}'. "
                f"It has {repo.node_count} code entities and {repo.edge_count} relationships. "
                "Use the provided tools to query the code graph, then synthesize a clear answer. "
                "Be concise and specific. Reference node IDs when relevant."
            ),
        },
        {"role": "user", "content": question},
    ]

    source_nodes: list[str] = []
    max_rounds = 5

    for _ in range(max_rounds):
        result = llm.complete(messages, tools=_INNER_TOOLS, max_tokens=1500)

        if "tool_calls" not in result:
            # Final answer
            cost = llm.get_cost_summary()
            return {
                "repo_id": repo_id,
                "question": question,
                "answer": result["content"],
                "sources": source_nodes,
                "cost": cost.model_dump(),
            }

        # Execute tool calls
        messages.append({
            "role": "assistant",
            "content": result.get("content", ""),
            "tool_calls": result["tool_calls"],
        })

        for tc in result["tool_calls"]:
            func_name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            tool_result = _execute_inner_tool(func_name, args, store, repo_id)

            # Track source nodes
            if func_name == "query_nodes":
                try:
                    for n in json.loads(tool_result):
                        if n.get("node_id"):
                            source_nodes.append(n["node_id"])
                except (json.JSONDecodeError, TypeError):
                    pass

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_result,
            })

    # Exhausted rounds
    cost = llm.get_cost_summary()
    return {
        "repo_id": repo_id,
        "question": question,
        "answer": "Unable to fully answer within the tool-call limit.",
        "sources": source_nodes,
        "cost": cost.model_dump(),
    }


# ── Impact Analysis ──────────────────────────────────────────────────────────


def _impl_analyze_impact(
    repo_id: str, node_id: str, store: GraphStore,
) -> dict:
    """Analyze what would be affected by changing a node."""
    llm = get_llm_client()
    if not llm.is_available():
        return {"error": "OPENAI_API_KEY not configured."}

    repo = store.get_repo(repo_id)
    if not repo:
        return {"error": f"Repo not found: {repo_id}"}

    g = store._get_graph(repo_id)
    if node_id not in g:
        return {"error": f"Node not found: {node_id}"}

    llm.reset_cost_tracking()

    # Build calls-only subgraph and find ancestors (callers of callers)
    calls_view = nx.subgraph_view(
        g,
        filter_edge=lambda u, v: g.edges[u, v].get("edge_type") == "calls",
    )

    # Find all nodes that call this node (directly or transitively)
    try:
        ancestors = nx.ancestors(calls_view, node_id)
    except nx.NetworkXError:
        ancestors = set()

    # Also get direct callers from other edge types
    direct_dependents: set[str] = set()
    for pred in g.predecessors(node_id):
        edge_data = g.edges[pred, node_id]
        if edge_data.get("edge_type") in ("calls", "imports", "inherits"):
            direct_dependents.add(pred)

    all_affected = ancestors | direct_dependents
    if not all_affected:
        cost = llm.get_cost_summary()
        return {
            "repo_id": repo_id,
            "node_id": node_id,
            "affected_count": 0,
            "risk_level": "low",
            "explanation": f"No callers or dependents found for {node_id}.",
            "affected_nodes": [],
            "impact_chain": [],
            "cost": cost.model_dump(),
        }

    # Build impact chain with depth info
    impact_chain = []
    for affected_id in sorted(all_affected):
        attrs = dict(g.nodes.get(affected_id, {}))
        if "node_type" not in attrs:
            continue
        try:
            path_len = nx.shortest_path_length(
                calls_view, affected_id, node_id
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_len = 1
        impact_chain.append({
            "node_id": affected_id,
            "name": attrs.get("name", ""),
            "type": attrs.get("node_type", ""),
            "file": attrs.get("file_path", ""),
            "summary": attrs.get("summary", ""),
            "depth": path_len,
        })

    impact_chain.sort(key=lambda x: x["depth"])

    # Ask LLM to assess risk
    chain_text = "\n".join(
        f"  depth={c['depth']}: {c['type']} `{c['node_id']}` — {c['summary'] or 'no summary'}"
        for c in impact_chain[:30]
    )

    target_attrs = dict(g.nodes.get(node_id, {}))
    target_summary = target_attrs.get("summary", "") or target_attrs.get("docstring", "")

    result = llm.complete([
        {
            "role": "system",
            "content": (
                "You analyze the impact of changing code. Given a target node and its "
                "dependents, assess the risk level (low/medium/high/critical) and explain "
                "what could break. Be specific about which callers are most at risk and why. "
                "Return JSON: {\"risk_level\": \"...\", \"explanation\": \"...\"}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Target: {node_id}\n"
                f"Type: {target_attrs.get('node_type', 'unknown')}\n"
                f"Summary: {target_summary}\n"
                f"Parameters: {target_attrs.get('parameters', [])}\n\n"
                f"Affected nodes ({len(impact_chain)} total):\n{chain_text}"
            ),
        },
    ], temperature=0.3, max_tokens=500)

    risk_level = "medium"
    explanation = result.get("content", "")
    content = explanation.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(content)
        risk_level = parsed.get("risk_level", risk_level)
        explanation = parsed.get("explanation", explanation)
    except (json.JSONDecodeError, KeyError):
        pass

    cost = llm.get_cost_summary()
    return {
        "repo_id": repo_id,
        "node_id": node_id,
        "affected_count": len(impact_chain),
        "risk_level": risk_level,
        "explanation": explanation,
        "affected_nodes": [c["node_id"] for c in impact_chain],
        "impact_chain": impact_chain,
        "cost": cost.model_dump(),
    }


# ── Codebase Narrative ───────────────────────────────────────────────────────


def _impl_narrate(repo_id: str, store: GraphStore) -> dict:
    """Generate a guided tour of the codebase."""
    llm = get_llm_client()
    if not llm.is_available():
        return {"error": "OPENAI_API_KEY not configured."}

    repo = store.get_repo(repo_id)
    if not repo:
        return {"error": f"Repo not found: {repo_id}"}

    llm.reset_cost_tracking()

    stats = store.get_stats(repo_id)
    clusters = store.get_clusters(repo_id)

    # Collect module-level info
    modules = store.query_nodes(repo_id, node_type="module", limit=50)
    classes = store.query_nodes(repo_id, node_type="class", limit=50)

    # Build context
    module_info = "\n".join(
        f"- {m.node_id} ({m.file_path}): {m.summary or m.docstring or 'no description'}"
        for m in modules
    )
    class_info = "\n".join(
        f"- {c.node_id}: {c.summary or c.docstring or 'no description'}"
        + (f" (bases: {', '.join(c.bases)})" if c.bases else "")
        for c in classes
    )
    cluster_info = "\n".join(
        f"- {c.label}: {c.description or 'no description'} ({c.member_count} members)"
        for c in clusters
    ) if clusters else "No clusters detected. Run enrich_repo first for cluster analysis."

    # Top connected nodes
    top_nodes = stats.get("top_connected", [])[:5]
    top_info = "\n".join(
        f"- {t['node_id']} ({t['connections']} connections)"
        for t in top_nodes
    )

    result = llm.complete([
        {
            "role": "system",
            "content": (
                "You are a technical writer generating a codebase walkthrough. "
                "Write a structured narrative that helps a developer understand "
                "this Python project. Use the graph data provided to explain "
                "the architecture, key components, and how they relate. "
                "Return JSON with format: "
                '{"title": "...", "sections": [{"title": "...", '
                '"summary": "...", "key_nodes": ["node_id1", ...], '
                '"relationships": "..."}]}'
                "\nAim for 3-6 sections. Be specific, reference actual node IDs."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Project: {repo.name}\n"
                f"Stats: {stats.get('node_count', 0)} nodes, "
                f"{stats.get('edge_count', 0)} edges\n"
                f"Node types: {json.dumps(stats.get('nodes_by_type', {}))}\n"
                f"Edge types: {json.dumps(stats.get('edges_by_type', {}))}\n\n"
                f"Modules:\n{module_info}\n\n"
                f"Classes:\n{class_info}\n\n"
                f"Clusters:\n{cluster_info}\n\n"
                f"Most connected nodes:\n{top_info}"
            ),
        },
    ], temperature=0.4, max_tokens=3000)

    content = result.get("content", "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0]

    title = repo.name
    sections: list[dict] = []
    try:
        parsed = json.loads(content)
        title = parsed.get("title", title)
        sections = parsed.get("sections", [])
    except (json.JSONDecodeError, KeyError):
        sections = [{"title": "Overview", "summary": content, "key_nodes": [], "relationships": ""}]

    cost = llm.get_cost_summary()
    return {
        "repo_id": repo_id,
        "title": title,
        "sections": sections,
        "cost": cost.model_dump(),
    }
