"""LLM-powered enrichment: summaries, clusters, semantic edges."""

import json
import struct
from collections import defaultdict

from .llm import get_llm_client
from .models import ClusterInfo, EdgeType, GraphEdge, GraphNode
from .store import GraphStore

# ── Summaries ────────────────────────────────────────────────────────────────


def _build_summary_prompt(nodes: list[GraphNode], file_path: str) -> list[dict[str, str]]:
    """Build a prompt for summarizing all nodes in a single file."""
    node_descriptions = []
    for n in nodes:
        parts = [f"- {n.node_type} `{n.node_id}` ({n.name})"]
        if n.docstring:
            parts.append(f"  docstring: {n.docstring[:200]}")
        if n.parameters:
            parts.append(f"  params: {', '.join(n.parameters)}")
        if n.decorators:
            parts.append(f"  decorators: {', '.join(n.decorators)}")
        if n.bases:
            parts.append(f"  bases: {', '.join(n.bases)}")
        node_descriptions.append("\n".join(parts))

    return [
        {
            "role": "system",
            "content": (
                "You summarize Python code entities. For each entity, write a single "
                "concise sentence (under 15 words) describing what it does. "
                "Return valid JSON: an array of objects with 'node_id' and 'summary' fields. "
                "No markdown, no explanation, just the JSON array."
            ),
        },
        {
            "role": "user",
            "content": (
                f"File: {file_path}\n\n"
                f"Summarize each of these code entities:\n\n"
                + "\n\n".join(node_descriptions)
            ),
        },
    ]


def _generate_summaries(store: GraphStore, repo_id: str) -> int:
    """Generate summaries for nodes that don't have one."""
    llm = get_llm_client()
    nodes = store.get_nodes_without_summary(repo_id, limit=1000)
    if not nodes:
        return 0

    # Group by file
    by_file: dict[str, list[GraphNode]] = defaultdict(list)
    for n in nodes:
        by_file[n.file_path].append(n)

    # Build prompts (one per file)
    prompts = []
    file_keys = []
    for file_path, file_nodes in by_file.items():
        prompts.append(_build_summary_prompt(file_nodes, file_path))
        file_keys.append(file_path)

    # Batch call
    results = llm.batch_complete(
        prompts,
        max_concurrent=5,
        temperature=0.2,
        max_tokens=2000,
    )

    # Parse results and collect summaries
    all_summaries: list[tuple[str, str]] = []
    for result in results:
        content = result.get("content", "")
        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "node_id" in item and "summary" in item:
                        all_summaries.append((item["node_id"], item["summary"]))
        except (json.JSONDecodeError, KeyError):
            continue

    if all_summaries:
        store.update_node_summaries(repo_id, all_summaries)
    return len(all_summaries)


# ── Clusters ─────────────────────────────────────────────────────────────────


def _detect_clusters(store: GraphStore, repo_id: str) -> int:
    """Detect communities and label them with LLM."""
    import networkx as nx

    g = store._get_graph(repo_id)
    if g.number_of_nodes() < 3:
        return 0

    # Community detection on undirected view
    undirected = g.to_undirected()
    try:
        communities = nx.community.louvain_communities(undirected, seed=42)
    except Exception:
        return 0

    if not communities:
        return 0

    llm = get_llm_client()
    clusters: list[ClusterInfo] = []
    assignments: dict[str, str] = {}

    # Build labeling prompts
    prompts = []
    cluster_members: list[list[str]] = []
    for i, community in enumerate(communities):
        if len(community) < 2:
            continue
        members = sorted(community)
        cluster_members.append(members)

        # Collect node info for the prompt
        member_info = []
        for nid in members[:20]:  # Cap at 20 to keep prompt short
            attrs = dict(g.nodes.get(nid, {}))
            name = attrs.get("name", nid.split(".")[-1])
            node_type = attrs.get("node_type", "unknown")
            summary = attrs.get("summary", "")
            line = f"- {node_type} `{name}`: {summary}" if summary else f"- {node_type} `{name}`"
            member_info.append(line)

        prompts.append([
            {
                "role": "system",
                "content": (
                    "You label code clusters. Given a list of Python code entities that "
                    "form a community in the code graph, provide a short label (2-4 words) "
                    "and a one-sentence description. Return JSON: "
                    '{\"label\": \"...\", \"description\": \"...\"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Cluster with {len(members)} members:\n"
                    + "\n".join(member_info)
                ),
            },
        ])

    if not prompts:
        return 0

    results = llm.batch_complete(prompts, max_concurrent=5, temperature=0.3, max_tokens=200)

    for i, result in enumerate(results):
        cluster_id = f"cluster_{i}"
        members = cluster_members[i]
        content = result.get("content", "").strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0]

        label = f"Cluster {i}"
        description = None
        try:
            parsed = json.loads(content)
            label = parsed.get("label", label)
            description = parsed.get("description")
        except (json.JSONDecodeError, KeyError):
            pass

        clusters.append(ClusterInfo(
            cluster_id=cluster_id,
            label=label,
            description=description,
            member_ids=members,
            member_count=len(members),
        ))

        for nid in members:
            assignments[nid] = cluster_id

    store.save_clusters(repo_id, clusters)
    store.update_node_clusters(repo_id, assignments)
    return len(clusters)


# ── Semantic edges ───────────────────────────────────────────────────────────


def _compute_semantic_edges(
    store: GraphStore,
    repo_id: str,
    threshold: float = 0.8,
) -> int:
    """Embed node signatures and add similarity edges above threshold."""
    try:
        import numpy as np
    except ImportError:
        return 0

    llm = get_llm_client()
    nodes = store.query_nodes(repo_id, limit=5000)
    # Only embed functions, methods, classes (not modules or externals)
    embeddable = [
        n for n in nodes
        if n.node_type in ("function", "method", "class")
    ]
    if len(embeddable) < 2:
        return 0

    # Build signature strings
    texts = []
    node_ids = []
    for n in embeddable:
        sig = f"{n.name}"
        if n.docstring:
            sig += f": {n.docstring[:150]}"
        if n.parameters:
            sig += f" | params: {', '.join(n.parameters)}"
        if n.decorators:
            sig += f" | decorators: {', '.join(n.decorators)}"
        texts.append(sig)
        node_ids.append(n.node_id)

    # Get embeddings
    embeddings = llm.embed(texts)
    if not embeddings:
        return 0

    # Store as BLOBs
    emb_data = []
    for nid, emb in zip(node_ids, embeddings):
        emb_bytes = struct.pack(f"{len(emb)}f", *emb)
        emb_data.append((nid, emb_bytes, "text-embedding-3-small"))
    store.save_embeddings(repo_id, emb_data)

    # Compute cosine similarity
    emb_matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = emb_matrix / norms
    similarity = normalized @ normalized.T

    # Create edges above threshold
    semantic_edges: list[GraphEdge] = []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            sim = float(similarity[i, j])
            if sim >= threshold:
                semantic_edges.append(GraphEdge(
                    source=node_ids[i],
                    target=node_ids[j],
                    edge_type=EdgeType.SEMANTIC_SIMILARITY,
                    weight=round(sim, 4),
                ))

    return store.add_semantic_edges(repo_id, semantic_edges)


# ── Public API ───────────────────────────────────────────────────────────────


def _impl_enrich_repo(repo_id: str, store: GraphStore) -> dict:
    """Generate summaries, detect clusters, compute semantic edges."""
    llm = get_llm_client()
    if not llm.is_available():
        return {
            "error": (
                "OPENAI_API_KEY not configured. "
                "Set the environment variable to enable LLM features."
            )
        }

    repo = store.get_repo(repo_id)
    if not repo:
        return {"error": f"Repo not found: {repo_id}"}

    llm.reset_cost_tracking()

    summaries_count = _generate_summaries(store, repo_id)
    clusters_count = _detect_clusters(store, repo_id)
    semantic_count = _compute_semantic_edges(store, repo_id)

    cost = llm.get_cost_summary()

    return {
        "repo_id": repo_id,
        "status": "enriched",
        "summaries_generated": summaries_count,
        "clusters_found": clusters_count,
        "semantic_edges_added": semantic_count,
        "cost": cost.model_dump(),
    }


def _impl_get_clusters(repo_id: str, store: GraphStore) -> dict:
    """Return labeled clusters."""
    repo = store.get_repo(repo_id)
    if not repo:
        return {"error": f"Repo not found: {repo_id}"}

    clusters = store.get_clusters(repo_id)
    return {
        "repo_id": repo_id,
        "count": len(clusters),
        "clusters": [c.model_dump() for c in clusters],
    }
