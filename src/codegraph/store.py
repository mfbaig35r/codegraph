"""Graph storage: SQLite persistence + networkx in-memory graph."""

import json
import sqlite3
import threading
from pathlib import Path

import networkx as nx

from .models import ClusterInfo, EdgeType, GraphEdge, GraphNode, NodeType, RepoInfo

_SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    repo_id     TEXT PRIMARY KEY,
    repo_path   TEXT NOT NULL,
    repo_url    TEXT,
    name        TEXT NOT NULL,
    indexed_at  TEXT NOT NULL,
    file_count  INTEGER DEFAULT 0,
    node_count  INTEGER DEFAULT 0,
    edge_count  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id     TEXT NOT NULL,
    repo_id     TEXT NOT NULL,
    node_type   TEXT NOT NULL,
    name        TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    line_start  INTEGER NOT NULL,
    line_end    INTEGER,
    docstring   TEXT,
    decorators  TEXT DEFAULT '[]',
    parameters  TEXT DEFAULT '[]',
    bases       TEXT DEFAULT '[]',
    module_path TEXT DEFAULT '',
    PRIMARY KEY (repo_id, node_id)
);

CREATE TABLE IF NOT EXISTS edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id     TEXT NOT NULL,
    source      TEXT NOT NULL,
    target      TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    file_path   TEXT,
    line        INTEGER
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(repo_id, node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(repo_id, name);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(repo_id, source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(repo_id, target);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(repo_id, edge_type);
"""


class GraphStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._graphs: dict[str, nx.DiGraph] = {}
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS clusters (
                    cluster_id  TEXT NOT NULL,
                    repo_id     TEXT NOT NULL,
                    label       TEXT NOT NULL,
                    description TEXT,
                    member_ids  TEXT NOT NULL DEFAULT '[]',
                    PRIMARY KEY (repo_id, cluster_id)
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    node_id     TEXT NOT NULL,
                    repo_id     TEXT NOT NULL,
                    embedding   BLOB NOT NULL,
                    model       TEXT NOT NULL DEFAULT 'text-embedding-3-small',
                    PRIMARY KEY (repo_id, node_id)
                );
            """)
            # v0.2 migration: LLM enrichment columns
            for col_ddl in [
                "ALTER TABLE nodes ADD COLUMN summary TEXT",
                "ALTER TABLE nodes ADD COLUMN cluster_id TEXT",
                "ALTER TABLE edges ADD COLUMN weight REAL",
            ]:
                try:
                    self._conn.execute(col_ddl)
                except sqlite3.OperationalError:
                    pass
            self._conn.commit()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _executemany(self, sql: str, params_list: list[tuple]) -> None:
        with self._lock:
            self._conn.executemany(sql, params_list)
            self._conn.commit()

    def _fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    @staticmethod
    def _parse_node(row: dict) -> GraphNode:
        return GraphNode.model_validate(row)

    @staticmethod
    def _parse_edge(row: dict) -> GraphEdge:
        return GraphEdge.model_validate(row)

    # ── Repo CRUD ────────────────────────────────────────────────────────

    def save_repo(
        self,
        repo: RepoInfo,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> None:
        with self._lock:
            # Clear any existing data for this repo
            self._conn.execute("DELETE FROM edges WHERE repo_id = ?", (repo.repo_id,))
            self._conn.execute("DELETE FROM nodes WHERE repo_id = ?", (repo.repo_id,))
            self._conn.execute("DELETE FROM repos WHERE repo_id = ?", (repo.repo_id,))

            self._conn.execute(
                "INSERT INTO repos (repo_id, repo_path, repo_url, name, "
                "indexed_at, file_count, node_count, edge_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    repo.repo_id, repo.repo_path, repo.repo_url, repo.name,
                    repo.indexed_at, repo.file_count, repo.node_count, repo.edge_count,
                ),
            )

            if nodes:
                self._conn.executemany(
                    "INSERT INTO nodes (node_id, repo_id, node_type, name, file_path, "
                    "line_start, line_end, docstring, decorators, parameters, bases, "
                    "module_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            n.node_id, repo.repo_id, n.node_type, n.name, n.file_path,
                            n.line_start, n.line_end, n.docstring,
                            json.dumps(n.decorators), json.dumps(n.parameters),
                            json.dumps(n.bases), n.module_path,
                        )
                        for n in nodes
                    ],
                )

            if edges:
                self._conn.executemany(
                    "INSERT INTO edges (repo_id, source, target, edge_type, "
                    "file_path, line) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            repo.repo_id, e.source, e.target, e.edge_type,
                            e.file_path, e.line,
                        )
                        for e in edges
                    ],
                )

            self._conn.commit()

        # Invalidate cached graph
        self._graphs.pop(repo.repo_id, None)

    def delete_repo(self, repo_id: str) -> bool:
        row = self._fetchone("SELECT repo_id FROM repos WHERE repo_id = ?", (repo_id,))
        if not row:
            return False
        with self._lock:
            self._conn.execute("DELETE FROM edges WHERE repo_id = ?", (repo_id,))
            self._conn.execute("DELETE FROM nodes WHERE repo_id = ?", (repo_id,))
            self._conn.execute("DELETE FROM repos WHERE repo_id = ?", (repo_id,))
            self._conn.commit()
        self._graphs.pop(repo_id, None)
        return True

    def list_repos(self) -> list[RepoInfo]:
        rows = self._fetchall("SELECT * FROM repos ORDER BY indexed_at DESC")
        return [RepoInfo.model_validate(r) for r in rows]

    def get_repo(self, repo_id: str) -> RepoInfo | None:
        row = self._fetchone("SELECT * FROM repos WHERE repo_id = ?", (repo_id,))
        return RepoInfo.model_validate(row) if row else None

    # ── networkx graph loading ───────────────────────────────────────────

    def _load_graph(self, repo_id: str) -> nx.DiGraph:
        g = nx.DiGraph()
        nodes = self._fetchall(
            "SELECT * FROM nodes WHERE repo_id = ?", (repo_id,)
        )
        for row in nodes:
            node = self._parse_node(row)
            g.add_node(
                node.node_id,
                node_type=node.node_type,
                name=node.name,
                file_path=node.file_path,
                line_start=node.line_start,
                line_end=node.line_end,
                docstring=node.docstring,
                decorators=node.decorators,
                parameters=node.parameters,
                bases=node.bases,
                module_path=node.module_path,
                summary=node.summary,
                cluster_id=node.cluster_id,
            )
        edges = self._fetchall(
            "SELECT * FROM edges WHERE repo_id = ?", (repo_id,)
        )
        for row in edges:
            edge = self._parse_edge(row)
            g.add_edge(
                edge.source,
                edge.target,
                edge_type=edge.edge_type,
                file_path=edge.file_path,
                line=edge.line,
            )
        return g

    def _get_graph(self, repo_id: str) -> nx.DiGraph:
        if repo_id not in self._graphs:
            self._graphs[repo_id] = self._load_graph(repo_id)
        return self._graphs[repo_id]

    # ── Node queries ─────────────────────────────────────────────────────

    def query_nodes(
        self,
        repo_id: str,
        node_type: str | None = None,
        name: str | None = None,
        file_path: str | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        clauses = ["repo_id = ?"]
        params: list[str | int] = [repo_id]
        if node_type:
            clauses.append("node_type = ?")
            params.append(node_type)
        if name:
            clauses.append("name LIKE ?")
            params.append(f"%{name}%")
        if file_path:
            clauses.append("file_path LIKE ?")
            params.append(f"%{file_path}%")
        where = " AND ".join(clauses)
        params.append(limit)
        rows = self._fetchall(
            f"SELECT * FROM nodes WHERE {where} ORDER BY file_path, line_start LIMIT ?",
            tuple(params),
        )
        return [self._parse_node(r) for r in rows]

    # ── Edge queries ─────────────────────────────────────────────────────

    def get_edges(
        self,
        repo_id: str,
        node_id: str,
        direction: str = "both",
        edge_type: str | None = None,
    ) -> dict:
        incoming: list[GraphEdge] = []
        outgoing: list[GraphEdge] = []

        if direction in ("incoming", "both"):
            clauses = ["repo_id = ?", "target = ?"]
            params: list[str] = [repo_id, node_id]
            if edge_type:
                clauses.append("edge_type = ?")
                params.append(edge_type)
            rows = self._fetchall(
                f"SELECT * FROM edges WHERE {' AND '.join(clauses)}",
                tuple(params),
            )
            incoming = [self._parse_edge(r) for r in rows]

        if direction in ("outgoing", "both"):
            clauses = ["repo_id = ?", "source = ?"]
            params = [repo_id, node_id]
            if edge_type:
                clauses.append("edge_type = ?")
                params.append(edge_type)
            rows = self._fetchall(
                f"SELECT * FROM edges WHERE {' AND '.join(clauses)}",
                tuple(params),
            )
            outgoing = [self._parse_edge(r) for r in rows]

        return {"incoming": incoming, "outgoing": outgoing}

    # ── Subgraph extraction ──────────────────────────────────────────────

    def get_subgraph(
        self,
        repo_id: str,
        node_id: str,
        depth: int = 2,
        edge_types: list[str] | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        g = self._get_graph(repo_id)
        if node_id not in g:
            return [], []

        # Filter edges by type if requested
        if edge_types:
            view = nx.subgraph_view(
                g,
                filter_edge=lambda u, v: g.edges[u, v].get("edge_type") in edge_types,
            )
        else:
            view = g

        # BFS to collect nodes within depth
        visited: set[str] = set()
        frontier = {node_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for n in frontier:
                if n in visited:
                    continue
                visited.add(n)
                next_frontier.update(view.successors(n))
                next_frontier.update(view.predecessors(n))
            frontier = next_frontier - visited
        visited.update(frontier)

        # Collect nodes and edges from the subgraph
        # Skip stub nodes (created by add_edge but never defined by the parser)
        sub_nodes: list[GraphNode] = []
        for nid in visited:
            if nid in g.nodes:
                attrs = dict(g.nodes[nid])
                if "node_type" not in attrs:
                    continue  # stub node — edge target with no definition
                sub_nodes.append(GraphNode(node_id=nid, **attrs))

        sub_edges: list[GraphEdge] = []
        for u, v, data in g.edges(data=True):
            if u in visited and v in visited:
                if not edge_types or data.get("edge_type") in edge_types:
                    sub_edges.append(GraphEdge(source=u, target=v, **data))

        return sub_nodes, sub_edges

    # ── Shortest path ────────────────────────────────────────────────────

    def find_path(
        self,
        repo_id: str,
        source_id: str,
        target_id: str,
    ) -> tuple[list[str], list[GraphEdge]]:
        g = self._get_graph(repo_id)
        if source_id not in g or target_id not in g:
            return [], []
        try:
            path = nx.shortest_path(g, source_id, target_id)
        except nx.NetworkXNoPath:
            # Try undirected
            try:
                path = nx.shortest_path(g.to_undirected(), source_id, target_id)
            except nx.NetworkXNoPath:
                return [], []

        path_edges: list[GraphEdge] = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if g.has_edge(u, v):
                data = dict(g.edges[u, v])
                path_edges.append(GraphEdge(source=u, target=v, **data))
            elif g.has_edge(v, u):
                data = dict(g.edges[v, u])
                path_edges.append(GraphEdge(source=v, target=u, **data))
        return path, path_edges

    # ── Export ───────────────────────────────────────────────────────────

    def export_graph(self, repo_id: str, format: str = "d3") -> dict:
        g = self._get_graph(repo_id)
        repo = self.get_repo(repo_id)

        d3_nodes = []
        for nid, attrs in g.nodes(data=True):
            if "node_type" not in attrs:
                continue  # stub node
            node_data: dict = {
                "id": nid,
                "name": attrs.get("name", ""),
                "type": attrs.get("node_type", ""),
                "file": attrs.get("file_path", ""),
                "line": attrs.get("line_start", 0),
                "group": attrs.get("file_path", ""),
                "size": g.degree(nid),
            }
            if format == "full":
                node_data["docstring"] = attrs.get("docstring")
                node_data["decorators"] = attrs.get("decorators", [])
                node_data["parameters"] = attrs.get("parameters", [])
                node_data["bases"] = attrs.get("bases", [])
                node_data["module_path"] = attrs.get("module_path", "")
                node_data["line_end"] = attrs.get("line_end")
                node_data["summary"] = attrs.get("summary")
                node_data["cluster_id"] = attrs.get("cluster_id")
            d3_nodes.append(node_data)

        d3_links = []
        for u, v, data in g.edges(data=True):
            d3_links.append({
                "source": u,
                "target": v,
                "type": data.get("edge_type", ""),
                "value": 1,
            })

        return {
            "nodes": d3_nodes,
            "links": d3_links,
            "metadata": {
                "repo_id": repo_id,
                "name": repo.name if repo else "",
                "node_count": g.number_of_nodes(),
                "edge_count": g.number_of_edges(),
            },
        }

    # ── Stats ────────────────────────────────────────────────────────────

    def get_stats(self, repo_id: str) -> dict:
        repo = self.get_repo(repo_id)
        if not repo:
            return {"error": f"Repo not found: {repo_id}"}

        nodes_by_type = {}
        for nt in NodeType:
            rows = self._fetchall(
                "SELECT COUNT(*) as n FROM nodes WHERE repo_id = ? AND node_type = ?",
                (repo_id, nt),
            )
            nodes_by_type[nt] = rows[0]["n"] if rows else 0

        edges_by_type = {}
        for et in EdgeType:
            rows = self._fetchall(
                "SELECT COUNT(*) as n FROM edges WHERE repo_id = ? AND edge_type = ?",
                (repo_id, et),
            )
            edges_by_type[et] = rows[0]["n"] if rows else 0

        # Top connected nodes
        g = self._get_graph(repo_id)
        degree_list = sorted(g.degree(), key=lambda x: x[1], reverse=True)[:10]
        top_connected = [
            {"node_id": nid, "connections": deg}
            for nid, deg in degree_list
        ]

        total_degree = sum(d for _, d in g.degree())
        node_count = g.number_of_nodes()

        return {
            "repo_id": repo_id,
            "name": repo.name,
            "file_count": repo.file_count,
            "node_count": repo.node_count,
            "edge_count": repo.edge_count,
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
            "top_connected": top_connected,
            "avg_connections": round(total_degree / node_count, 2) if node_count else 0,
        }

    # ── LLM enrichment methods ──────────────────────────────────────────

    def get_nodes_without_summary(
        self, repo_id: str, limit: int = 500,
    ) -> list[GraphNode]:
        rows = self._fetchall(
            "SELECT * FROM nodes WHERE repo_id = ? AND summary IS NULL "
            "AND node_type != 'module' ORDER BY file_path, line_start LIMIT ?",
            (repo_id, limit),
        )
        return [self._parse_node(r) for r in rows]

    def update_node_summaries(
        self, repo_id: str, summaries: list[tuple[str, str]],
    ) -> int:
        with self._lock:
            for node_id, summary in summaries:
                self._conn.execute(
                    "UPDATE nodes SET summary = ? WHERE repo_id = ? AND node_id = ?",
                    (summary, repo_id, node_id),
                )
            self._conn.commit()
        self._graphs.pop(repo_id, None)
        return len(summaries)

    def update_node_clusters(
        self, repo_id: str, assignments: dict[str, str],
    ) -> None:
        with self._lock:
            for node_id, cluster_id in assignments.items():
                self._conn.execute(
                    "UPDATE nodes SET cluster_id = ? WHERE repo_id = ? AND node_id = ?",
                    (cluster_id, repo_id, node_id),
                )
            self._conn.commit()
        self._graphs.pop(repo_id, None)

    def save_clusters(
        self, repo_id: str, clusters: list[ClusterInfo],
    ) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM clusters WHERE repo_id = ?", (repo_id,)
            )
            for c in clusters:
                self._conn.execute(
                    "INSERT INTO clusters (cluster_id, repo_id, label, "
                    "description, member_ids) VALUES (?, ?, ?, ?, ?)",
                    (c.cluster_id, repo_id, c.label, c.description,
                     json.dumps(c.member_ids)),
                )
            self._conn.commit()

    def get_clusters(self, repo_id: str) -> list[ClusterInfo]:
        rows = self._fetchall(
            "SELECT * FROM clusters WHERE repo_id = ? ORDER BY cluster_id",
            (repo_id,),
        )
        return [ClusterInfo.model_validate(r) for r in rows]

    def save_embeddings(
        self, repo_id: str, embeddings: list[tuple[str, bytes, str]],
    ) -> None:
        """Save embeddings as (node_id, embedding_bytes, model)."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM embeddings WHERE repo_id = ?", (repo_id,)
            )
            self._conn.executemany(
                "INSERT INTO embeddings (node_id, repo_id, embedding, model) "
                "VALUES (?, ?, ?, ?)",
                [(nid, repo_id, emb, model) for nid, emb, model in embeddings],
            )
            self._conn.commit()

    def get_embeddings(self, repo_id: str) -> list[tuple[str, bytes]]:
        """Return (node_id, embedding_bytes) pairs."""
        rows = self._fetchall(
            "SELECT node_id, embedding FROM embeddings WHERE repo_id = ?",
            (repo_id,),
        )
        return [(r["node_id"], r["embedding"]) for r in rows]

    def add_semantic_edges(
        self, repo_id: str, edges: list[GraphEdge],
    ) -> int:
        """Add semantic similarity edges. Clears existing ones first."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM edges WHERE repo_id = ? AND edge_type = ?",
                (repo_id, EdgeType.SEMANTIC_SIMILARITY),
            )
            if edges:
                self._conn.executemany(
                    "INSERT INTO edges (repo_id, source, target, edge_type, "
                    "file_path, line) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (repo_id, e.source, e.target, e.edge_type, None, None)
                        for e in edges
                    ],
                )
            self._conn.commit()
        self._graphs.pop(repo_id, None)
        return len(edges)
