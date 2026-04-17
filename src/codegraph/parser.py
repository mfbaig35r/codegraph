"""AST-based Python code parser for codegraph."""

import ast
import logging
from pathlib import Path

from .models import EdgeType, GraphEdge, GraphNode, NodeType

log = logging.getLogger("codegraph.parser")


def _path_to_module(rel_path: Path) -> str:
    """Convert a relative file path to a dotted module path."""
    parts = list(rel_path.with_suffix("").parts)
    # Drop __init__ — the package is the parent
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else "__root__"


def _get_docstring(node: ast.AST) -> str | None:
    """Extract docstring from a function or class node."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
        return None
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].value.value
    return None


def _get_end_line(node: ast.AST) -> int | None:
    """Get end_lineno if available (Python 3.8+)."""
    return getattr(node, "end_lineno", None)


def _format_arg(arg: ast.arg) -> str:
    """Format a function argument as a string."""
    name = arg.arg
    if arg.annotation:
        try:
            return f"{name}: {ast.unparse(arg.annotation)}"
        except Exception:
            return name
    return name


def _format_params(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Extract parameter list from a function definition."""
    params: list[str] = []
    args = func_node.args
    for a in args.posonlyargs + args.args + args.kwonlyargs:
        params.append(_format_arg(a))
    if args.vararg:
        params.append(f"*{_format_arg(args.vararg)}")
    if args.kwarg:
        params.append(f"**{_format_arg(args.kwarg)}")
    return params


def _resolve_decorator(node: ast.expr) -> str:
    """Extract decorator name from its AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        try:
            return ast.unparse(node)
        except Exception:
            return node.attr
    if isinstance(node, ast.Call):
        return _resolve_decorator(node.func)
    try:
        return ast.unparse(node)
    except Exception:
        return "<unknown>"


def _resolve_call_name(node: ast.Call) -> str | None:
    """Extract the callee name from a Call node. Returns None if unresolvable."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        # e.g. self.method(), obj.func()
        try:
            return ast.unparse(func)
        except Exception:
            return func.attr
    return None


class CodeGraphVisitor(ast.NodeVisitor):
    """Extract code graph nodes and edges from a single Python file."""

    def __init__(self, file_path: str, module_path: str) -> None:
        self._file_path = file_path
        self._module_path = module_path
        self._nodes: list[GraphNode] = []
        self._edges: list[GraphEdge] = []
        self._scope_stack: list[str] = [module_path]
        self._class_stack: list[str] = []  # Track if we're inside a class

    @property
    def _current_scope(self) -> str:
        return self._scope_stack[-1]

    def parse(self, source: str) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Parse source code and return extracted nodes and edges."""
        tree = ast.parse(source, filename=self._file_path)

        # Create module node
        self._nodes.append(GraphNode(
            node_id=self._module_path,
            node_type=NodeType.MODULE,
            name=(
                self._module_path.split(".")[-1]
                if "." in self._module_path
                else self._module_path
            ),
            file_path=self._file_path,
            line_start=1,
            line_end=len(source.splitlines()),
            docstring=_get_docstring(tree),
            module_path=self._module_path,
        ))

        self.visit(tree)
        return self._nodes, self._edges

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_id = f"{self._current_scope}.{node.name}"

        # Extract base classes
        bases: list[str] = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                if isinstance(base, ast.Name):
                    bases.append(base.id)

        # Extract decorators
        decorators = [_resolve_decorator(d) for d in node.decorator_list]

        self._nodes.append(GraphNode(
            node_id=class_id,
            node_type=NodeType.CLASS,
            name=node.name,
            file_path=self._file_path,
            line_start=node.lineno,
            line_end=_get_end_line(node),
            docstring=_get_docstring(node),
            decorators=decorators,
            bases=bases,
            module_path=self._module_path,
        ))

        # Contains edge: parent scope → class
        self._edges.append(GraphEdge(
            source=self._current_scope,
            target=class_id,
            edge_type=EdgeType.CONTAINS,
            file_path=self._file_path,
            line=node.lineno,
        ))

        # Inherits edges
        for base_name in bases:
            self._edges.append(GraphEdge(
                source=class_id,
                target=base_name,
                edge_type=EdgeType.INHERITS,
                file_path=self._file_path,
                line=node.lineno,
            ))

        # Decorator edges
        for dec_name in decorators:
            self._edges.append(GraphEdge(
                source=dec_name,
                target=class_id,
                edge_type=EdgeType.DECORATES,
                file_path=self._file_path,
                line=node.lineno,
            ))

        # Visit children in class scope
        self._scope_stack.append(class_id)
        self._class_stack.append(class_id)
        self.generic_visit(node)
        self._class_stack.pop()
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node)

    def _handle_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        func_id = f"{self._current_scope}.{node.name}"
        is_method = bool(self._class_stack)
        node_type = NodeType.METHOD if is_method else NodeType.FUNCTION

        decorators = [_resolve_decorator(d) for d in node.decorator_list]
        parameters = _format_params(node)

        self._nodes.append(GraphNode(
            node_id=func_id,
            node_type=node_type,
            name=node.name,
            file_path=self._file_path,
            line_start=node.lineno,
            line_end=_get_end_line(node),
            docstring=_get_docstring(node),
            decorators=decorators,
            parameters=parameters,
            module_path=self._module_path,
        ))

        # Contains edge
        self._edges.append(GraphEdge(
            source=self._current_scope,
            target=func_id,
            edge_type=EdgeType.CONTAINS,
            file_path=self._file_path,
            line=node.lineno,
        ))

        # Decorator edges
        for dec_name in decorators:
            self._edges.append(GraphEdge(
                source=dec_name,
                target=func_id,
                edge_type=EdgeType.DECORATES,
                file_path=self._file_path,
                line=node.lineno,
            ))

        # Visit children in function scope (for calls)
        self._scope_stack.append(func_id)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            self._edges.append(GraphEdge(
                source=self._module_path,
                target=module_name,
                edge_type=EdgeType.IMPORTS,
                file_path=self._file_path,
                line=node.lineno,
            ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            # Resolve relative imports
            target = node.module
            if node.level and node.level > 0:
                parts = self._module_path.split(".")
                if node.level < len(parts):
                    prefix = ".".join(parts[: -node.level])
                    target = f"{prefix}.{node.module}"
                # else: level >= parts depth — keep node.module as-is
            self._edges.append(GraphEdge(
                source=self._module_path,
                target=target,
                edge_type=EdgeType.IMPORTS,
                file_path=self._file_path,
                line=node.lineno,
            ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _resolve_call_name(node)
        if callee and self._current_scope != self._module_path:
            # Only record calls from within functions/methods
            self._edges.append(GraphEdge(
                source=self._current_scope,
                target=callee,
                edge_type=EdgeType.CALLS,
                file_path=self._file_path,
                line=node.lineno,
            ))
        self.generic_visit(node)


def _resolve_calls(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> list[GraphEdge]:
    """Resolve call edges: match short names to fully qualified node IDs."""
    # Build lookup tables
    by_name: dict[str, list[str]] = {}
    by_id: set[str] = set()
    for node in nodes:
        by_id.add(node.node_id)
        by_name.setdefault(node.name, []).append(node.node_id)

    resolved: list[GraphEdge] = []
    for edge in edges:
        if edge.edge_type != EdgeType.CALLS:
            resolved.append(edge)
            continue

        target = edge.target
        # Already fully qualified and exists
        if target in by_id:
            resolved.append(edge)
            continue

        # Strip self. / cls. prefix for method calls
        bare = target
        for prefix in ("self.", "cls."):
            if bare.startswith(prefix):
                bare = bare[len(prefix):]
                break

        # Try to resolve: check if the source's parent class has this method
        source_parts = edge.source.split(".")
        candidates: list[str] = []

        # Look in the source's class scope
        if len(source_parts) >= 2:
            class_scope = ".".join(source_parts[:-1])
            class_method = f"{class_scope}.{bare}"
            if class_method in by_id:
                candidates.append(class_method)

        # Look in the source's module scope
        module_parts = edge.source.split(".")
        for i in range(len(module_parts) - 1, 0, -1):
            module_scope = ".".join(module_parts[:i])
            module_func = f"{module_scope}.{bare}"
            if module_func in by_id:
                candidates.append(module_func)
                break

        # Globally unique name
        if not candidates and bare in by_name and len(by_name[bare]) == 1:
            candidates.append(by_name[bare][0])

        if candidates:
            resolved.append(GraphEdge(
                source=edge.source,
                target=candidates[0],
                edge_type=EdgeType.CALLS,
                file_path=edge.file_path,
                line=edge.line,
            ))
        else:
            log.debug(
                "Unresolved call: %s -> %s (line %s)",
                edge.source, target, edge.line,
            )

    return resolved


def parse_repository(repo_path: str) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Parse all .py files in a repository and build the code graph."""
    root = Path(repo_path)
    all_nodes: list[GraphNode] = []
    all_edges: list[GraphEdge] = []

    for py_file in sorted(root.rglob("*.py")):
        # Skip hidden directories and common non-source paths
        rel_path = py_file.relative_to(root)
        parts = rel_path.parts
        if any(p.startswith(".") or p in ("__pycache__", "node_modules", ".git") for p in parts):
            continue

        module_path = _path_to_module(rel_path)
        source = py_file.read_text(errors="replace")

        try:
            visitor = CodeGraphVisitor(str(rel_path), module_path)
            nodes, edges = visitor.parse(source)
            all_nodes.extend(nodes)
            all_edges.extend(edges)
        except SyntaxError:
            continue

    # Second pass: resolve call targets
    all_edges = _resolve_calls(all_nodes, all_edges)

    return all_nodes, all_edges
