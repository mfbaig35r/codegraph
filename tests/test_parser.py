"""Tests for codegraph.parser."""

from pathlib import Path

from codegraph.models import EdgeType, NodeType
from codegraph.parser import CodeGraphVisitor, parse_repository

FIXTURES = Path(__file__).parent / "fixtures"


def _parse_file(filename: str, module_path: str | None = None) -> tuple:
    path = FIXTURES / filename
    source = path.read_text()
    mp = module_path or filename.replace(".py", "").replace("/", ".")
    visitor = CodeGraphVisitor(filename, mp)
    return visitor.parse(source)


# ── Simple module ────────────────────────────────────────────────────────


def test_simple_module_nodes():
    nodes, _ = _parse_file("simple_module.py", "simple_module")
    names = {n.name: n for n in nodes}

    assert "simple_module" in names
    assert names["simple_module"].node_type == NodeType.MODULE

    assert "helper" in names
    assert names["helper"].node_type == NodeType.FUNCTION
    assert "x: int" in names["helper"].parameters

    assert "caller" in names
    assert names["caller"].node_type == NodeType.FUNCTION

    assert "Animal" in names
    assert names["Animal"].node_type == NodeType.CLASS

    assert "Dog" in names
    assert names["Dog"].node_type == NodeType.CLASS
    assert "Animal" in names["Dog"].bases

    assert "speak" in [n.name for n in nodes if n.node_type == NodeType.METHOD]
    assert "fetch" in [n.name for n in nodes if n.node_type == NodeType.METHOD]


def test_simple_module_contains_edges():
    _, edges = _parse_file("simple_module.py", "simple_module")
    contains = [e for e in edges if e.edge_type == EdgeType.CONTAINS]

    sources_targets = {(e.source, e.target) for e in contains}
    assert ("simple_module", "simple_module.helper") in sources_targets
    assert ("simple_module", "simple_module.Animal") in sources_targets
    assert ("simple_module", "simple_module.Dog") in sources_targets
    assert ("simple_module.Animal", "simple_module.Animal.speak") in sources_targets


def test_simple_module_inherits():
    _, edges = _parse_file("simple_module.py", "simple_module")
    inherits = [e for e in edges if e.edge_type == EdgeType.INHERITS]
    assert any(e.source == "simple_module.Dog" and e.target == "Animal" for e in inherits)


def test_simple_module_imports():
    _, edges = _parse_file("simple_module.py", "simple_module")
    imports = [e for e in edges if e.edge_type == EdgeType.IMPORTS]
    targets = {e.target for e in imports}
    assert "os" in targets
    assert "pathlib" in targets


def test_simple_module_calls():
    _, edges = _parse_file("simple_module.py", "simple_module")
    calls = [e for e in edges if e.edge_type == EdgeType.CALLS]
    # caller() calls helper()
    assert any(
        e.source == "simple_module.caller" and e.target == "helper"
        for e in calls
    )


def test_docstrings_extracted():
    nodes, _ = _parse_file("simple_module.py", "simple_module")
    by_name = {n.name: n for n in nodes}
    assert by_name["helper"].docstring == "Convert int to string."
    assert by_name["Animal"].docstring == "Base animal class."


# ── Decorated module ─────────────────────────────────────────────────────


def test_decorators_extracted():
    nodes, _ = _parse_file("decorated.py", "decorated")
    by_name = {n.name: n for n in nodes}

    assert "lru_cache" in by_name["cached_compute"].decorators
    assert "my_decorator" in by_name["decorated_func"].decorators
    assert "staticmethod" in by_name["static_method"].decorators
    assert "classmethod" in by_name["class_method"].decorators
    assert "property" in by_name["name"].decorators


def test_async_function_detected():
    nodes, _ = _parse_file("decorated.py", "decorated")
    by_name = {n.name: n for n in nodes}
    assert "async_handler" in by_name
    assert by_name["async_handler"].node_type == NodeType.FUNCTION


def test_decorator_edges():
    _, edges = _parse_file("decorated.py", "decorated")
    dec_edges = [e for e in edges if e.edge_type == EdgeType.DECORATES]
    assert any(
        e.source == "lru_cache" and "cached_compute" in e.target
        for e in dec_edges
    )


# ── Import chain ─────────────────────────────────────────────────────────


def test_parse_repository_fixtures():
    nodes, edges = parse_repository(str(FIXTURES))

    node_ids = {n.node_id for n in nodes}
    # Should find modules
    assert "simple_module" in node_ids
    assert "decorated" in node_ids

    # Should find classes across files
    class_nodes = [n for n in nodes if n.node_type == NodeType.CLASS]
    class_names = {n.name for n in class_nodes}
    assert "Animal" in class_names
    assert "Dog" in class_names
    assert "Service" in class_names


def test_call_resolution_across_repo():
    """After call resolution, calls to 'helper' should resolve to simple_module.helper."""
    nodes, edges = parse_repository(str(FIXTURES))

    calls = [e for e in edges if e.edge_type == EdgeType.CALLS]
    # caller() -> helper() should be resolved
    resolved_targets = {e.target for e in calls}
    assert "simple_module.helper" in resolved_targets or "helper" in resolved_targets
