"""Boundary guardrails for the public tin package."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PKG = _ROOT / "tin"
_ALLOWED_ROOTS = frozenset(sys.stdlib_module_names) | {"numpy", "tin"}


def _iter_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.level == 0:
                yield node.lineno, "from", node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, "import", alias.name


class TestImportBoundary:
    """The public tin package imports only the stdlib, numpy, and itself."""

    def test_tin_package_imports_stay_inside_the_boundary(self):
        violations: list[str] = []
        for py in _PKG.rglob("*.py"):
            for lineno, kind, name in _iter_imports(py):
                root = name.split(".")[0]
                if root not in _ALLOWED_ROOTS:
                    violations.append(f"{py.relative_to(_ROOT)}:{lineno}: {kind} {name}")

        assert violations == [], (
            "tin/ imports outside the allowed set (stdlib, numpy, tin):\n"
            + "\n".join(violations)
        )
