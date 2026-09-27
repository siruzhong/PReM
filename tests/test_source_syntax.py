import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("models", "qwen_vl_utils", "scripts")


def source_files():
    return [
        path
        for root in SOURCE_ROOTS
        for path in sorted((REPO_ROOT / root).rglob("*.py"))
    ]


@pytest.mark.parametrize(
    "path",
    source_files(),
    ids=lambda path: str(path.relative_to(REPO_ROOT)),
)
def test_python_source_parses(path: Path):
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
