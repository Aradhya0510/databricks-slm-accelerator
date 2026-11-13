"""Repo-wide invariants that nothing else checks.

Notebooks and Streamlit pages are never imported by the rest of the suite, so
a mistake in them ships silently. These are cheap and catch that.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

PYTHON_FILES = sorted(
    p for p in REPO.rglob("*.py")
    if ".git" not in p.parts
    and ".github" not in p.parts
    and "build" not in p.parts
)


def test_there_are_python_files_to_check():
    assert len(PYTHON_FILES) > 10


@pytest.mark.parametrize(
    "path", PYTHON_FILES, ids=lambda p: str(p.relative_to(REPO)),
)
def test_file_parses(path):
    """Every Python file in the repo must be syntactically valid."""
    ast.parse(path.read_text(), filename=str(path))


NOTEBOOKS = sorted(
    [*(REPO / "jobs").glob("*.py"), *(REPO / "notebooks").glob("*.py")]
)

_HEADER = "# Databricks notebook source"
_CELL = "# COMMAND ----------"


def _cells(path):
    return path.read_text().removeprefix(_HEADER).split(_CELL)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(REPO)))
def test_notebooks_are_databricks_source(path):
    """Without the header, a bundle deploys the file as a plain script, and a
    notebook task pointed at it fails to start."""
    assert path.read_text().startswith(_HEADER), f"{path.name} lacks the notebook header"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(REPO)))
def test_magic_cells_hold_only_magic(path):
    """A plain Python line ahead of ``# MAGIC %pip`` in the same cell turns the
    whole cell into Python, and the install silently never runs."""
    for cell in _cells(path):
        lines = [l for l in cell.strip().splitlines() if l.strip()]
        if any(l.startswith("# MAGIC %") for l in lines):
            stray = [l for l in lines if not l.startswith("# MAGIC")]
            assert not stray, f"{path.name}: non-magic line in a magic cell: {stray[0]!r}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(REPO)))
def test_notebooks_install_from_the_pinned_requirements(path):
    """Every install goes through requirements.txt, as the full set or as
    constraints, so a notebook cannot drift onto versions nothing tested."""
    for line in path.read_text().splitlines():
        if line.startswith("# MAGIC %pip install"):
            assert "requirements.txt" in line, f"{path.name}: unpinned install: {line!r}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(REPO)))
def test_notebooks_do_not_add_src_to_the_path(path):
    """Adding both the root and src/ makes one module importable under two names.

    ``import config`` and ``import src.config`` would then resolve to two
    different module objects, each with its own registry state.
    """
    source = path.read_text()
    assert '"src"' not in source and "'src'" not in source, (
        f"{path.name} appears to add src/ to sys.path"
    )
