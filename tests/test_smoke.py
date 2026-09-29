"""骨架 smoke test:確認 package 可 import 且子模組齊備。"""

import importlib

import selectrag


def test_version_is_exposed() -> None:
    assert selectrag.__version__ == "0.1.0"


def test_all_submodules_importable() -> None:
    for name in ("corpus", "retrieval", "evaluation", "runner", "perf"):
        assert importlib.import_module(f"selectrag.{name}") is not None


def test_pyproject_mirrors_requirements() -> None:
    """pyproject 的釘選必須與 requirements*.txt 完全一致(單一真相源)。"""
    import re
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    def pins(path: Path) -> set[str]:
        return {
            line.split("#")[0].strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if re.match(r"^[A-Za-z0-9_.-]+==", line.strip())
        }

    assert set(project["dependencies"]) == pins(root / "requirements.txt")
    assert set(project["optional-dependencies"]["dev"]) == pins(root / "requirements-dev.txt")
