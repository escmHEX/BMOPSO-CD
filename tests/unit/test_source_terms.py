from __future__ import annotations

from pathlib import Path


def test_removed_runtime_test_backend_terms_do_not_return():
    root = Path(__file__).resolve().parents[2]
    forbidden = ["".join(("fa", "ke")), "".join(("mo", "ck"))]
    paths = [
        *list((root / "src").rglob("*.py")),
        *list((root / "tests").rglob("*.py")),
        *list((root / "configs").rglob("*.yaml")),
        root / "README.md",
        root / "IMPLEMENTATION_NOTES.md",
    ]
    offenders = []
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        if any(term in text for term in forbidden):
            offenders.append(path.relative_to(root).as_posix())
    assert offenders == []
