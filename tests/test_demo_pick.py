"""tools/demo_pick.py:左輸右贏選題。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("demo_pick", Path("tools/demo_pick.py"))
assert _spec is not None and _spec.loader is not None
demo_pick = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo_pick)


def _q(item_id: str, hit3: float, rr: float) -> dict:
    return {"item_id": item_id, "hit": {"hit@3": hit3}, "reciprocal_rank": rr}


def test_pick_keeps_only_left_miss_right_hit_sorted_by_right_rr() -> None:
    lq = [_q("a", 0.0, 0.0), _q("b", 1.0, 1.0), _q("c", 0.0, 0.0), _q("d", 0.0, 0.0)]
    rq = [_q("a", 1.0, 0.5), _q("b", 1.0, 1.0), _q("c", 1.0, 1.0), _q("d", 0.0, 0.0)]
    left, right = {"per_query": lq}, {"per_query": rq}
    assert demo_pick.pick(left, right, 3) == ["c", "a"]


def test_pick_ignores_ids_missing_on_right() -> None:
    left = {"per_query": [_q("a", 0.0, 0.0)]}
    right = {"per_query": []}
    assert demo_pick.pick(left, right, 3) == []


def test_main_prints_question_text(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    left = tmp_path / "l.json"
    right = tmp_path / "r.json"
    ev = tmp_path / "e.json"
    left.write_text(json.dumps({"per_query": [_q("x", 0.0, 0.0)]}), encoding="utf-8")
    right.write_text(json.dumps({"per_query": [_q("x", 1.0, 1.0)]}), encoding="utf-8")
    ev_doc = {"manifest": {}, "items": [{"id": "x", "question": "<問題文字>"}]}
    ev.write_text(json.dumps(ev_doc), encoding="utf-8")
    assert demo_pick.main([str(left), str(right), str(ev)]) == 0
    out = capsys.readouterr().out
    assert "1 題" in out and "x\t<問題文字>" in out
