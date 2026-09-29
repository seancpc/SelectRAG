"""tools/mutants.py:基線檢查與擊殺可稽核性(2026-09-24)。文字皆為中性占位符。

背景:桌機 gauntlet 第 4b 層報 140/162,雲端同程式碼 85/162。單獨重跑其中一個
「桌機被殺」的變異,桌機本機也是存活——gauntlet 當下的擊殺是假的,但原因查不出來,
因為 run_suite 只留結束碼、丟掉整份輸出。本檔鎖住兩件事:
①基線(未植入變異)不綠時必須中止,否則每個變異都會被記成「殺死」;
②每個擊殺都要印出是哪個測試、什麼例外,讓假擊殺看得見。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mutants_tool", TOOLS / "mutants.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mutants_tool"] = mod
    spec.loader.exec_module(mod)
    return mod


mt = _load()


def test_failure_summary_names_test_and_exception() -> None:
    out = (
        "E   assert 1 == 2\n"
        "\n"
        "tests/<檔>.py:12: AssertionError\n"
        "=== short test summary info ===\n"
        "FAILED tests/<檔>.py::<測試>\n"
    )
    got = mt.failure_summary(1, out, "")
    assert "FAILED tests/<檔>.py::<測試>" in got
    assert "tests/<檔>.py:12: AssertionError" in got


def test_failure_summary_falls_back_to_rc_and_last_line() -> None:
    # 沒選到任何測試(rc 5)或收集中斷時沒有 FAILED 行:至少要看得到結束碼與最後訊息
    got = mt.failure_summary(5, "\n<最後一行訊息>\n\n", "")
    assert got == "rc=5 <最後一行訊息>"


def test_main_aborts_before_mutating_when_baseline_is_red(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    calls: list[Any] = []

    def fake_run(selector: str | None) -> tuple[bool, str]:
        calls.append(selector)
        return False, "FAILED tests/<檔>.py::<測試>"

    monkeypatch.setattr(mt, "MUTANTS", [])  # 不碰真實檔案:變異層執行中本測試也會被跑到
    monkeypatch.setattr(mt, "run_suite", fake_run)
    monkeypatch.setattr(sys, "argv", ["mutants.py", "<選擇器>"])
    assert mt.main() == 4
    assert calls == ["<選擇器>"]  # 只跑了基線一次,沒有進入變異迴圈
    out = capsys.readouterr().out
    assert "基線" in out and "FAILED tests/<檔>.py::<測試>" in out


def test_main_prints_reason_for_each_kill(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    target = tmp_path / "t.py"
    target.write_text("<前> OLD <後>\n", encoding="utf-8")
    monkeypatch.setattr(mt, "MUTANTS", [(target, "<變異名>", "<失效模式>", "OLD", "NEW")])
    results = iter(
        [(True, ""), (False, "FAILED tests/<檔>.py::<測試>  [tests/<檔>.py:3: KeyError]")]
    )
    monkeypatch.setattr(mt, "run_suite", lambda _s: next(results))
    monkeypatch.setattr(sys, "argv", ["mutants.py"])
    assert mt.main() == 0
    out = capsys.readouterr().out
    assert "✓ 殺死 <變異名>" in out
    assert "FAILED tests/<檔>.py::<測試>" in out and "KeyError" in out
    assert target.read_text(encoding="utf-8") == "<前> OLD <後>\n"  # 一定還原


@pytest.mark.parametrize("rc", [0, 1])
def test_run_suite_returns_pass_flag_and_reason(monkeypatch, rc: int) -> None:  # type: ignore[no-untyped-def]
    class R:
        returncode = rc
        stdout = "FAILED tests/<檔>.py::<測試>\n" if rc else "\n"
        stderr = ""

    monkeypatch.setattr(mt.subprocess, "run", lambda *_a, **_k: R())
    ok, why = mt.run_suite(None)
    assert ok is (rc == 0)
    assert (why == "") if rc == 0 else ("FAILED" in why)
