"""照腳本回答的檢索器——供 pipeline 端到端跑通,不是真實檢索策略。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class ScriptedRetriever:
    """對每個問題回傳預先指定的結果;問題不在腳本中即 raise。

    刻意不對未知問題回傳空結果:那會讓「漏題」偽裝成「檢索不到」,
    使平均分虛低卻毫無徵兆。
    """

    def __init__(self, script: Mapping[str, Sequence[str]]) -> None:
        self._script = {question: tuple(docs) for question, docs in script.items()}

    def retrieve(self, question: str, top_k: int) -> Sequence[str]:
        if question not in self._script:
            raise ValueError(f"no scripted result for question: {question!r}")
        return self._script[question][:top_k]


class ScoredScriptedRetriever(ScriptedRetriever):
    """帶分數的腳本檢索器:script 值為 [[id, score], …];供拒答門檻的無 GPU 真實執行。"""

    def __init__(self, script: Mapping[str, Sequence[Sequence[object]]]) -> None:
        self._scored: dict[str, tuple[tuple[str, float], ...]] = {}
        for question, rows in script.items():
            pairs = []
            for row in rows:
                if len(row) != 2:
                    raise ValueError(f"scored script row must be [id, score]: {row!r}")
                pairs.append((str(row[0]), float(row[1])))  # type: ignore[arg-type]
            self._scored[question] = tuple(pairs)
        super().__init__({q: [c for c, _ in pairs] for q, pairs in self._scored.items()})

    def retrieve_scored(self, question: str, top_k: int) -> Sequence[tuple[str, float]]:
        if question not in self._scored:
            raise ValueError(f"no scripted result for question: {question!r}")
        return self._scored[question][:top_k]
