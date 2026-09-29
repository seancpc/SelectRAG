"""檢索器的共同介面。

BM25 / dense / hybrid / reranker 各策略都實作這個 Protocol,
runner 只依賴介面,不依賴任何特定實作。
"""

from collections.abc import Sequence
from typing import Protocol


class Retriever(Protocol):
    """依問題取回排序後的 doc id;排序即檢索名次,順序有語意。"""

    def retrieve(self, question: str, top_k: int) -> Sequence[str]:
        """回傳至多 top_k 筆 doc id,由最相關排到最不相關。"""
        ...


class ScoredRetriever(Protocol):
    """可露出分數的檢索器(拒答門檻用):與 retrieve 同序,附每筆的相關分數。"""

    def retrieve(self, question: str, top_k: int) -> Sequence[str]: ...

    def retrieve_scored(self, question: str, top_k: int) -> Sequence[tuple[str, float]]: ...
