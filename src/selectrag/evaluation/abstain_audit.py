"""棄答的可見化與真值可及性(2026-09-22,發現 47 的處置)。

★ 檔名是 `abstain_audit` 而不是 `abstain`:`src/selectrag/demo/abstain.py` 早已存在
(拒答門檻網格掃描)。兩者概念不同——那裡是**門檻拒答**(top-1 分數低於門檻,
不打 LLM),這裡是**模型自答棄答**(回覆含棄答字樣)的表列化。同名會讓兩個概念混為一談。

背景:`src/selectrag/demo/core.py` 的 `sentence_support` 對含棄答字樣的句子 `continue`,
所以棄答從首輪起就不在 support 表裡——表上「沒有任何句子」的 (preset, 題) 組合
首輪 32 / v14 25 / v15 21,完整率與 support 一直只在「有作答的題」上計算。
而棄答本身會錯:v15 的 21 筆裡至少 11 筆真值 chunk 就在取回結果內,有的排第一。
**錯誤棄答是一整類被合併進別人分母而消失的錯型**(與 §6.8 發現 32 的引用錯置同型)。

刻意**不改** `sentence_support`:那一行的剔除被 `core.py` 的 uncited 計算依賴,
而 §5.3 的 faithfulness 數字建立在它上面,改它會讓備忘錄既有數字全部變動。
盲區在 support 表,就補在 support 層。

`abstain_marker` 由呼叫端傳入(`core.ABSTAIN`),不在本模組另存一份——
同一概念的字樣只能有一個定義處。
"""

from __future__ import annotations

from collections.abc import Collection, Sequence


def is_abstain(answer: str, abstain_marker: str) -> bool:
    """整段回答是否含棄答字樣。

    判「含」而非「等於」是刻意的:發現 48 的 `b4-q15` 是**第一句作答、第二句宣稱缺席**
    (「參考資料未提及其他限制」,而所引段落明寫該限制)。只認整段相等會漏掉這一型,
    而它正是最需要被看見的——句子沒有引用標記,兩個既有欄位都收不到它。

    空字樣會讓任何回答都算棄答,屬設定錯誤,一律 raise(2026-08-27 決策 4 的精神:
    壞輸入不得靜默,否則壞設定與真的全棄答無法區分)。
    """
    if not abstain_marker:
        raise ValueError("abstain_marker must not be empty")
    return abstain_marker in answer


def truth_rank(relevant_doc_ids: Collection[str], hit_chunk_ids: Sequence[str]) -> int | None:
    """回真值 chunk 在取回結果中的**最小 1-based 名次**,完全沒取到則 None。

    這一欄只報名次,**不判棄答對錯**:真值 chunk 排第一不等於那一段真的含答案
    (§6.6 的教訓——字面指標當分數就會變成措辭一致度的代理)。它的用途是把
    「該取到卻棄答」這一型排到人工視線前面,判定仍屬人類保留區。

    空真值集一律 raise(決策 4):靜默回 None 會與「真的沒取到」混為一談,
    而兩者的意思完全相反——一個是題目壞了,一個是檢索失敗。
    """
    if not relevant_doc_ids:
        raise ValueError("relevant_doc_ids must not be empty")
    wanted = set(relevant_doc_ids)
    for i, cid in enumerate(hit_chunk_ids, start=1):
        if cid in wanted:
            return i
    return None
