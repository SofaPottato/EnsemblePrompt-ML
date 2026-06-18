from string import Formatter
from typing import Dict, List, Optional
from .schemas import RESERVED_PAIR_FIELDS


def _safeFormat(template: str, fields: Dict) -> str:
    # 佔位符命名使用field
    # 如果模板佔位符對應的資料值是 None，代表來源資料缺值
    # fail-fast：raise 並指出哪些佔位符對應的值是 None。
    referencedFieldSet = {fieldName for _, fieldName, _, _ in Formatter().parse(template) if fieldName}
    noneFieldList = sorted(name for name in referencedFieldSet if name in fields and fields[name] is None)
    if noneFieldList:
        raise ValueError(f"[Formatter] 模板欄位值為 None（來源資料缺值）: {noneFieldList}")
    try:
        return template.format_map({fieldName: str(fieldValue) for fieldName, fieldValue in fields.items()})
    except KeyError as e:
        # 模板寫了但 fields 沒有 → 這是「模板佔位符與資料欄位對不上」的設定錯誤，而非資料問題，附上提示訊息幫助 debug。
        raise KeyError(f"[Formatter] 模板佔位符 {e} 在資料中不存在") from e


class PromptFormatter:
    """
    將 context 與 pairs 填入 taskTemplate / pairTemplate，產出 userPrompt。
    pairTemplate 存在 
    → 批次模式（多 pair 展開後塞入 {pairs}）。
    → 單筆模式（context + pairs[0] 合併填入）。
    """
    def __init__(self, taskTemplate: str, pairTemplate: Optional[str] = None,
                 pairColumns: Optional[List[str]] = None):
        self.taskTemplate = taskTemplate
        self.pairTemplate = pairTemplate
        self.pairColumns = pairColumns

    def format(self, contextDict: Dict, pairs: List[Dict]) -> str:
        """pairTemplate 存在 → 批次模式；否則 → 單筆模式。"""
        if self.pairTemplate:
            return self._formatBatch(contextDict, pairs)
        return self._formatSingle(contextDict, pairs)

    def _formatBatch(self, contextDict: Dict, pairs: List[Dict]) -> str:
        """批次模式：每個 pair 渲染後拼接，整段填入 {pairs} 佔位符。"""
        # 先把每個 pair 各自渲染成一段文字再串起來，最後當成單一 {pairs} 值塞進 taskTemplate。
        pairsText = ""
        for i, pairDict in enumerate(pairs, 1):
            pairsText += _safeFormat(self.pairTemplate, {'i': i, **self._extractPairFields(pairDict)})
        return _safeFormat(self.taskTemplate, {**contextDict, 'pairs': pairsText})

    def _formatSingle(self, contextDict: Dict, pairs: List[Dict]) -> str:
        """單筆模式：context 與 pairs[0] 合併後直接填入 taskTemplate。pair 欄優先，同名時覆蓋 context。"""     
        # PPI 只有一個 pair，沒有獨立的 pairTemplate，直接把 context 與 pair 欄位攤平餵進 taskTemplate。
        allFieldDict = dict(contextDict)
        if pairs:
            allFieldDict.update(self._extractPairFields(pairs[0]))
        return _safeFormat(self.taskTemplate, allFieldDict)

    def _extractPairFields(self, pairDict: Dict) -> Dict:
        """從 pair dict 抽取要送進模板的欄位，一律排除 RESERVED_PAIR_FIELDS 中的內部欄位。"""
        candidateNameList = self.pairColumns if self.pairColumns else list(pairDict.keys())
        return {
            name: pairDict[name]
            for name in candidateNameList
            if name in pairDict and name not in RESERVED_PAIR_FIELDS
        }
