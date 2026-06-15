import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Optional
from .schemas import ParsingError, Classification, RESERVED_PAIR_FIELDS
from .utils import sanitizeFilename
# 將所有輸出拆解後的命名改為SentID（sentenceID）

class OutputParser:
    """
    將 LLM structured JSON 回應解析為結構化資料表。
    raw.csv → result.csv（predLabel = classes 索引 0..N-1，無法判定一律 -1）。
    JSON 解析失敗或 label 不在 classes 中一律標 -1，由下游評估排除。
    """

    _CSV_KWARGS = {'index': False, 'encoding': 'utf-8-sig'}

    def __init__(self, rawOutputCsvPath: Path, parsedOutputCsvPath: Path, singlePromptCmbOutputDir: Path, labelSet: Classification):
        self.rawOutputCsvPath = Path(rawOutputCsvPath)
        self.parsedOutputCsvPath = Path(parsedOutputCsvPath)
        self.singlePromptCmbOutputDir = Path(singlePromptCmbOutputDir)
        self.labelSet = labelSet

    def run(self) -> Path:
        """主流程：讀 raw.csv → 逐 task 解析 → pair 展開（long format）→ 排序 → 存檔。"""
        # 整段包 try：本層自己拋的 ParsingError 原樣往上拋（保留具體訊息)
        try:
            rawDf = self._loadRawCsv()
            sentRowsList = self._parseTasksToRows(rawDf)
            resultDf = self._buildResultDf(sentRowsList)
            self._writeResultCsvs(resultDf)
            logging.info(f"[Parser] 解析完成: {len(rawDf)} tasks → {len(resultDf)} samples → {self.parsedOutputCsvPath}")
            return self.parsedOutputCsvPath
        except ParsingError:
            raise
        except Exception as e:
            raise ParsingError(f"解析暫存檔時發生錯誤: {e}") from e

    # ── 私有流程方法 ─────────────────────────────────────────────────────────

    def _loadRawCsv(self) -> pd.DataFrame:
        if not self.rawOutputCsvPath.exists():
            raise ParsingError(f"找不到暫存結果檔案: {self.rawOutputCsvPath}")
        return pd.read_csv(str(self.rawOutputCsvPath), encoding='utf-8-sig')

    def _parseTasksToRows(self, rawDf: pd.DataFrame) -> List[dict]:
        """整體任務解析流程：每列 task row → 多列 sentRow（每 pair 一列）。"""
        # raw.csv 一列是「一次推論（一個 batch）」，可能含多個 pair
        # 這裡把它攤平成「一個 pair 一列」的 long format，後續才好 pivot 與評估。
        sentRowsList = []
        for _, taskRow in rawDf.iterrows():
            sentRowsList.extend(self._parseTaskRow(taskRow))
        return sentRowsList

    def _parseTaskRow(self, taskRow: pd.Series) -> List[dict]:
        """一列 task row → 展開成多列 sentRow（每個 pair 一列）。"""
        model    = taskRow.get('model')
        promptID = taskRow.get('promptID')
        taskID   = str(taskRow.get('taskID', ''))

        # pairs 是寫檔時序列化的 JSON；解析回 list。空 list→ 跳過此 task 並 warning，
        # 而非 raise，避免單列異常打斷整批解析。
        pairsList = self._parseJsonCell(taskRow.get('pairs'), default=[])
        if not pairsList:
            logging.warning(f"[Parser] 跳過任務: pairs 為空 (model={model}, promptID={promptID})")
            return []

        rawOutput     = str(taskRow.get('rawOutput', ''))
        predLabels    = self._extractPredLabels(rawOutput, len(pairsList))
        contextDict   = self._parseJsonCell(taskRow.get('context'), default={})

        return [
            self._buildSentRow(model, promptID, taskID, rawOutput,
                           pairDict, predLabels[j], j, len(pairsList), contextDict)
            for j, pairDict in enumerate(pairsList)
        ]

    def _buildSentRow(self, model, promptID, taskID, rawOutput,
                  pairDict: dict, predLabel: int,
                  pairIndex: int, totalPairs: int, contextDict: dict) -> dict:
        """構建 sentRow 字典：包含基本欄位 + pairDict（除保留欄位外）+ contextDict（不覆蓋前者）。"""
        # sentID 來源優先序：pair 自帶的 sentID（如 BC5CDR 原始 ID）> 多 pair 時用 taskID_序號合成 > 單 pair 直接用 taskID。確保每列都有可追溯且唯一的識別碼。
        sentID = pairDict.get('sentID') or (f"{taskID}_{pairIndex}" if totalPairs > 1 else taskID)
        # 先放固定的核心欄（順序穩定，下游好處理）。
        sentRow = {
            "sentID":  sentID,
            "model":     model,
            "promptID":  promptID,
            "trueLabel": pairDict.get('label', ''),
            "predLabel": predLabel,
            "rawOutput": rawOutput,
        }
        # 再疊上 pair 的其他欄位（如 e1/e2），但濾掉 RESERVED_PAIR_FIELDS。
        for otherColName, otherColVal in pairDict.items():
            if otherColName not in RESERVED_PAIR_FIELDS:
                sentRow[otherColName] = otherColVal
        # 最後疊 context 欄，且「不覆蓋」已存在的欄——核心欄與 pair 欄優先，context 只補空缺。
        for otherColName, otherColVal in contextDict.items():
            if otherColName not in sentRow:
                sentRow[otherColName] = otherColVal
        return sentRow

    def _buildResultDf(self, sentRowsList: list) -> pd.DataFrame:
        # 全部 task 的 pairs 都空（整批被跳過）→ 沒東西可評估，raise錯誤
        if not sentRowsList:
            raise ParsingError("解析後沒有產生任何有效資料。")
        resultDf = pd.DataFrame(sentRowsList)
        # 依 (model, promptID, sentID) 排序：讓同一組合的資料連續，方便下游 groupby/pivot，也方便人工對照。
        return resultDf.sort_values(['model', 'promptID', 'sentID'])

    def _writeResultCsvs(self, resultDf: pd.DataFrame) -> None:
        """輸出合併版 result.csv，同時按 promptID 分檔。"""
        # 兩種輸出：合併版給下游 LLMResultProcessor 一次處理；
        # 按 promptID分檔版給人快速檢視單一 prompt 的表現。sanitizeFilename 處理 promptID 當檔名的跨平台安全。
        resultDf.to_csv(str(self.parsedOutputCsvPath), **self._CSV_KWARGS)
        for promptID, groupDf in resultDf.groupby('promptID'):
            singleCsvPath = self.singlePromptCmbOutputDir / f"{sanitizeFilename(promptID)}_result.csv"
            groupDf.to_csv(singleCsvPath, **self._CSV_KWARGS)

    # ── 解析工具方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _parseJsonCell(rawValue, default):
        """
        寬鬆解析 raw.csv 的 JSON 欄位（pairs / context）。
        NaN/None/非法型別 → default；字串 → json.loads（失敗記 warning 後 → default）。
        """
        # raw.csv 是 append-only 累積檔，跨多次執行或人工編輯後容易有壞 row。
        # 任何不可解析的情況都回 default 而非 raise，單筆壞資料不該中斷整批解析。
        if rawValue is None or (isinstance(rawValue, float) and pd.isna(rawValue)):
            return default
        # 已是 dict/list（程式式呼叫情境）→ 原樣回傳，不必再 loads。
        if isinstance(rawValue, (dict, list)):
            return rawValue
        if isinstance(rawValue, str):
            stripped = rawValue.strip()
            if not stripped:
                return default
            try:
                return json.loads(stripped)
            except Exception as e:
                logging.warning(f"[Parser] JSON 欄位解析失敗，回傳 default: {e}")
                return default
        return default

    def _extractPredLabels(self, text: str, batchSize: int) -> List[int]:
        """
        解析 LLM structured JSON 輸出，回傳長度為 batchSize 的預測碼 list（classes 索引 0..N-1 / -1）。
        "Error:"／空／非合法 JSON 物件 → 全部 -1。
        single：{"label": ...} → [labelCode]；
        batch：{"answers": [{"id", "label"}]} → 依 id（1-based）或順序回填。
        結構不符 → 對應位置維持 -1（ downstream 評估排除）。
        """
        # 先全填 -1：任何「無法判定」的位置就停在 -1，後面只覆寫「能判定」的，省去缺漏處理。
        predLabels = [-1] * batchSize
        # 空字串或含 "Error:"（來自 _safeGenerate 的失敗 marker）→ 整批維持 -1。
        if not text or "Error:" in text:
            return predLabels

        obj = self._loadJsonObject(text)
        if obj is None:
            logging.warning("[Parser] 輸出非合法 JSON 物件，全標 -1")
            return predLabels
    
        answers = obj.get("answers")
        if isinstance(answers, list):
            seenIdxSet = set()
            for answerOrderIdx, ans in enumerate(answers):
                if not isinstance(ans, dict):
                    continue
                # id 優先（1-based 編號，能對抗亂序）、出現順序後備；都超界回 None 直接丟棄。
                idx = self._resolveAnswerIndex(ans.get("id"), answerOrderIdx, batchSize)
                if idx is None:
                    continue
                if idx in seenIdxSet:
                    logging.warning(f"[Parser] answers 出現重複位置 idx={idx}（id={ans.get('id')}），後者覆蓋前者")
                seenIdxSet.add(idx)
                predLabels[idx] = self.labelSet.labelToLabelCode(ans.get("label"))
            return predLabels

        if "label" in obj:
            predLabels[0] = self.labelSet.labelToLabelCode(obj.get("label"))
        return predLabels

    @staticmethod
    def _loadJsonObject(text: str):
        """structured 模式下 rawOutput 必為合法 JSON 物件；解析失敗或非物件型一律回 None。"""
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    @staticmethod
    def _resolveAnswerIndex(idValue, answerOrderIdx: int, batchSize: int) -> Optional[int]:
        """id 為合法 1-based 編號 → id-1；否則退回出現順序 answerOrderIdx；皆超出範圍回 None。"""
        # id 優先：LLM 亂序回答（先 id=3 再 id=1）時，靠 id 才能正確配對 pair。
        try:
            idNum = int(str(idValue).strip())
            if 1 <= idNum <= batchSize:
                return idNum - 1
        except (TypeError, ValueError):
            pass
        # id 不合法/越界 → 退回出現順序 answerOrderIdx（少數模型漏 id 或給 0-based 時的兜底）；也超出去則丟棄。
        return answerOrderIdx if answerOrderIdx < batchSize else None
