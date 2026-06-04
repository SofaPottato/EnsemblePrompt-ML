import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Optional
from .schemas import ParsingError, Classification, RESERVED_PAIR_FIELDS
from .utils import sanitizeFilename


class OutputParser:
    """
    將 LLM structured JSON 回應解析為結構化資料表。
    raw.csv → result.csv（predLabel = classes 索引 0..N-1，無法判定一律 -1）。
    JSON 解析失敗或 label 不在 classes 中一律標 -1，由下游評估排除。
    """

    _CSV_KWARGS = {'index': False, 'encoding': 'utf-8-sig'}

    def __init__(self, rawOutputCsvPath: Path, parsedOutputCsvPath: Path,
                 singlePromptCmbOutputDir: Path,
                 labelSet: Optional[Classification] = None):
        self.rawOutputCsvPath = Path(rawOutputCsvPath)
        self.parsedOutputCsvPath = Path(parsedOutputCsvPath)
        self.singlePromptCmbOutputDir = Path(singlePromptCmbOutputDir)
        self.labelSet = labelSet or Classification()

    def run(self) -> Path:
        """主流程：讀 raw.csv → 逐 task 解析 → pair 展開（long format）→ 排序 → 存檔。"""
        try:
            rawDf = self._loadRawCsv()
            sampleRowsList = self._parseTasksToRows(rawDf)
            resultDf = self._buildResultDf(sampleRowsList)
            self._writeResultCsvs(resultDf)
            logging.info(f"[Parser] 解析完成: {len(resultDf)} 筆 → {self.parsedOutputCsvPath}")
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

    def _parseTasksToRows(self, rawDf: pd.DataFrame) -> list:
        b_hasContextCol = 'context' in rawDf.columns
        sampleRowsList = []
        for _, taskRow in rawDf.iterrows():
            sampleRowsList.extend(self._parseTaskRow(taskRow, b_hasContextCol))
        return sampleRowsList

    def _parseTaskRow(self, taskRow, b_hasContextCol: bool) -> list:
        """一列 task row → 展開成多列 sampleRow（每個 pair 一列）。"""
        model    = taskRow.get('model')
        promptID = taskRow.get('promptID')
        taskID   = str(taskRow.get('taskID', ''))

        pairsList = self._parseJsonCell(taskRow.get('pairs'), default=[])
        if not pairsList:
            logging.warning(f"[Parser] 跳過任務: pairs 為空 (model={model}, promptID={promptID})")
            return []

        rawOutput   = str(taskRow.get('rawOutput', ''))
        predCodes   = self._extractPredCodes(rawOutput, len(pairsList))
        contextDict = (self._parseJsonCell(taskRow.get('context'), default={})
                       if b_hasContextCol else {})

        return [
            self._buildSampleRow(model, promptID, taskID, rawOutput,
                           pairDict, predCodes[j], j, len(pairsList), contextDict)
            for j, pairDict in enumerate(pairsList)
        ]

    def _buildSampleRow(self, model, promptID, taskID, rawOutput,
                  pairDict: dict, predLabel: int,
                  pairIndex: int, totalPairs: int, contextDict: dict) -> dict:
        """單一 pair + 對應預測標籤 → sampleRow。"""
        sampleID = pairDict.get('sampleID') or (f"{taskID}_{pairIndex}" if totalPairs > 1 else taskID)
        sampleRow = {
            "sampleID":  sampleID,
            "model":     model,
            "promptID":  promptID,
            "trueLabel": pairDict.get('label', ''),
            "predLabel": predLabel,
            "rawOutput": rawOutput,
        }
        for fieldName, fieldVal in pairDict.items():
            if fieldName not in RESERVED_PAIR_FIELDS:
                sampleRow[fieldName] = fieldVal
        for fieldName, fieldVal in contextDict.items():
            if fieldName not in sampleRow:
                sampleRow[fieldName] = fieldVal
        return sampleRow

    def _buildResultDf(self, sampleRowsList: list) -> pd.DataFrame:
        if not sampleRowsList:
            raise ParsingError("解析後沒有產生任何有效資料。")
        resultDf = pd.DataFrame(sampleRowsList)
        return resultDf.sort_values(['model', 'promptID', 'sampleID'])

    def _writeResultCsvs(self, resultDf: pd.DataFrame) -> None:
        """輸出合併版 result.csv，同時按 promptID 分檔。"""
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
        if rawValue is None or (isinstance(rawValue, float) and pd.isna(rawValue)):
            return default
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

    def _extractPredCodes(self, text: str, batchSize: int) -> List[int]:
        """
        解析 LLM JSON 輸出，回傳長度為 batchSize 的 code list（classes 索引 / -1）。
        "Error:" 或空 → 全部 -1。
        """
        predCodes = [-1] * batchSize
        if not text or "Error:" in text:
            return predCodes
        return self._parseJsonToCodes(text, batchSize)

    def _parseJsonToCodes(self, text: str, batchSize: int) -> List[int]:
        """
        解析 structured JSON 輸出。
        single：{"label": ...} → [code]；batch：{"answers": [{"id", "label"}]} → 依 id（1-based）或順序回填。
        JSON 解析失敗或結構不符 → 對應位置維持 -1（下游評估排除）。
        """
        predCodes = [-1] * batchSize
        obj = self._loadJsonObject(text)
        if obj is None:
            logging.warning("[Parser] 輸出非合法 JSON 物件，全標 -1")
            return predCodes

        answers = obj.get("answers")
        if isinstance(answers, list):
            for pos, ans in enumerate(answers):
                if not isinstance(ans, dict):
                    continue
                idx = self._resolveAnswerIndex(ans.get("id"), pos, batchSize)
                if idx is not None:
                    predCodes[idx] = self.labelSet.labelToCode(ans.get("label"))
            return predCodes

        # single-target schema：{"label": <enum>}
        if "label" in obj:
            predCodes[0] = self.labelSet.labelToCode(obj.get("label"))
        return predCodes

    @staticmethod
    def _loadJsonObject(text: str):
        """structured 模式下 rawOutput 必為合法 JSON 物件；解析失敗或非物件型一律回 None。"""
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    @staticmethod
    def _resolveAnswerIndex(idValue, pos: int, batchSize: int) -> Optional[int]:
        """id 為合法 1-based 編號 → id-1；否則退回出現順序 pos；皆超出範圍回 None。"""
        try:
            idNum = int(str(idValue).strip())
            if 1 <= idNum <= batchSize:
                return idNum - 1
        except (TypeError, ValueError):
            pass
        return pos if pos < batchSize else None
