import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict
from .schemas import PipelineError, Classification


class LLMResultProcessor:
    """
    將 OutputParser 的長表格清理後轉置為寬表格，供下游 Evaluate 與人工檢視使用。
    result.csv (long) → partialInfo.csv (wide) + fullInfo.csv（含 rawOutput / sysPrompt）。
    """

    _PRED_SUFFIX = '__pred'
    _RAW_SUFFIX = '__raw'
    _SYS_PROMPT_SUFFIX = '__sysPrompt'
    _RUN_KEY_SEPARATOR = '|'   # model 與 promptID 之間的分隔字元；'_' 會與模型名沖突，'|' 更醒目
    _CSV_KWARGS = {'index': False, 'encoding': 'utf-8-sig'}
    _REQUIRED_COLS = ('itemID', 'model', 'promptID', 'predLabel', 'trueLabel')
    _NON_INDEX_COLS = {'model', 'promptID', 'runKey', 'predLabel', 'rawOutput'}

    def __init__(self, parsedOutputCsvPath: Path, partialInfoCsvPath: Path, fullInfoCsvPath: Path = None,
                 promptCmbList: List[Dict] = None, labelSet: Classification = None):
        self.parsedOutputCsvPath = Path(parsedOutputCsvPath)
        self.partialInfoCsvPath = Path(partialInfoCsvPath)
        self.fullInfoCsvPath = Path(fullInfoCsvPath) if fullInfoCsvPath else None
        self.promptCmbList = promptCmbList or []
        self.labelSet = labelSet or Classification()

        self.inputDf = None
        self.partialDf = None
        self.fullDf = None

    def run(self) -> Path:
        """主流程協調者：讀取 → 標準化 → Pivot → 存檔，回傳精簡版寬表格路徑。"""
        logging.info(f"[Processor] 啟動: {self.parsedOutputCsvPath}")
        self._loadData()
        self._prepareDf()
        self._pivotData()
        return self._saveData()

    # ── 私有流程方法 ─────────────────────────────────────────────────────────

    def _loadData(self):
        """讀取並驗證輸入 CSV，缺必要欄位即拋 PipelineError。"""
        if not self.parsedOutputCsvPath.exists():
            raise PipelineError(f"File not found: {self.parsedOutputCsvPath}")
        try:
            self.inputDf = pd.read_csv(self.parsedOutputCsvPath, encoding='utf-8-sig')
        except Exception as e:
            raise PipelineError(f"Failed to read CSV: {e}") from e

        missingList = [c for c in self._REQUIRED_COLS if c not in self.inputDf.columns]
        if missingList:
            raise PipelineError(f"Missing required columns: {missingList}")

    def _prepareDf(self):
        """備份 originalLabel、標準化 trueLabel、組 runKey。"""
        self.inputDf['originalLabel'] = self.inputDf['trueLabel']
        self.inputDf['trueLabel'] = self.inputDf['trueLabel'].apply(self._convertTrueLabel)

        unknownCount = (self.inputDf['trueLabel'] == -1).sum()
        if unknownCount > 0:
            logging.warning(f"[Processor] 共 {unknownCount} 筆 trueLabel 未識別")

        self.inputDf['runKey'] = (self.inputDf['model'].astype(str) + self._RUN_KEY_SEPARATOR +
                                  self.inputDf['promptID'].astype(str))

    def _convertTrueLabel(self, labelValue) -> int:
        """trueLabel 字串 → classes 索引 code；未命中 -1（表示前處理 label 未對齊 classes）。"""
        code = self.labelSet.labelToCode(labelValue)
        if code == -1:
            logging.warning(f"[Processor] trueLabel '{labelValue}' 不在 classes 中 → -1（請檢查前處理對齊）")
        return code

    def _pivotData(self):
        """
        long → wide pivot：以樣本為列、runKey 為欄、predLabel 為值。
        index 欄動態偵測（排除 _NON_INDEX_COLS），讓上游新增資料欄不需要改本模組。
        同時產出 partialDf（predLabel）與 fullDf（predLabel + __raw 後綴欄）。
        所有 runKey 衍生欄位皆帶後綴：__pred / __raw / __sysPrompt，便於後續過濾。
        """
        indexCols = [c for c in self.inputDf.columns if c not in self._NON_INDEX_COLS]

        try:
            predOnlyDf = self.inputDf.pivot_table(
                index=indexCols,
                columns='runKey',
                values='predLabel',
                aggfunc='first'
            ).fillna(-1)
            predOnlyDf.columns = [f"{c}{self._PRED_SUFFIX}" for c in predOnlyDf.columns]
            self.partialDf = predOnlyDf.reset_index()

            rawOnlyDf = self.inputDf.pivot_table(
                index=indexCols,
                columns='runKey',
                values='rawOutput',
                aggfunc='first'
            ).fillna('')
            rawOnlyDf.columns = [f"{c}{self._RAW_SUFFIX}" for c in rawOnlyDf.columns]

            self.fullDf = pd.concat([predOnlyDf, rawOnlyDf], axis=1).reset_index()
        except Exception as e:
            raise PipelineError(f"Pivot failed: {e}") from e

    def _saveData(self) -> Path:
        """寫精簡版（必出）與完整版（若提供路徑），記錄統計後回傳精簡版路徑。"""
        try:
            self._writePartialCsv()
            if self.fullInfoCsvPath and self.fullDf is not None:
                self._writeFullCsv()
            self._logSummary()
            return self.partialInfoCsvPath
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(f"Failed to save results: {e}") from e

    def _writePartialCsv(self) -> None:
        """精簡版：itemID + trueLabel + 各 runKey 的 __pred 欄。"""
        predCols = [c for c in self.partialDf.columns if c.endswith(self._PRED_SUFFIX)]
        leanCols = [c for c in ('itemID', 'trueLabel') if c in self.partialDf.columns] + predCols
        self.partialDf[leanCols].to_csv(self.partialInfoCsvPath, **self._CSV_KWARGS)

    def _writeFullCsv(self) -> None:
        """完整版：補 {runKey}__sysPrompt 欄 → 欄位重排 → 寫檔。"""
        # 補 {runKey}__sysPrompt 欄（promptCmbList 為空則跳過）
        sysPromptCols: List[str] = []
        if self.promptCmbList:
            promptIDToText = {p['promptID']: p['promptText'] for p in self.promptCmbList}
            runKeyIter = (self.inputDf[['model', 'promptID', 'runKey']]
                          .drop_duplicates()
                          .itertuples(index=False))
            for row in runKeyIter:
                colName = f"{row.runKey}{self._SYS_PROMPT_SUFFIX}"
                self.fullDf[colName] = promptIDToText.get(row.promptID, '')
                if colName not in sysPromptCols:
                    sysPromptCols.append(colName)

        # 欄位重排：itemID → labels → __raw → __pred → __sysPrompt → 其他 index 欄
        predCols = [c for c in self.fullDf.columns if c.endswith(self._PRED_SUFFIX)]
        rawCols = [c for c in self.fullDf.columns if c.endswith(self._RAW_SUFFIX)]
        indexCols = [c for c in self.fullDf.columns
                     if c not in predCols and c not in rawCols and c not in sysPromptCols]
        labelCols = [c for c in ('originalLabel', 'trueLabel') if c in indexCols]
        idCols = [c for c in ('itemID',) if c in indexCols]
        otherIndexCols = [c for c in indexCols if c not in labelCols and c not in idCols]
        orderedCols = idCols + labelCols + rawCols + predCols + sysPromptCols + otherIndexCols
        self.fullDf[orderedCols].to_csv(self.fullInfoCsvPath, **self._CSV_KWARGS)

    def _logSummary(self) -> None:
        """輸出 parse rate 與 shape 資訊；fullDf 不存在時 shape 標 N/A。"""
        validCount = (self.inputDf['predLabel'] != -1).sum()
        totalCount = len(self.inputDf)
        fullShapeStr = str(self.fullDf.shape) if self.fullDf is not None else "N/A"
        logging.info(
            f"[Processor] 完成: partial={self.partialDf.shape}, full={fullShapeStr}, "
            f"parse rate={validCount}/{totalCount} ({validCount/totalCount:.1%}) "
            f"→ {self.partialInfoCsvPath}"
        )
