import asyncio
import logging
from dataclasses import dataclass
from typing import List, Set, Iterator, Tuple, Dict, Any
import pandas as pd

from .OllamaEngine import LLMEngine, RAW_CSV_SCHEMA, TASK_RUN_ID_COLUMNS
from .OutputParser import OutputParser
from .LLMResultProcessor import LLMResultProcessor
from .Evaluate import PromptCmbEval
from .PromptFormatter import PromptFormatter
from .utils import parsePairListField
from .schemas import (
    DataLoadError,
    TaskBuildError,
    InferenceError,
    LLMAppConfig,
    LLMTask,
    ModelName,
    PromptID,
    TaskID,
)


@dataclass
class PromptCmb:
    promptID: PromptID
    promptText: str


@dataclass
class TaskBatch:
    taskID: TaskID
    pairList: List[Dict[str, Any]]
    userPrompt: str
    contextDict: Dict[str, Any]


@dataclass(frozen=True)
class TaskRunID:
    """單次 LLM 推論的唯一識別三元組；用於 raw.csv checkpoint 比對。"""
    model:    ModelName
    promptID: PromptID
    taskID:   TaskID


# ==============================
# Pipeline
# ==============================

class ExperimentPipeline:
    """
    實驗流程統籌：載入 → 建構任務 → 推論 → 解析 → 後處理 → 評估。
    各階段失敗拋對應子類例外，由 call_LLM.py 統一捕捉。
    """

    def __init__(self, config: LLMAppConfig):
        self.config = config
        self.pathConfig = config.paths
        self.promptFormatter = PromptFormatter(
            config.taskTemplate,
            config.pairTemplate,
            config.pairColumns or None,
        )
        logging.info(f"[Pipeline] 初始化完成: outputRoot={self.pathConfig.outputRoot}")


    def run(self):
        """六階段：載入 → 建構任務 → 推論 → 解析 → 後處理 → 評估。"""
        logging.info("[Step 1/6] Loading Task CSV & Prompts")
        taskDataDf = self.loadTaskData()
        promptCmbList = self.loadPromptCmbs()

        logging.info("[Step 2/6] Building LLM Tasks")
        taskBatchList = self._buildTaskBatches(taskDataDf)
        self.savePromptPreview(taskBatchList, promptCmbList)
        completedTaskRunIDSet = self.loadCompletedTaskRunIDs()
        pendingTaskList = self.buildPendingTasks(taskBatchList, promptCmbList, completedTaskRunIDSet)

        if not pendingTaskList and not completedTaskRunIDSet: #避免實驗意外終止沒有跑到後面的評估階段
            raise TaskBuildError("No tasks to run and no checkpoint found.")

        if pendingTaskList:
            logging.info(f"[Step 3/6] Running Inference ({len(pendingTaskList)} tasks)")
            try:
                self.runInference(pendingTaskList)
            except Exception as e:
                raise InferenceError(f"Inference failed: {e}") from e
        else:
            logging.info("[Step 3/6] All tasks completed. Skipping inference.")

        logging.info("[Step 4/6] Parsing LLM Outputs")
        parsedOutputPath = OutputParser(
            rawOutputCsvPath=self.pathConfig.rawOutputPath,
            parsedOutputCsvPath=self.pathConfig.resultPath,
            singlePromptCmbOutputDir=self.pathConfig.singlePromptCmbOutputDir,
            labelSet=self.config.labelSet,
        ).run()

        logging.info("[Step 5/6] Processing Results")
        partialInfoPath = LLMResultProcessor(
            parsedOutputCsvPath=parsedOutputPath,
            partialInfoCsvPath=self.pathConfig.partialInfoPath,
            fullInfoCsvPath=self.pathConfig.fullInfoPath,
            promptCmbList=[promptCmb.__dict__ for promptCmb in promptCmbList],
            labelSet=self.config.labelSet,
        ).run()

        logging.info("[Step 6/6] Evaluating")
        PromptCmbEval(
            partialInfoCsvPath=partialInfoPath,
            outputDirPath=self.pathConfig.evalDir,
            labelSet=self.config.labelSet,
        ).run()

        logging.info("[Pipeline] 流程結束")

    # ==============================
    # Step 1: Load
    # ==============================

    def loadTaskData(self) -> pd.DataFrame:
        """
        載入 Task CSV 並驗證必要欄位。
        single-target 需有 taskID + labelColumn + contextColumns；
        multi-target 需有 taskID + pairs + contextColumns。
        """
        csvPath = self.pathConfig.taskCsvPath
        if not csvPath.exists():
            raise DataLoadError(f"找不到 Task CSV: {csvPath}")

        taskDataDf = pd.read_csv(csvPath, encoding='utf-8-sig')
        if self.config.isSingleTarget:
            requiredColSet = {'taskID', self.config.labelColumn} | set(self.config.contextColumns)
        else:
            requiredColSet = {'taskID', 'pairs'} | set(self.config.contextColumns)

        missingColSet = requiredColSet - set(taskDataDf.columns)
        if missingColSet:
            raise DataLoadError(f"Task CSV 缺少必要欄位: {missingColSet}")

        logging.info(f"[Loader] Task CSV 載入完成: {len(taskDataDf)} 筆 from {csvPath}")
        return taskDataDf

    def loadPromptCmbs(self) -> List[PromptCmb]:
        """載入 Prompt 組合 CSV（必要欄位：promptID, promptText），回傳 PromptCmb list。"""
        csvPath = self.pathConfig.promptCmbPath
        if not csvPath.exists():
            raise DataLoadError(f"找不到 Prompt 組合檔案: {csvPath}")

        promptDatadf = pd.read_csv(csvPath, encoding='utf-8-sig')
        if 'promptID' not in promptDatadf.columns or 'promptText' not in promptDatadf.columns:
            raise DataLoadError("Prompt CSV 缺少 'promptID' 或 'promptText' 欄位。")

        return [PromptCmb(**row) for row in promptDatadf[['promptID', 'promptText']].to_dict('records')]

    def loadCompletedTaskRunIDs(self) -> Set[TaskRunID]:
        """
        讀取 raw.csv 取得已完成任務的 TaskRunID set，供斷點續傳使用。
        schema 不符 → raise DataLoadError；讀取失敗 → warning + 回傳空 set（從頭跑）。
        """
        completedTaskRunIDSet: Set[TaskRunID] = set()
        csvPath = self.pathConfig.rawOutputPath
        if not csvPath.exists():
            return completedTaskRunIDSet

        try:
            rawOutputDataDf = pd.read_csv(csvPath, encoding='utf-8-sig')
        except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError, UnicodeDecodeError) as e:
            logging.warning(f"[Checkpoint] 讀取失敗，從頭開始: {e}")
            return completedTaskRunIDSet

        missingColSet = set(RAW_CSV_SCHEMA) - set(rawOutputDataDf.columns)
        if missingColSet:
            raise DataLoadError(
                f"raw.csv schema 不符，缺欄位: {sorted(missingColSet)}。"
                f" 請刪除或備份 {csvPath} 後重跑。"
            )

        taskRunIDDf = rawOutputDataDf[list(TASK_RUN_ID_COLUMNS)].dropna()
        completedTaskRunIDSet.update(
            TaskRunID(str(r.model).strip(), str(r.promptID).strip(), str(r.taskID).strip())
            for r in taskRunIDDf.itertuples(index=False)
        )
        logging.info(f"[Checkpoint] 已完成任務: {len(completedTaskRunIDSet)} 筆")

        return completedTaskRunIDSet

    # ==============================
    # Step 2: Build
    # ==============================

    def savePromptPreview(self, taskBatchList: List[TaskBatch], promptCmbList: List[PromptCmb]):
        """渲染所有 promptID × task 組合並存成 prompt_preview.csv 以供檢視。"""
        previewRecordList = []
        for promptCmb in promptCmbList:
            for taskBatch in taskBatchList:
                previewRecordList.append({
                    'taskID':     taskBatch.taskID,
                    'promptID':   promptCmb.promptID,
                    'sysPrompt':  promptCmb.promptText,
                    'userPrompt': taskBatch.userPrompt,
                })

        csvPath = self.pathConfig.promptPreviewPath
        pd.DataFrame(previewRecordList).to_csv(str(csvPath), index=False, encoding='utf-8-sig')
        logging.info(f"[Loader] Prompt preview 已寫入: {len(previewRecordList)} 筆 → {csvPath}")

    def buildPendingTasks(self,
        taskBatchList: List[TaskBatch],
        promptCmbList: List[PromptCmb],
        completedTaskRunIDSet: Set[TaskRunID], ) -> List[LLMTask]:
        """
        TaskBatch × models × prompts 排列組合，跳過已完成的 TaskRunID，
        回傳尚未執行的 LLMTask 清單。
        """
        if not self.config.selectedModels:
            raise TaskBuildError("config.selectedModels 為空，無可執行模型。")
        if not promptCmbList:
            raise TaskBuildError("Prompt 組合清單為空。")

        pendingTaskList: List[LLMTask] = []
        skippedCount = 0

        for modelName in self.config.selectedModels:
            for promptCmb in promptCmbList:
                for taskBatch in taskBatchList:
                    taskRunID = TaskRunID(modelName, promptCmb.promptID, taskBatch.taskID)

                    if taskRunID in completedTaskRunIDSet:
                        skippedCount += 1
                        continue

                    pendingTaskList.append(LLMTask(
                        taskID=taskBatch.taskID,
                        model=modelName,
                        promptID=promptCmb.promptID,
                        sysPrompt=promptCmb.promptText,
                        userPrompt=taskBatch.userPrompt,
                        pairs=taskBatch.pairList,
                        context=taskBatch.contextDict,
                    ))

        if skippedCount > 0:
            logging.info(f"[Builder] 跳過已完成任務: {skippedCount} 筆")
        logging.info(f"[Builder] 待執行任務: {len(pendingTaskList)} 筆")
        return pendingTaskList

    def _buildTaskBatches(self, taskDataDf: pd.DataFrame) -> List[TaskBatch]:
        """
        將 Task CSV 每列預處理成 TaskBatch（parse JSON、依 maxPairsPerBatch 切片、format userPrompt）。
        """
        maxPairsPerBatch = self.config.maxPairsPerBatch
        isSingleTarget = self.config.isSingleTarget
        labelColumn = self.config.labelColumn
        taskBatchList: List[TaskBatch] = []

        for _, row in taskDataDf.iterrows():
            baseTaskID = str(row['taskID'])
            taskContextDict: Dict[str, Any] = {col: row[col] for col in self.config.contextColumns}

            if isSingleTarget:
                allLabelPairList: List[Dict[str, Any]] = [{'label': row[labelColumn]}]
            else:
                allLabelPairList = parsePairListField(row['pairs'], 'pairs', baseTaskID)

            for offset in range(0, len(allLabelPairList), maxPairsPerBatch):
                batchLabelPairList = allLabelPairList[offset:offset + maxPairsPerBatch]

                batchTaskID = (
                    f"{baseTaskID}_{offset}"
                    if len(allLabelPairList) > maxPairsPerBatch
                    else baseTaskID
                )

                userPrompt = self.promptFormatter.format(taskContextDict, batchLabelPairList)

                taskBatchList.append(
                    TaskBatch(batchTaskID, batchLabelPairList, userPrompt, taskContextDict)
                )

        return taskBatchList


    # ==============================
    # Step 3: Inference
    # ==============================

    def runInference(self, pendingTaskList: List[LLMTask]):
        """建立 LLMEngine 並以 asyncio.run 執行推論；finally 確保 close 釋放連線池。"""
        llmEngine = LLMEngine.fromConfig(self.config, self.pathConfig.rawOutputPath)
        logging.info(f"[Engine] 派送任務: {len(pendingTaskList)} 筆")

        async def runInferenceAndClose():
            try:
                await llmEngine.runTasks(pendingTaskList)
            finally:
                await llmEngine.close()

        asyncio.run(runInferenceAndClose())
        logging.info(f"[Engine] 推論完成 → {self.pathConfig.rawOutputPath}")

