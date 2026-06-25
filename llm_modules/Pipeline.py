import asyncio
import json
import logging
from dataclasses import dataclass
from typing import List, Set, Dict, Any
import pandas as pd

from .OllamaEngine import LLMEngine, RAW_CSV_COLS, TASK_RUN_ID_COLUMNS
from .OutputParser import OutputParser
from .LLMResultProcessor import LLMResultProcessor
from .Evaluate import PromptCmbEval
from .PromptFormatter import PromptFormatter
from .schemas import (
    DataLoadError,
    TaskBuildError,
    InferenceError,
    PipelineConfig,
    LLMTask,
    ModelName,
    PromptID,
    TaskID,
)


@dataclass
class Prompt:
    promptID: PromptID
    promptText: str


@dataclass
class TaskBatch:
    taskID: TaskID
    itemList: List[Dict[str, Any]]
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
    各階段失敗拋對應子類例外，由 Main_PromptCmb.py 統一捕捉。
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pathConfig = config.paths
        self.promptFormatter = PromptFormatter(
            config.taskTemplate,
            config.itemTemplate,
            config.itemColumns or None,
        )
        logging.info(f"[Pipeline] 初始化完成: outputRoot={self.pathConfig.outputRoot}")


    def run(self):
        """六階段：載入 → 建構任務 → 推論 → 解析 → 後處理 → 評估。"""
        logging.info("[Step 1/6] Loading Task CSV & Prompts")
        taskDf = self.loadTaskData()
        promptList = self.loadPrompts()
        completedTaskRunIDSet = self.loadCompletedTaskRunIDs()

        logging.info("[Step 2/6] Building LLM Tasks")
        taskBatchList = self._buildTaskBatches(taskDf)
        self._validateLabels(taskBatchList)
        self.savePromptPreview(taskBatchList, promptList)
        pendingTaskList = self.buildPendingTasks(taskBatchList, promptList, completedTaskRunIDSet)

        # 兩個條件都成立才 raise：沒待跑任務且沒 checkpoint = 真正空輸入，避免靜默產出空評估。
        # 沒待跑但有 checkpoint = 全部已完成，屬正常，會往下走到評估。
        if not pendingTaskList and not completedTaskRunIDSet:
            raise TaskBuildError("No tasks to run and no checkpoint found.")

        if pendingTaskList:
            logging.info(f"[Step 3/6] Running Inference ({len(pendingTaskList)} tasks)")
            # 只有 Inference 在這裡包 try：把任何例外統一成 InferenceError，讓上層能按子類分流。
            try:
                self.runInference(pendingTaskList)
            except Exception as e:
                raise InferenceError(f"Inference failed: {e}") from e
        else:
            # 全部已完成的常見情境：改了評估指標想重跑後段，不必重新推論。
            logging.info("[Step 3/6] All tasks completed. Skipping inference.")

        # 以下三階段都「讀上一階段寫的檔 → 寫自己的檔 → 回傳路徑」，串成 raw → result → partial → eval。
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
            promptList=[prompt.__dict__ for prompt in promptList],#for fullinfo用
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
        PPI 需有 taskID + labelColumn + contextColumns；
        BC5CDR 需有 taskID + items + contextColumns。
        """
        csvPath = self.pathConfig.taskCsvPath
        if not csvPath.exists():
            raise DataLoadError(f"找不到 Task CSV: {csvPath}")
        
        taskDf = pd.read_csv(csvPath, encoding='utf-8-sig')
        
        if self.config.taskType == "PPI":
            requiredColSet = {'taskID', self.config.labelColumn} | set(self.config.contextColumns)
        else:
            requiredColSet = {'taskID', 'items'} | set(self.config.contextColumns)
            
        #    taskType        | 必要欄位
        #    "PPI"	         |taskID + labelColumn + 所有 contextColumns
        #    "BC5CDR"（else} |taskID + items + 所有 contextColumns
        # 之後有新資料集再擴充對應的必要欄位組合。
        
        missingColSet = requiredColSet - set(taskDf.columns)
        if missingColSet:
            raise DataLoadError(f"Task CSV 缺少必要欄位: {missingColSet}")
        # 用 set 差集找缺漏欄，錯誤訊息直接列出缺哪些，方便對照前處理輸出。
        
        logging.info(f"[Loader] Task CSV 載入完成: {len(taskDf)} 筆 from {csvPath}")
        return taskDf

    def loadPrompts(self) -> List[Prompt]:
        """載入 Prompt 組合 CSV（必要欄位：promptID, promptText），回傳 Prompt list。"""
        csvPath = self.pathConfig.promptCmbPath
        if not csvPath.exists():
            raise DataLoadError(f"找不到 Prompt 組合檔案: {csvPath}")

        promptDf = pd.read_csv(csvPath, encoding='utf-8-sig')
        if 'promptID' not in promptDf.columns or 'promptText' not in promptDf.columns:
            raise DataLoadError("Prompt CSV 缺少 'promptID' 或 'promptText' 欄位。")

        promptList = [Prompt(**row) for row in promptDf[['promptID', 'promptText']].to_dict('records')]
        logging.info(f"[Loader] Prompt CSV 載入完成: {len(promptList)} 筆 from {csvPath}")
        return promptList

    def loadCompletedTaskRunIDs(self) -> Set[TaskRunID]:
        """
        讀取 raw.csv 取得已完成任務的 TaskRunID set，供斷點續傳使用
        schema 不符 / 讀取失敗 → raise DataLoadError
        """
        completedTaskRunIDSet: Set[TaskRunID] = set()
        csvPath = self.pathConfig.rawOutputPath

        if not csvPath.exists():
            return completedTaskRunIDSet

        try:
            rawDf = pd.read_csv(csvPath, encoding='utf-8-sig')
        except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError, UnicodeDecodeError) as e:
            raise DataLoadError(
                f"raw.csv 讀取失敗（壞檔或編碼問題）: {e}。"
                f" 請刪除或備份 {csvPath} 後重跑。"
            ) from e

        # schema 不符則 raise，欄位對不上代表 raw.csv 格式錯誤
        missingColSet = set(RAW_CSV_COLS) - set(rawDf.columns)
        if missingColSet:
            raise DataLoadError(
                f"raw.csv schema 不符，缺欄位: {sorted(missingColSet)}。"
                f" 請刪除或備份 {csvPath} 後重跑。"
            )

        # 只取三個 key 欄、dropna 後組成 set；strip 對齊寫入端的格式，避免空白造成比對失準。
        taskRunIDDf = rawDf[list(TASK_RUN_ID_COLUMNS)].dropna()
        completedTaskRunIDSet.update(
            TaskRunID(str(r.model).strip(), str(r.promptID).strip(), str(r.taskID).strip())
            for r in taskRunIDDf.itertuples(index=False)
        )
        logging.info(f"[Checkpoint] 已完成任務: {len(completedTaskRunIDSet)} 筆")

        return completedTaskRunIDSet

    # ==============================
    # Step 2: Build
    # ==============================

    def savePromptPreview(self, taskBatchList: List[TaskBatch], promptList: List[Prompt]):
        """渲染所有 promptID × task 組合並存成 prompt_preview.csv 以供檢視。"""
        # 把每個 prompt × task 渲染後的 userPrompt 列出來，讓使用者在正式跑之前先確認 prompt 組合邏輯沒問題。
        previewRecordList = []
        for prompt in promptList:
            for taskBatch in taskBatchList:
                previewRecordList.append({
                    'taskID':     taskBatch.taskID,
                    'promptID':   prompt.promptID,
                    'sysPrompt':  prompt.promptText,
                    'userPrompt': taskBatch.userPrompt,
                })

        csvPath = self.pathConfig.promptPreviewPath
        pd.DataFrame(previewRecordList).to_csv(str(csvPath), index=False, encoding='utf-8-sig')
        logging.info(f"[Loader] Prompt preview 已寫入: {len(previewRecordList)} 筆 → {csvPath}")

    def buildPendingTasks(self,
        taskBatchList: List[TaskBatch],
        promptList: List[Prompt],
        completedTaskRunIDSet: Set[TaskRunID], ) -> List[LLMTask]:
        """
        TaskBatch × models × prompts 排列組合，跳過已完成的 TaskRunID，
        回傳尚未執行的 LLMTask 清單。
        """
        # 空模型/空 prompt 都是無意義的執行，提前 raise 比讓下游產出空結果好。
        if not self.config.selectedModels:
            raise TaskBuildError("config.selectedModels 為空，無可執行模型。")
        if not promptList:
            raise TaskBuildError("Prompt 組合清單為空。")

        pendingTaskList: List[LLMTask] = []
        skippedCount = 0

        # 三層迴圈展開 model × prompt × taskBatch 的完整組合，逐一比對 checkpoint 跳過已完成的。
        for modelName in self.config.selectedModels:
            for prompt in promptList:
                for taskBatch in taskBatchList:
                    taskRunID = TaskRunID(modelName, prompt.promptID, taskBatch.taskID)

                    if taskRunID in completedTaskRunIDSet:
                        skippedCount += 1
                        continue

                    pendingTaskList.append(LLMTask(
                        taskID=taskBatch.taskID,
                        model=modelName,
                        promptID=prompt.promptID,
                        sysPrompt=prompt.promptText,
                        userPrompt=taskBatch.userPrompt,
                        items=taskBatch.itemList,
                        context=taskBatch.contextDict,
                    ))

        if skippedCount > 0:
            logging.info(f"[Builder] 跳過已完成任務: {skippedCount} 筆")
        logging.info(f"[Builder] 待執行任務: {len(pendingTaskList)} 筆")
        return pendingTaskList

    def _validateLabels(self, taskBatchList: List[TaskBatch]) -> None:
        """
        在攤平後的 itemList 上做一次性對齊檢查，PPI / BC5CDR 共用同一條路徑。
        任何 item 缺 'label' 欄、或 'label' 對不到 labelSet.classes 即 fail-fast，
        作為 trueLabel 的唯一把關點，避免錯誤組態浪費整輪 inference，下游階段只負責轉碼。
        """
        labelSet = self.config.labelSet

        # 缺 'label' 欄的 item 無 true label 可評估，視為資料錯誤直接擋下
        # 用 taskID 去重回報（缺 label 的 item 根本沒有值可列），方便定位是哪些 task 出問題。
        missingLabelTaskIDSet = {
            batch.taskID
            for batch in taskBatchList
            for item in batch.itemList
            if 'label' not in item
        }
        if missingLabelTaskIDSet:
            samplePreview = sorted(missingLabelTaskIDSet)[:5]
            raise DataLoadError(
                f"Task CSV 內共 {len(missingLabelTaskIDSet)} 個 task 的 item 缺少 'label' 欄，無法評估。"
                f"涉及 taskID（前 5）：{samplePreview}。請檢查前處理輸出。"
            )

        # 第二段：label 值對不到 classes。只列前 5 個，讓使用者判斷是哪些Label。
        unknownSet = {
            item['label']
            for batch in taskBatchList
            for item in batch.itemList
            if labelSet.labelToLabelCode(item['label']) == -1
        }
        if unknownSet:
            samplePreview = sorted(str(v) for v in unknownSet)[:5]
            raise DataLoadError(
                f"Task CSV 內共 {len(unknownSet)} 種 label 不在 labelSet={labelSet.classes} 中："
                f"{samplePreview}。請檢查前處理輸出或 config 的 labelSet 是否一致（比對為去空白、大小寫不敏感）。"
            )

    @staticmethod
    def _parseJsonCell(value: Any, fieldName: str, taskID: str) -> Any:
        """
        解析 Task CSV 的 JSON 欄位字串。
        None / NaN → raise TaskBuildError；其他非字串 → 原樣回傳；字串 → json.loads（失敗往上拋）。
        """
        # 正常路徑：pandas 讀進來的 JSON 欄位是字串，直接交給 json.loads；
        # 解析失敗讓 JSONDecodeError 往上冒，因為壞掉的 items 沒有合理的預設值。
        if isinstance(value, str):
            return json.loads(value)
        # 空欄（None / NaN）代表這筆 task 根本沒有 pair 可跑 → fail-fast，附上 taskID 方便定位。
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise TaskBuildError(f"Task {taskID} 的欄位 '{fieldName}' 為空。")
        # 已是 list/dict（程式式呼叫、非從 CSV 讀）→ 原樣回傳，不重複解析。
        return value

    def _buildTaskBatches(self, taskDf: pd.DataFrame) -> List[TaskBatch]:
        """
        將 Task CSV 每列預處理成 TaskBatch（parse JSON、依 maxItemsPerBatch 切片、format userPrompt）。
        """
        maxItemsPerBatch = self.config.maxItemsPerBatch
        taskType = self.config.taskType
        labelColumn = self.config.labelColumn
        taskBatchList: List[TaskBatch] = []

        for _, row in taskDf.iterrows():
            baseTaskID = str(row['taskID'])
            taskContextDict: Dict[str, Any] = {col: row[col] for col in self.config.contextColumns}
            # 兩種模式攤平成「同一種 itemList 結構」，後續切片/渲染就能共用同一條路徑：
            #  - PPI：把單一 labelColumn 包成單元素 list（型別與 multi 一致）。
            #  - BC5CDR：解析 items JSON 欄成 list。
            # taskType 已於 config 驗證階段限定，PPI 以外即 BC5CDR
            if taskType == "PPI":
                allItemList: List[Dict[str, Any]] = [{'label': row[labelColumn]}]
            else:
                allItemList = self._parseJsonCell(row['items'], 'items', baseTaskID)
            # 之後如果要新增資料集格式可以在這裡，但一定要有taskID與contextcolumn
            # 依 maxItemsPerBatch 切片，每片產生一個 TaskBatch；PPI 因 maxItemsPerBatch=1 永遠只切出 1 片。
            for offset in range(0, len(allItemList), maxItemsPerBatch):
                batchItemList = allItemList[offset:offset + maxItemsPerBatch]
                # 有切片（item 數 > 一批容量）才在 taskID 加 _offset 區分；否則沿用原 taskID。
                batchTaskID = (
                    f"{baseTaskID}_{offset}"
                    if len(allItemList) > maxItemsPerBatch
                    else baseTaskID
                )
                # 在建構期把字串渲染做完，inference 階段只負責拿渲染好的 prompt 丟給 LLM。
                userPrompt = self.promptFormatter.format(taskContextDict, batchItemList)
                taskBatchList.append(
                    TaskBatch(batchTaskID, batchItemList, userPrompt, taskContextDict)
                )

        return taskBatchList


    # ==============================
    # Step 3: Inference
    # ==============================

    def runInference(self, pendingTaskList: List[LLMTask]):
        """建立 LLMEngine 並以 asyncio.run 執行推論；finally 確保 close 釋放連線池。"""
        llmEngine = LLMEngine.fromConfig(self.config, self.pathConfig.rawOutputPath)
        logging.info(f"[Engine] 派送任務: {len(pendingTaskList)} 筆")

        # 內層 coroutine 純粹是為了把 try/finally 包進同一個 async context：
        # 確保 runTasks 即使中途拋例外，close() 仍會在同一個 event loop 內執行、釋放連線池。
        async def runInferenceAndClose():
            try:
                await llmEngine.runTasks(pendingTaskList)
            finally:
                await llmEngine.close()

        # 整條 Pipeline 是同步流程，只在這一步進入非同步；asyncio.run 同步等待它跑完。
        asyncio.run(runInferenceAndClose())
        logging.info(f"[Engine] 推論完成 → {self.pathConfig.rawOutputPath}")
