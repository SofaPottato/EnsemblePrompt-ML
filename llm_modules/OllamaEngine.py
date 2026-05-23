import asyncio
import json
import logging
import time
import csv
import os
import httpx
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Union
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from .schemas import LLMAppConfig, LLMTask, ModelName, RawOutput

# raw.csv 欄位順序的單一事實來源；下游讀檔時應據此驗證
RAW_CSV_SCHEMA: List[str] = [
    "model", "promptID", "taskID",
    "systemPrompt", "userPrompt", "rawOutput", "pairs", "context",
]

# raw.csv 用來判斷任務是否已完成的 composite key 欄位（對應 TaskRunID 三元組）
TASK_RUN_ID_COLUMNS: Tuple[str, str, str] = ("model", "promptID", "taskID")

class OllamaClient:
    """非同步 Ollama API 客戶端，封裝連線池與 tenacity 重試（最多 3 次，指數退避）。"""
    def __init__(self, apiUrl: str, timeout: float, llmOptions: Dict[str, Any],
                 responseFormat: Dict[str, Any]):
        self.apiUrl = apiUrl
        self.timeout = timeout
        self.llmOptions = llmOptions
        self.responseFormat = responseFormat
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=100)
        self.httpClient = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(self.timeout, connect=30.0)
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError, httpx.ReadTimeout)),
        reraise=True
    )
    async def generate(self, modelName: ModelName, sysPrompt: str, userPrompt: str) -> RawOutput:
        """送出單次推論請求，回傳模型文字回應。網路錯誤自動重試，超限後往上拋。"""
        payload = {
            "model": modelName,
            "messages": [
                {"role": "system", "content": sysPrompt},
                {"role": "user", "content": userPrompt}
            ],
            "stream": False,
            "options": self.llmOptions,
            "format": self.responseFormat,
        }

        try:
            response = await self.httpClient.post(self.apiUrl, json=payload)
            response.raise_for_status()
            return response.json().get('message', {}).get('content', '')
        except Exception as e:
            logging.warning(f"[Client] {modelName} 連線異常: {e}")
            raise

    async def close(self):
        """關閉 HTTP 客戶端，釋放底層 TCP 連線池。"""
        await self.httpClient.aclose()


class LLMEngine:
    """
    支援多模型動態路由的非同步推論引擎。
    雙層併發控制：modelSemaphoreDict（每模型）+ modelConcurrencySemaphore（跨模型）。
    每筆完成即 append 寫入 raw.csv 並 fsync 落盤，中斷可從 checkpoint 續跑。
    已完成任務的過濾由呼叫端（Pipeline.buildPendingTasks）負責，引擎只執行收到的清單。
    結果不透過回傳值傳遞，下游階段直接從 raw.csv 讀取。
    """

    # ── 建構與生命週期 ──────────────────────────────────────────────────────
    def __init__(self,
                 apiUrl: str,
                 timeout: float,
                 llmOptions: Dict[str, Any],
                 concurrencyPerModel: int,
                 maxConcurrentModels: int,
                 outputFile: Union[str, Path],
                 responseFormat: Dict[str, Any]):
        self.concurrencyPerModel = concurrencyPerModel
        self.maxConcurrentModels = maxConcurrentModels
        self.outputFile = str(outputFile)
        self.ollamaClient = OllamaClient(
            apiUrl=apiUrl, timeout=timeout, llmOptions=llmOptions,
            responseFormat=responseFormat,
        )
        # defaultdict 讓每個模型第一次存取時自動建立專屬 Semaphore
        self.modelSemaphoreDict = defaultdict(lambda: asyncio.Semaphore(self.concurrencyPerModel))
        self.modelConcurrencySemaphore = asyncio.Semaphore(self.maxConcurrentModels)
        self.initializedModelSet = set()
        self.fileLock = asyncio.Lock()

    @classmethod
    def fromConfig(cls, config: LLMAppConfig, outputFile: Union[str, Path]) -> "LLMEngine":
        """由 LLMAppConfig 建立 engine，集中映射 config 欄位到引擎建構參數，避免 Pipeline 端硬寫。"""
        return cls(
            apiUrl=config.ollamaServer.url,
            timeout=config.ollamaServer.timeout,
            llmOptions=config.llmOptions,
            concurrencyPerModel=config.concurrencyPerModel,
            maxConcurrentModels=config.maxConcurrentModels,
            outputFile=str(outputFile),
            responseFormat=config.buildResponseSchema(),
        )

    # ── Step 1: 全批次調度（公開入口）────────────────────────────────────
    async def runTasks(self, taskList: List[LLMTask]) -> None:
        """
        所有任務的調度入口。依 model 分組後以 gather 同時啟動各組，
        組內用 as_completed 交錯執行並即時更新 tqdm 進度條。
        結果直接 append 至 raw.csv，本方法不回傳；下游讀檔取得。
        """
        if not taskList:
            logging.warning("[Engine] 任務清單為空")
            return

        # 依 model 分桶，為每個模型起一條獨立 group coroutine
        tasksByModelDict: Dict[ModelName, List[LLMTask]] = defaultdict(list)
        for task in taskList:
            tasksByModelDict[task.model].append(task)

        logging.info(
            f"[Engine] 推論啟動: {len(taskList)} 任務、{len(tasksByModelDict)} 模型 "
            f"({', '.join(tasksByModelDict)}) "
            f"(maxConcurrentModels={self.maxConcurrentModels}, "
            f"concurrencyPerModel={self.concurrencyPerModel})"
        )

        # tqdm 用 context manager 包住，gather 拋例外時也能保證 bar 正確關閉
        with tqdm(total=len(taskList), desc="總推論進度", unit="batch") as progressBar, \
             logging_redirect_tqdm():
            await asyncio.gather(*[
                self._processModelGroup(modelName, modelTaskList, progressBar)
                for modelName, modelTaskList in tasksByModelDict.items()
            ])

    async def _processModelGroup(self, modelName: ModelName,
                                 modelTaskList: List[LLMTask],
                                 progressBar: tqdm) -> None:
        """處理單一模型的所有任務；外層 modelConcurrencySemaphore 控制同時載入幾個模型。"""
        async with self.modelConcurrencySemaphore:
            logging.info(f"[Engine] 模型 {modelName} 取得許可: {len(modelTaskList)} 筆任務")

            taskCoroutineList = [self._processSingleTask(task) for task in modelTaskList]
            for completedCoroutine in asyncio.as_completed(taskCoroutineList):
                await completedCoroutine
                progressBar.update(1)

            logging.info(f"[Engine] 模型 {modelName} 完成，釋放許可")

    # ── Step 2: 單筆任務處理 ─────────────────────────────────────────────
    async def _processSingleTask(self, task: LLMTask) -> None:
        """
        單一任務的完整生命週期：Semaphore 排隊 → API 呼叫 → append 寫 CSV。
        API 失敗不 raise，改寫 "Error:..." 字串，下游 OutputParser 看到後標 -1 跳過。
        """
        async with self.modelSemaphoreDict[task.model]:
            # 每個 model 第一次取得 semaphore 時印一次啟動訊息
            if task.model not in self.initializedModelSet:
                logging.info(f"[Engine] 模型 {task.model} 啟動排程: 併發上限 {self.concurrencyPerModel}")
                self.initializedModelSet.add(task.model)

            rawOutput = await self._safeGenerate(task)
            await self._appendCsv(task, rawOutput)

    async def _safeGenerate(self, task: LLMTask) -> RawOutput:
        """送出 LLM 請求；例外與空回應都統一回 Error 字串，讓批次能繼續且下游用同方式辨識。"""
        try:
            rawOutput = await self.ollamaClient.generate(
                task.model, task.sysPrompt, task.userPrompt
            )
        except Exception as e:
            logging.error(f"[Engine] 任務 {task.taskID} 失敗: {e}")
            rawOutput = ""

        return rawOutput or "Error: Max retries exceeded or connection failed"

    async def _appendCsv(self, task: LLMTask, rawOutput: RawOutput) -> None:
        """以 fileLock 序列化 append 寫 raw.csv；fsync 確保中斷時 checkpoint 不遺失最後一筆。"""
        rowDataDict = {
            "model":        task.model,
            "promptID":     task.promptID,
            "taskID":       task.taskID,
            "systemPrompt": task.sysPrompt,
            "userPrompt":   task.userPrompt,
            "rawOutput":    rawOutput,
            "pairs":        json.dumps(task.pairs, ensure_ascii=False),
            "context":      json.dumps(task.context, ensure_ascii=False),
        }

        async with self.fileLock:
            fileExists = os.path.isfile(self.outputFile)
            with open(self.outputFile, 'a', encoding='utf-8-sig', newline='') as f:
                # 固定 fieldnames = RAW_CSV_SCHEMA，保證欄位順序與下游驗證一致
                csvWriter = csv.DictWriter(f, fieldnames=RAW_CSV_SCHEMA)
                if not fileExists:
                    csvWriter.writeheader()
                csvWriter.writerow(rowDataDict)
                f.flush()
                os.fsync(f.fileno())

    async def close(self):
        """釋放底層 httpx 連線池。Pipeline 結束時務必呼叫。"""
        await self.ollamaClient.close()
