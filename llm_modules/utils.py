import yaml
import json
import logging
import os
import random
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Any
from .schemas import PipelineConfig, TaskBuildError


# 全專案統一的 CSV 編碼：utf-8-sig 帶 BOM，讓 Excel 開啟時正確辨識 UTF-8，中文不亂碼。
# 讀寫兩端都引用同一常數，避免某處漏帶 sig 造成編碼不一致。
CSV_ENCODING = 'utf-8-sig'
# 寫出 CSV 的共用參數：不輸出 pandas 預設 index、統一編碼。所有 to_csv 都用 **CSV_WRITE_KWARGS。
CSV_WRITE_KWARGS = {'index': False, 'encoding': CSV_ENCODING}


def sanitizeFilename(name: Any) -> str:
    """將 promptID / runKey 中的特殊字元置換成 '_'，確保檔名安全。新增字元在此擴充。"""
    # promptID / runKey 會直接拿來當檔名（如 singleOutput/<promptID>_result.csv）。
    # 以下字元在 Windows/路徑語意上不合法或有歧義：':' '/' '|' 是保留字元、' ' 易踩雷、
    # '+' 在某些情境會被當特殊符號。集中在這裡置換，呼叫端就不必各自處理。
    return (str(name)
            .replace(":", "_")
            .replace("+", "_")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("|", "_"))


def parseJsonCell(value: Any, fieldName: str, taskID: str) -> Any:
    """
    解析 Task CSV 的 JSON 欄位字串。
    None / NaN → raise TaskBuildError；其他非字串 → 原樣回傳；字串 → json.loads（失敗往上拋）。
    """
    # 正常路徑：pandas 讀進來的 JSON 欄位是字串，直接交給 json.loads；
    # 解析失敗讓 JSONDecodeError 往上冒，因為壞掉的 pairs 沒有合理的預設值。
    if isinstance(value, str):
        return json.loads(value)
    # 空欄（None / NaN）代表這筆 task 根本沒有 pair 可跑 → fail-fast，附上 taskID 方便定位。
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise TaskBuildError(f"Task {taskID} 的欄位 '{fieldName}' 為空。")
    # 已是 list/dict（程式式呼叫、非從 CSV 讀）→ 原樣回傳，不重複解析。
    return value

class ConfigLoader:
    """載入 YAML 設定檔並透過 Pydantic (PipelineConfig) 驗證。"""
    def __init__(self, configPath: str):
        # 載入後立刻丟進 Pydantic：所有相容性檢查（taskType、labelSet…）都在 model 建構時觸發，
        # 因此 config 一旦建成，下游就能信任它的合法性，不必再各自驗。
        rawYamlDict = self.loadYaml(configPath)
        self.config: PipelineConfig = PipelineConfig(**rawYamlDict)

    def loadYaml(self, path: str) -> Dict[str, Any]:
        """以 UTF-8 載入 YAML 並回傳 dict。"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)


def initializeGlobalLogger(logDir: str = "./logs", logName: str = "experiment.log") -> None:
    """
    設定全域 Logger，同時輸出到檔案與標準輸出。
    httpx logger 拉到 WARNING，避免推論時被連線層 INFO 訊息淹沒。
    """
    os.makedirs(logDir, exist_ok=True)
    logPath = os.path.join(logDir, logName)

    # force=True：洗掉任何既有 handler，確保重複呼叫（如測試）或第三方套件先動過 logging 時，
    # 設定仍以這裡為準。同時掛檔案 + stdout 兩個 handler，讓 log 既留存又即時可見。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  
        force=True,
        handlers=[
            logging.FileHandler(logPath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.info(f"[Logger] 初始化完成 → {logPath}")

def setupSeed(seed: int = 42) -> None:
    """固定 Python random / NumPy / PYTHONHASHSEED，確保實驗可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"[Setup] 隨機種子設定為 {seed}")
