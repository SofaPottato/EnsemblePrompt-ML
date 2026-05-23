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
from .schemas import LLMAppConfig, TaskBuildError


def sanitizeFilename(name: Any) -> str:
    """將 promptID / runKey 中的特殊字元置換成 '_'，確保跨 OS 檔名安全。新增字元在此擴充。"""
    return (str(name)
            .replace(":", "_")
            .replace("+", "_")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("|", "_"))


def parsePairListField(value: Any, fieldName: str, taskID: str) -> Any:
    """
    解析 Task CSV 的 JSON 欄位字串。
    None / NaN → raise TaskBuildError；其他非字串 → 原樣回傳；字串 → json.loads（失敗往上拋）。
    """
    if isinstance(value, str):
        return json.loads(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise TaskBuildError(f"Task {taskID} 的欄位 '{fieldName}' 為空。")
    return value

class ReadLLMConfig:
    """讀取 YAML 設定檔並透過 Pydantic (LLMAppConfig) 驗證。"""
    def __init__(self, configPath: str):
        rawYamlDict = self.loadYaml(configPath)
        self.config: LLMAppConfig = LLMAppConfig(**rawYamlDict)

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
    logPathStr = os.path.join(logDir, logName)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
        force=True,
        handlers=[
            logging.FileHandler(logPathStr, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.info(f"[Logger] 初始化完成 → {logPathStr}")

def setupSeed(seed: int = 42) -> None:
    """固定 Python random / NumPy / PYTHONHASHSEED，確保實驗可重現。Ollama 端隨機性由 temperature=0 控制。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"[Setup] 隨機種子設定為 {seed}")
