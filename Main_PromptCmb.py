# Main_PromptCmb.py (主程式)
import argparse
import logging
import sys
import os
import random
import yaml
import numpy as np
from pathlib import Path
from llm_modules.schemas import PipelineConfig
from llm_modules.Pipeline import ExperimentPipeline

_ROOT = Path(__file__).parent
os.chdir(_ROOT)


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


def loadConfig(configPath: str) -> PipelineConfig:
    """載入 YAML 設定檔並透過 Pydantic (PipelineConfig) 驗證。"""
    # 載入後立刻丟進 Pydantic：所有相容性檢查（taskType、labelSet…）都在 model 建構時觸發，
    # 因此 config 一旦建成，下游就能信任它的合法性，不必再各自驗。
    with open(configPath, 'r', encoding='utf-8') as f:
        rawYamlDict = yaml.safe_load(f)
    return PipelineConfig(**rawYamlDict)


def startLLMPipeline() -> int:
    """初始化 logger/seed、建立 Pipeline 並執行。回傳 0 成功，1 失敗。"""
    parser = argparse.ArgumentParser(description="LLM Inference Runner")
    parser.add_argument('--config', type=str, default=str(_ROOT / 'configs' / 'PPI_config.yaml'),
                           help='Path to YAML config file (e.g. configs/PPI_config.yaml)')
    args = parser.parse_args()
    initializeGlobalLogger(logDir=str(_ROOT / 'logs'), logName="llmLog.log")
    setupSeed(42)

    logging.info("========================================")
    logging.info("        Ollama            ")
    logging.info("========================================")

    try:
        config = loadConfig(args.config)
        pipeline = ExperimentPipeline(config)
        pipeline.run()
        return 0
    except Exception as e:
        logging.critical(f"發生未預期的錯誤: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(startLLMPipeline())
