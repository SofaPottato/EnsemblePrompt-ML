# call_LLM.py (主程式)
import argparse
import logging
import sys
from pathlib import Path
from llm_modules.utils import ReadLLMConfig, initializeGlobalLogger, setupSeed
from llm_modules.Pipeline import ExperimentPipeline

_ROOT = Path(__file__).parent

def startLLMPipeline() -> int:
    """初始化 logger/seed、建立 Pipeline 並執行。回傳 0 成功，1 失敗。"""
    parserObj = argparse.ArgumentParser(description="LLM Inference Runner")
    parserObj.add_argument('--config', type=str, default=str(_ROOT / 'configs' / 'PPI_config.yaml'),
                           help='Path to YAML config file (e.g. configs/PPI_config.yaml)')
    argsObj = parserObj.parse_args()
    initializeGlobalLogger(logDir=str(_ROOT / 'logs'), logName="llmLog.log")
    setupSeed(42)

    logging.info("========================================")
    logging.info("        Ollama            ")
    logging.info("========================================")

    try:
        configManagerObj = ReadLLMConfig(argsObj.config)
        pipelineObj = ExperimentPipeline(configManagerObj.config)
        pipelineObj.run()
        return 0
    except Exception as e:
        logging.critical(f"發生未預期的錯誤: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(startLLMPipeline())
