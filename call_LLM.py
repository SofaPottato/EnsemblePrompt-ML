# call_LLM.py (主程式)
import argparse
import logging
import sys
import os
from pathlib import Path
from llm_modules.utils import ReadLLMConfig, initializeGlobalLogger, setupSeed
from llm_modules.Pipeline import ExperimentPipeline

_ROOT = Path(__file__).parent
os.chdir(_ROOT)
def startLLMPipeline() -> int:
    """初始化 logger/seed、建立 Pipeline 並執行。回傳 0 成功，1 失敗。"""
    parserObj = argparse.ArgumentParser(description="LLM Inference Runner")
    parserObj.add_argument('--config', type=str, default=str(_ROOT / 'configs' / 'BC5CDR_config.yaml'),
                           help='Path to YAML config file (e.g. configs/_config.yaml)')
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
