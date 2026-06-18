import os
import logging
import yaml
import pandas as pd
from itertools import combinations, product
from pathlib import Path

# ==========================================
# 負責處理資料夾與設定讀取
# ==========================================
class PromptConfig:
    @staticmethod
    def loadYaml(configPath="promptGenerate\prompt_config.yaml"):
        if not os.path.exists(configPath):
            raise FileNotFoundError(f"找不到設定檔: {configPath}")
        with open(configPath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def loadMethodPool(yamlPath):
        if not os.path.exists(yamlPath):
            logging.error(f"找不到檔案: {yamlPath}")
            raise FileNotFoundError(f"找不到檔案: {yamlPath}")

        try:
            with open(yamlPath, 'r', encoding='utf-8') as f:
                dataDict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logging.error(f"YAML 格式錯誤: {yamlPath}")
            raise ValueError(f"YAML 格式錯誤，請檢查縮排與語法: {e}")

        if not dataDict:
            logging.warning(f"YAML 檔案內容為空: {yamlPath}")
            return {}

        result = dataDict.get('prompts', dataDict)
        if result is None:
            logging.warning(f"YAML 'prompts' 欄位為空: {yamlPath}")
            return {}

        emptyKeyList = [k for k, v in result.items() if not isinstance(v, dict)]
        if emptyKeyList:
            print(f"以下 method 內容為空，已略過: {emptyKeyList}")
            logging.warning(f"略過空 method: {emptyKeyList}")
            result = {k: v for k, v in result.items() if isinstance(v, dict)}

        return result

    @staticmethod
    def ensureDirectories(outputDirPath):
        outputDirPath = Path(outputDirPath)
        outputDirPath.mkdir(parents=True, exist_ok=True)
        print(f"已確認輸出目錄：\n   - {outputDirPath}\n")
        return outputDirPath

# ==========================================
# 負責核心的 Prompt 生成邏輯
# ==========================================
class PromptGenerator:
    def __init__(self, methodPoolDict, config):
        self.methodPoolDict = methodPoolDict
        self.config = config
        self.generatedPromptList = []

    def generate(self):
        b_isExhaustiveCmb = self.config.get('b_isExhaustiveCmb', True)
        modeName = 'AUTO (Exhaustive)' if b_isExhaustiveCmb else 'MANUAL'
        logging.info(f"Prompt Generation Mode: {modeName}")

        if b_isExhaustiveCmb:
            self.generateAutoMode()
        else:
            self.generateManualMode()

        self.sortResults()
        logging.info(f"成功生成 {len(self.generatedPromptList)} 組 Prompt。")
        return self.generatedPromptList

    def sortResults(self):
        methodOrderList = list(self.methodPoolDict.keys())
        sortedMethodList = sorted(methodOrderList, key=len, reverse=True)

        def parsePartId(part):
            for method in sortedMethodList:
                if part.startswith(method):
                    methodIdx = methodOrderList.index(method)
                    itemIdx = int(part[len(method):])
                    return (methodIdx, itemIdx)
            return (999, 999)

        def sortKey(itemDict):
            partsList = itemDict['promptID'].split(' + ')
            return (len(partsList), tuple(parsePartId(p) for p in partsList))

        try:
            self.generatedPromptList.sort(key=sortKey)
        except Exception as e:
            logging.warning(f"排序時發生錯誤，跳過排序: {e}")

    def generateAutoMode(self):
        autoSettingsDict = self.config.get('autoSettings', {})
        methodsList = autoSettingsDict.get('selectedMethods', list(self.methodPoolDict.keys()))
        if methodsList == ['ALLMethod']:
            targetMethodList = list(self.methodPoolDict.keys())
        else:
            invalidMethodList = [m for m in methodsList if m not in self.methodPoolDict]
            if invalidMethodList:
                print(f"以下 method 在 YAML 中找不到，將跳過: {invalidMethodList}")
                logging.warning(f"無效 method: {invalidMethodList}")
            targetMethodList = [m for m in methodsList if m in self.methodPoolDict]

        if not targetMethodList:
            logging.error("沒有任何有效的 method，請檢查 prompt_config.yaml 的 selectedMethods 設定")
            print("錯誤: 沒有任何有效的 method，請檢查 prompt_config.yaml 的 selectedMethods 設定")
            return

        maxSize = autoSettingsDict.get('maxSize', len(targetMethodList))
        if not isinstance(maxSize, int) or maxSize < 1:
            print(f"maxSize 值 '{maxSize}' 無效，將使用預設值 {len(targetMethodList)}")
            maxSize = len(targetMethodList)
        limitNum = min(maxSize, len(targetMethodList))

        for r in range(1, limitNum + 1):
            for methodComboTuple in combinations(targetMethodList, r):
                promptList = []
                for cat in methodComboTuple:
                    if cat in self.methodPoolDict:
                        promptItemsList = [
                            (f"{cat}{str(k).zfill(2)}", v)
                            for k, v in self.methodPoolDict[cat].items()
                        ]
                        promptList.append(promptItemsList)

                for itemComboTuple in product(*promptList):
                    idList = [item[0] for item in itemComboTuple]
                    textList = [item[1] for item in itemComboTuple]
                    self.addCombination(idList, textList)

    def generateManualMode(self):
            manualKeyList = self.config.get('manualCombinations', [])
            if not manualKeyList:
                print("警告: manualCombinations 為空，沒有任何組合可以生成")
                logging.warning("manualCombinations 為空")
                return
            flatPoolDict = {}

            for cat, itemsDict in self.methodPoolDict.items():
                if not isinstance(itemsDict, dict):
                    continue
                for k, text in itemsDict.items():
                    paddedKey = str(k).zfill(2)
                    fullId = f"{cat}{paddedKey}"
                    flatPoolDict[fullId] = {'promptID': fullId, 'promptText': text.strip()}

            for comboKeyList in manualKeyList:
                idList, textList = [], []
                for itemKey in comboKeyList:
                    if itemKey in flatPoolDict:
                        idList.append(flatPoolDict[itemKey]['promptID'])
                        textList.append(flatPoolDict[itemKey]['promptText'])
                    else:
                        print(f"警告: 找不到指定的 Prompt ID '{itemKey}'，將跳過此項目。")

                if idList:

                    self.addCombination(idList, textList)

    def addCombination(self, idList, textList):
        self.generatedPromptList.append({
            "promptID": " + ".join(idList),
            "promptText": "\n".join(textList)
        })

# ==========================================
# 負責資料匯出 (隨時可抽換成匯出 JSON/Excel)
# ==========================================
class PromptExporter:
    def __init__(self, outputDirPath, fileName):
        self.outputDirPath = Path(outputDirPath)
        self.fileName = fileName

    def saveToCsv(self, generatedPromptList):
        if not generatedPromptList:
            print("警告：目前沒有任何生成的 Prompt 可以匯出！")
            return None

        csvDataList = [{"promptID": p['promptID'], "promptText": p['promptText']} for p in generatedPromptList]
        promptDf = pd.DataFrame(csvDataList)

        csvPath = self.outputDirPath / f"{self.fileName}.csv"
        try:
            promptDf.to_csv(csvPath, index=False, encoding='utf-8-sig')
        except PermissionError:
            print(f"錯誤: 無法寫入 {csvPath}，請確認檔案未被其他程式開啟")
            logging.error(f"寫入 CSV 失敗 (權限不足): {csvPath}")
            return None
        except OSError as e:
            print(f"錯誤: 寫入失敗 ({e})")
            logging.error(f"寫入 CSV 失敗: {e}")
            return None

        print(f"CSV 檔案已儲存至: {csvPath}")
        return csvPath

# ==========================================
# 指揮官 (Manager) - 整合以上模組
# ==========================================
class PromptManager:
    def __init__(self, config):
        self.config = config
        self._validateConfig()

        # 1. 處理設定與路徑
        self.yamlPath = config.get("promptYamlPath", "configs/prompts.yaml")
        rawOutputPath = Path(config.get('promptCmbOutputPath', 'prompt_output/generated_prompt_list.csv'))
        self.outputDirPath = PromptConfig.ensureDirectories(rawOutputPath.parent)
        self.fileName = rawOutputPath.stem

        # 2. 讀取資料
        self.methodPoolDict = PromptConfig.loadMethodPool(self.yamlPath)
        if not self.methodPoolDict:
            raise ValueError(f"'{self.yamlPath}' 中沒有任何有效的 method，請確認 YAML 格式")

        # 3. 初始化生成器
        self.generator = PromptGenerator(self.methodPoolDict, self.config)
        self.generatedPromptList = []

    def _validateConfig(self):
        b_isExhaustiveCmb = self.config.get('b_isExhaustiveCmb', True)
        if not isinstance(b_isExhaustiveCmb, bool):
            raise ValueError(f"b_isExhaustiveCmb 設定錯誤: '{b_isExhaustiveCmb}'，請填入 true 或 false")

        if not b_isExhaustiveCmb and not self.config.get('manualCombinations'):
            print("警告: b_isExhaustiveCmb 為 false (手動模式) 但未設定 manualCombinations")
            logging.warning("manual 模式但 manualCombinations 未設定")

        if b_isExhaustiveCmb:
            maxSize = self.config.get('autoSettings', {}).get('maxSize')
            if maxSize is not None and (not isinstance(maxSize, int) or maxSize < 1):
                raise ValueError(f"maxSize 設定錯誤: '{maxSize}'，請填入正整數")

    def generateCombinations(self):
        """呼叫生成器進行邏輯運算"""
        self.generatedPromptList = self.generator.generate()
        return self.generatedPromptList

    def exportPromptFiles(self):
        """呼叫匯出器處理存檔"""
        print(f"\n正在準備輸出 Prompt 列表至目錄: {self.outputDirPath} ...")
        exporter = PromptExporter(self.outputDirPath, self.fileName)
        return exporter.saveToCsv(self.generatedPromptList)

# ==========================================
# 主程式進入點
# ==========================================
if __name__ == "__main__":
    print("啟動階段一：生成與匯出 Prompt 組合")

    try:
        config = PromptConfig.loadYaml()
        manager = PromptManager(config=config)
        manager.generateCombinations()
        csvPath = manager.exportPromptFiles()

        if csvPath:
            print("\n" + "="*50)
            print("Prompt 檔案已生成！")
            print(f"存放於{csvPath} ")
            print("="*50)

    except FileNotFoundError as e:
        print(f"\n{e}")
    except ValueError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n發生未預期的錯誤: {e}")
