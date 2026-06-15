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
    def loadConfig(configPath="promptGenerate\prompt_config.yaml"):
        if not os.path.exists(configPath):
            raise FileNotFoundError(f"找不到設定檔: {configPath}")
        with open(configPath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def loadYaml(yamlPath):
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

        emptyKeys = [k for k, v in result.items() if not isinstance(v, dict)]
        if emptyKeys:
            print(f"以下 method 內容為空，已略過: {emptyKeys}")
            logging.warning(f"略過空 method: {emptyKeys}")
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
    def __init__(self, methodPoolDict, configDict):
        self.methodPoolDict = methodPoolDict
        self.cfgDict = configDict
        self.generatedPromptList = []

    def generate(self):
        b_isExhaustiveCmb = self.cfgDict.get('b_isExhaustiveCmb', True)
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
        methodOrder = list(self.methodPoolDict.keys())
        sortedMethods = sorted(methodOrder, key=len, reverse=True)

        def parsePartId(partStr):
            for method in sortedMethods:
                if partStr.startswith(method):
                    methodIdx = methodOrder.index(method)
                    itemIdx = int(partStr[len(method):])
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
        settingsDict = self.cfgDict.get('autoSettings', {})
        methodsList = settingsDict.get('selectedMethods', list(self.methodPoolDict.keys()))
        if methodsList == ['ALLMethod']:
            targetMethodList = list(self.methodPoolDict.keys())
        else:
            invalidMethods = [m for m in methodsList if m not in self.methodPoolDict]
            if invalidMethods:
                print(f"以下 method 在 YAML 中找不到，將跳過: {invalidMethods}")
                logging.warning(f"無效 method: {invalidMethods}")
            targetMethodList = [m for m in methodsList if m in self.methodPoolDict]

        if not targetMethodList:
            logging.error("沒有任何有效的 method，請檢查 prompt_config.yaml 的 selectedMethods 設定")
            print("錯誤: 沒有任何有效的 method，請檢查 prompt_config.yaml 的 selectedMethods 設定")
            return

        maxSize = settingsDict.get('maxSize', len(targetMethodList))
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
            manualKeyList = self.cfgDict.get('manualCombinations', [])
            if not manualKeyList:
                print("警告: manualCombinations 為空，沒有任何組合可以生成")
                logging.warning("manualCombinations 為空")
                return
            flatPoolDict = {}

            for catStr, itemsDict in self.methodPoolDict.items():
                if not isinstance(itemsDict, dict):
                    continue
                for k, textStr in itemsDict.items():
                    kStr = str(k).zfill(2)
                    fullIdStr = f"{catStr}{kStr}"
                    flatPoolDict[fullIdStr] = {'promptID': fullIdStr, 'promptText': textStr.strip()}

            for comboKeyList in manualKeyList:
                idList, textList = [], []
                for itemKeyStr in comboKeyList:
                    if itemKeyStr in flatPoolDict:
                        idList.append(flatPoolDict[itemKeyStr]['promptID'])
                        textList.append(flatPoolDict[itemKeyStr]['promptText'])
                    else:
                        print(f"警告: 找不到指定的 Prompt ID '{itemKeyStr}'，將跳過此項目。")

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

    def exportToCsv(self, generatedPromptList):
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
    def __init__(self, configDict):
        self.cfgDict = configDict
        self._validateConfig()

        # 1. 處理設定與路徑
        self.yamlPath = configDict.get("promptYamlPath", "configs/prompts.yaml")
        rawOutputPath = Path(configDict.get('promptCmbOutputPath', 'prompt_output/generated_prompt_list.csv'))
        self.outputDirPath = PromptConfig.ensureDirectories(rawOutputPath.parent)
        self.fileName = rawOutputPath.stem

        # 2. 讀取資料
        self.methodPoolDict = PromptConfig.loadYaml(self.yamlPath)
        if not self.methodPoolDict:
            raise ValueError(f"'{self.yamlPath}' 中沒有任何有效的 method，請確認 YAML 格式")

        # 3. 初始化生成器
        self.generatorObj = PromptGenerator(self.methodPoolDict, self.cfgDict)
        self.generatedPromptList = []

    def _validateConfig(self):
        b_isExhaustiveCmb = self.cfgDict.get('b_isExhaustiveCmb', True)
        if not isinstance(b_isExhaustiveCmb, bool):
            raise ValueError(f"b_isExhaustiveCmb 設定錯誤: '{b_isExhaustiveCmb}'，請填入 true 或 false")

        if not b_isExhaustiveCmb and not self.cfgDict.get('manualCombinations'):
            print("警告: b_isExhaustiveCmb 為 false (手動模式) 但未設定 manualCombinations")
            logging.warning("manual 模式但 manualCombinations 未設定")

        if b_isExhaustiveCmb:
            maxSize = self.cfgDict.get('autoSettings', {}).get('maxSize')
            if maxSize is not None and (not isinstance(maxSize, int) or maxSize < 1):
                raise ValueError(f"maxSize 設定錯誤: '{maxSize}'，請填入正整數")

    def generateCombinations(self):
        """呼叫生成器進行邏輯運算"""
        self.generatedPromptList = self.generatorObj.generate()
        return self.generatedPromptList

    def exportPromptFiles(self):
        """呼叫匯出器處理存檔"""
        print(f"\n正在準備輸出 Prompt 列表至目錄: {self.outputDirPath} ...")
        exporterObj = PromptExporter(self.outputDirPath, self.fileName)
        return exporterObj.exportToCsv(self.generatedPromptList)

# ==========================================
# 主程式進入點
# ==========================================
if __name__ == "__main__":
    print("啟動階段一：生成與匯出 Prompt 組合")

    try:
        cfgDict = PromptConfig.loadConfig()
        pmObj = PromptManager(configDict=cfgDict)
        pmObj.generateCombinations()
        csvPath = pmObj.exportPromptFiles()

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
