# Main_LLM

以 Ollama 為後端、針對生醫關係抽取（PPI、Chemical-Disease 等）任務的 LLM 推論與評估管線。

---

## 專案目標

- 以多 prompt × 多模型的排列組合，在標準化的 Task CSV 上做分類推論。
- 自動斷點續傳、結構化解析 LLM 文字回應、產出寬表格與分類指標報表。

## 主要流程

```
原始資料 ──preprocess──▶ Task CSV ──┐
                                   ├──▶ Pipeline ──▶ raw.csv ──▶ result.csv ──▶ partialInfo.csv ──▶ eval/
Prompt 組合 CSV ───────────────────┘
```

1. **Preprocess** — 將資料集（LLL、BC5CDR…）轉為標準 Task CSV（必含 `taskID`、`pairs`，以及 config 的 `contextColumns`）。
2. **Pipeline** — 載入 Task CSV × Prompt 組合 × Models，呼叫 Ollama 做非同步推論。
3. **OutputParser** — 用 regex 把 LLM 文字回應拆成 1/0/-1 的預測標籤。
4. **LLMResultProcessor** — 長表轉寬表，每個 `model_promptID` 一欄。
5. **Evaluate** — 計算 Accuracy/Precision/Recall/F1/MCC、混淆矩陣、難題清單、Upper Bound。

## 目錄結構

```
.
├── call_LLM.py              # 進入點
├── configs/
│   ├── PPI_config.yaml      # LLL（PPI）任務設定
│   └── BC5CDR_config.yaml   # BC5CDR 任務設定
├── llm_modules/
│   ├── Pipeline.py          # 流程統籌
│   ├── OllamaEngine.py      # 非同步推論引擎（含斷點續傳）
│   ├── OutputParser.py      # LLM 文字 → 結構化標籤
│   ├── LLMResultProcessor.py# 長表 → 寬表
│   ├── Evaluate.py          # 分類指標與圖表
│   ├── PromptFormatter.py   # Template 渲染
│   ├── schemas.py           # Pydantic config / Task / Exception
│   └── utils.py             # logger、seed、JSON 解析
├── preprocess/
│   ├── lll.py               # LLL 資料集前處理
│   └── bc5cdr.py            # BC5CDR 資料集前處理
└── data/                    # 輸入資料與輸出結果
```

## 環境需求

- Python ≥ 3.11
- Ollama 已安裝並在 `http://localhost:11434` 運行
- 對應模型已 `ollama pull`（例如 `llama3.2:1b`）
- `pip install -r requirements.txt`

## 執行方式

```bash
# 1) 前處理（依資料集擇一）
python preprocess/lll.py
# 或
python preprocess/bc5cdr.py

# 2) 跑 Pipeline
python call_LLM.py --config configs/PPI_config.yaml
```

成功時 exit code 0，失敗時 exit code 1（可被 shell / CI 偵測）。

## 重要 Config 欄位

| 欄位 | 說明 |
|---|---|
| `paths.taskCsvPath` | 前處理產出的 Task CSV |
| `paths.promptCmbPath` | Prompt 組合 CSV（欄位：`promptID`, `promptText`） |
| `paths.outputRoot` | 所有輸出的根目錄；其它 `*Path` 未填則自動衍生 |
| `selectedModels` | 要測試的 Ollama 模型清單 |
| `maxPairsPerBatch` | 每個 LLM task 包含的 item 數；1 = 單筆模式，>1 = 批次模式 |
| `contextColumns` | Task CSV 中對應 `taskTemplate` 佔位符的欄位 |
| `pairColumns` | `pairs` JSON 中對應 `pairTemplate` 佔位符的欄位 |
| `taskTemplate` / `pairTemplate` | Prompt 模板字串 |
| `labelMap.positive` / `negative` | 真實標籤 → 1/0 的對照表 |
| `concurrencyPerModel` / `maxConcurrentModels` | 非同步併發上限 |

## 斷點續傳

`raw.csv` 即 checkpoint。重跑時 `Pipeline.doGetCompletedTasks` 會讀取已存在的 `taskID` 並跳過，不需要額外旗標。

## 輸出檔說明

| 檔案 | 內容 |
|---|---|
| `raw.csv` | LLM 原始回應（含 timestamp / sysPrompt / userPrompt） |
| `result.csv` | 解析後長表（每個 pair 一列） |
| `partialInfo.csv` | 寬表（僅 dataID / trueLabel / 各 model_prompt 預測） |
| `fullInfo.csv` | 寬表 + 原始 rawOutput（供人工審閱） |
| `singleOutput/{promptID}_result.csv` | 依 promptID 切分的長表 |
| `eval/eval_summary.csv` | 各模型/prompt 的指標（按 F1 排序） |
| `eval/samples_to_review.csv` | 所有模型都答錯的難題清單 |
| `eval/correctness_heatmap.png` | 模型 × 樣本對錯熱圖 |
| `eval/plots/CM_*.png` | 各模型混淆矩陣 |
