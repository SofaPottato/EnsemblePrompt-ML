# Main_LLM

以 Ollama 為後端、針對生醫關係抽取（PPI、Chemical–Disease 等）任務的 LLM 推論與評估管線。

---

## 專案目標

- 以多 prompt × 多模型的排列組合，在標準化的 Task CSV 上做分類推論。
- 自動斷點續傳、用 JSON schema 強制 LLM 輸出格式、產出寬表格與分類指標報表。
- 一套 config 同時支援 **單目標**（每筆一個標的，如 LLL PPI）與 **多目標**（每筆多個 pair，如 BC5CDR）兩種任務。

## 主要流程

```
原始資料 ──preprocess──▶ Task CSV ──┐
                                   ├──▶ Pipeline ──▶ raw.csv ──▶ result.csv ──▶ partialInfo.csv ──▶ eval/
Prompt 組合 CSV ───────────────────┘                                        └─▶ fullInfo.csv
```

[ExperimentPipeline.run](llm_modules/Pipeline.py#L68) 將整個流程切成六階段，階段間以檔案傳遞、不靠回傳值：

1. **Load** — 載入 Task CSV × Prompt 組合 × 已完成 checkpoint。
2. **Build** — 依 `maxPairsPerBatch` 切批、渲染 `taskTemplate`/`pairTemplate`，扣掉已完成任務後產生 `LLMTask` 清單。
3. **Inference** — [LLMEngine](llm_modules/OllamaEngine.py) 非同步呼叫 Ollama，每筆完成即 append 寫入 `raw.csv` + `fsync()` 落盤。
4. **Parse** — [OutputParser](llm_modules/OutputParser.py) 解析 structured JSON 回應為 0..N-1 的 `predLabel`（無法解析一律 -1）。
5. **Process** — [LLMResultProcessor](llm_modules/LLMResultProcessor.py) 長表轉寬表：每個 `model|promptID` 一欄。
6. **Evaluate** — [PromptCmbEval](llm_modules/Evaluate.py) 計算 Accuracy/Precision/Recall/F1/MCC、混淆矩陣、難題清單與 Upper Bound。

## 系統架構

下圖呈現「進入點 → Pipeline 六階段 → 各階段負責模組 → 落盤產出 → 外部依賴」的整體結構。階段間以檔案傳遞，星號（★）標示的 `raw.csv` 同時是斷點續傳的單一狀態來源。

```
                              ┌──────────────────────────┐
                              │      call_LLM.py         │   進入點
                              │   --config xxx.yaml      │   exit 0 / 1
                              └────────────┬─────────────┘
                                           │  載入 + 驗證 LLMAppConfig
                                           ▼
        ┌────────────────────────────────────────────────────────────────┐
        │        ExperimentPipeline  (llm_modules/Pipeline.py)            │
        │                       六階段流程統籌                             │
        └────────────────────────────────────────────────────────────────┘

   輸入                       階段 / 模組                       落盤產出
 ───────────────         ──────────────────────────         ──────────────────────

 configs/*.yaml ──┐
                  │
 Task CSV ────────┼──▶  ① Load        schemas.py             (記憶體中的 DataFrame)
                  │       └─ 驗證 single/multi-target
 Prompt CSV ──────┘
                                                              promptPreview.csv
                       ② Build       PromptFormatter.py  ──▶  (所有 promptID×task
                          └─ taskTemplate / pairTemplate 渲染   渲染後的 prompt 預覽)
                          └─ maxPairsPerBatch 切批

                       ③ Inference   OllamaEngine.py     ──▶  raw.csv  ★checkpoint
                          └─ 雙層 semaphore                    (append-only;
                          └─ asyncio.Lock 序列化寫檔             flush()+fsync())
                          └─ httpx → Ollama HTTP API
                                          │
                                          ▼
                                ┌──────────────────────┐
                                │  Ollama Server       │   外部依賴
                                │  localhost:11434     │   (ollama serve)
                                └──────────────────────┘

                       ④ Parse       OutputParser.py    ──▶  result.csv
                          └─ structured JSON → predLabel       (長表;一列一 pair)
                          └─ 解析失敗 / Error → -1

                       ⑤ Process     LLMResultProcessor ──▶  partialInfo.csv
                          └─ long → wide pivot                fullInfo.csv
                          └─ trueLabel 對齊 labelSet           (寬表;一列一樣本)

                       ⑥ Evaluate    Evaluate.py        ──▶  eval/
                          └─ Accuracy/Precision/Recall          ├─ evalSummary.csv
                             /F1/MCC                            ├─ samplesToReview.csv
                          └─ 混淆矩陣 / 對錯熱圖                  ├─ correctnessHeatmap.png
                          └─ Upper Bound 分析                   └─ plots/CM_*.png

 ────────────────────────────────────────────────────────────────────────────────
  支援模組
    schemas.py   Pydantic config / PipelineError 家族 / LLMTask / Classification
    utils.py     logger / random seed / YAML 載入 / JSON 寬鬆解析
```

## 目錄結構

```
.
├── call_LLM.py               # 進入點；exit code 0 / 1
├── configs/
│   ├── PPI_config.yaml       # LLL（PPI，single-target）
│   └── BC5CDR_config.yaml    # BC5CDR（Chemical–Disease，multi-target）
├── llm_modules/
│   ├── Pipeline.py           # 流程統籌（六階段）
│   ├── OllamaEngine.py       # 非同步推論引擎；raw.csv schema 在這裡
│   ├── OutputParser.py       # structured JSON → predLabel
│   ├── LLMResultProcessor.py # long → wide pivot
│   ├── Evaluate.py           # 分類指標與圖表
│   ├── PromptFormatter.py    # Template 渲染
│   ├── schemas.py            # Pydantic config / Exception / LLMTask
│   └── utils.py              # logger、seed、YAML 載入、JSON 解析
├── preprocess/
│   ├── lll.py                # LLL → tasks.csv（single-target）
│   └── bc5cdr.py             # BC5CDR → tasks.csv（multi-target）
└── data/                     # 輸入資料與輸出（gitignored）
```

## 環境需求

- Python ≥ 3.11
- Ollama 已安裝並在 `http://localhost:11434` 運行
- 對應模型已 `ollama pull`（例如 `llama3.2:1b`）
- `pip install -r requirements.txt`

## 執行方式

```bash
# 1) 前處理（依資料集擇一，產出標準 Task CSV）
python preprocess/lll.py        # LLL（PPI，single-target）
python preprocess/bc5cdr.py     # BC5CDR（Chemical–Disease，multi-target）

# 2) 跑 Pipeline
python call_LLM.py --config configs/PPI_config.yaml
python call_LLM.py --config configs/BC5CDR_config.yaml
```

成功時 exit code 0，失敗時 exit code 1（任何 `PipelineError` 子類例外都會被 `call_LLM.py` 統一捕捉並記錄到 `logs/llmLog.log`）。

## Single-target vs Multi-target

整個 config 的行為由 `pairColumns` 是否為空決定，並由 [LLMAppConfig.validateTargetMode](llm_modules/schemas.py#L168) 強制檢查：

| 比較項 | Single-target | Multi-target |
|---|---|---|
| `pairColumns` | `[]`（空） | 非空，如 `["e1","e2"]` |
| `labelColumn` | **必填**（如 `"label"`） | 不使用 |
| `pairTemplate` | 不可設定 | **必填** |
| `maxPairsPerBatch` | 必須為 `1` | 任意 ≥1 |
| Task CSV 必要欄位 | `taskID` + `labelColumn` + `contextColumns` | `taskID` + `pairs` (JSON) + `contextColumns` |
| Ollama JSON schema | `{label}` | `{answers: [{id, label}]}` |
| 範例 | LLL（每句一個 PPI 判斷） | BC5CDR（每篇 abstract 多個 chemical–disease pair） |

[Pipeline._buildTaskBatches](llm_modules/Pipeline.py#L249) 在 single-target 模式會自動把 Task CSV 的 `labelColumn` 包成單元素 `pairs`，讓下游一律用 pair 為單位處理。

## 重要 Config 欄位

| 欄位 | 說明 |
|---|---|
| `paths.taskCsvPath` | 前處理產出的 Task CSV |
| `paths.promptCmbPath` | Prompt 組合 CSV（必要欄位：`promptID`, `promptText`） |
| `paths.outputRoot` | 所有輸出的根目錄；其它 `*Path` 未填 / 相對路徑時自動掛在底下 |
| `selectedModels` | 要測試的 Ollama 模型名稱清單 |
| `contextColumns` | Task CSV 中對應 `taskTemplate` 佔位符的欄位 |
| `pairColumns` | `pairs` JSON 中對應 `pairTemplate` 佔位符的欄位；空 → single-target |
| `labelColumn` | single-target 模式下攜帶 true label 的欄位名稱 |
| `taskTemplate` | 主 prompt 模板；佔位符對應 `contextColumns`（multi-target 額外帶 `{pairs}`） |
| `pairTemplate` | multi-target 模式下單筆 pair 的格式化模板（會被渲染到 `{pairs}`） |
| `labelSet` | 分類類別字串清單；清單索引即整數 code（`["no","yes"]` → no=0, yes=1） |
| `maxPairsPerBatch` | 每個 LLM task 包含的 pair 數；single-target 必為 1 |
| `concurrencyPerModel` / `maxConcurrentModels` | 雙層非同步併發上限 |
| `ollamaServer.url` / `timeout` | Ollama API 端點與請求超時秒數 |
| `llmOptions` | 透傳給 Ollama 的推論參數（`temperature`、`num_predict`、`num_ctx`…） |

## Label 編碼

[Classification](llm_modules/schemas.py#L74) 是標籤對應的單一事實來源：

- `classes` 清單**索引**即整數 code（`["no","yes"]` → no=0, yes=1）。
- `labelToCode` 比對時去空白、大小寫不敏感；未命中一律回 `-1`。
- 同一份 `classes` 也會序列化進 Ollama 的 `format` JSON schema，強制模型只能輸出清單裡的字串。
- 因此**前處理產出的 gold label 必須與 `labelSet` 完全對齊**；不對齊時 `LLMResultProcessor._convertTrueLabel` 會記 warning，這是 preprocess/config 不一致最早能被發現的地方。
- `-1`（無法解析或標籤錯誤）在指標計算時被排除，但在難題分析的對錯矩陣中一律計為答錯。

## 斷點續傳

`raw.csv` **就是** checkpoint，沒有額外狀態檔：

- 唯一鍵是 `(model, promptID, taskID)` 三元組（見 [TaskRunID](llm_modules/Pipeline.py#L39) 與 `TASK_RUN_ID_COLUMNS`）。
- 推論完成的每筆任務由 [LLMEngine._appendCsv](llm_modules/OllamaEngine.py#L187) 以 `asyncio.Lock` 序列化 append、`flush()+fsync()` 落盤。
- 重跑時 [Pipeline.loadCompletedTaskRunIDs](llm_modules/Pipeline.py#L157) 重建已完成集合，`buildPendingTasks` 過濾掉這些，不需要任何旗標。
- API 失敗（重試 3 次仍失敗）會寫入 `"Error: ..."` 而非 raise；該列**仍算已完成**，下游 `OutputParser` 看到後將該位置標 `-1`。
- 若改動 `raw.csv` 欄位，必須同步更新 [RAW_CSV_SCHEMA](llm_modules/OllamaEngine.py#L17) 與 `TASK_RUN_ID_COLUMNS`；不一致時讀取會拋 `DataLoadError`，必須刪除或備份舊檔再跑。

## 併發模型

[LLMEngine](llm_modules/OllamaEngine.py#L71) 採雙層 semaphore：

- `maxConcurrentModels` — 同時載入幾個模型（外層，避免 VRAM 爆掉）。
- `concurrencyPerModel` — 每個模型內部同時送出幾個請求（內層，由 `defaultdict` 第一次存取時自動建立）。
- 所有寫檔以單一 `fileLock` 序列化。
- `runInference` 用 `asyncio.run` 啟動，並在 `finally` 中關閉 `httpx` 連線池。

## 輸出檔說明

| 路徑（相對 `outputRoot`） | 內容 |
|---|---|
| `raw.csv` | 推論原始紀錄（append-only checkpoint）；欄位 = `RAW_CSV_SCHEMA` |
| `result.csv` | 長表，每個 pair × `(model, promptID)` 一列；`predLabel` ∈ {-1, 0..N-1} |
| `singleOutput/{promptID}_result.csv` | 同上但依 `promptID` 切分 |
| `partialInfo.csv` | 寬表，一列一樣本；欄位包含 `itemID`, `trueLabel`, 各 `model|promptID__pred` |
| `fullInfo.csv` | 同上再補 `__raw`（rawOutput）與 `__sysPrompt` 後綴欄，供人工審閱 |
| `promptPreview.csv` | 所有 `promptID × task` 渲染後的 userPrompt 預覽 |
| `eval/evalSummary.csv` | 各 `(model, promptID)` 的 Accuracy/Precision/Recall/F1/MCC（按 F1 排序） |
| `eval/samplesToReview.csv` | 所有 runKey 都答錯的難題清單 |
| `eval/correctnessHeatmap.png` | 模型 × 樣本對錯熱圖 |
| `eval/plots/CM_*.png` | 各 runKey 混淆矩陣 |

## 開發注意事項

- 命名風格：變數/方法 `camelCase`、類別 `PascalCase`（不是 PEP 8 標準，但全專案一致）。
- 錯誤統一走 `PipelineError` 家族（`DataLoadError` / `TaskBuildError` / `InferenceError` / `ParsingError`），由 `call_LLM.py` 一處 catch；各階段不要私吞例外。
- `RESERVED_PAIR_FIELDS = {'itemID', 'label'}` 是內部欄位，[PromptFormatter._extractPairFields](llm_modules/PromptFormatter.py#L46) 會剔除，永遠不會洩漏進 prompt。新增 pair metadata 時要嘛不加進 `pairColumns`，要嘛更新這個 frozenset。
- `data/`、`logs/`、`docs/` 都在 gitignore 中，不要把輸出檔 commit 進來。
- 沒有 test suite / linter / build step；驗證方式是手動檢查 `data/<dataset>/output/eval/` 下的輸出。
