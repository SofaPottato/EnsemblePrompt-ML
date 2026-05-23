# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Inference & evaluation pipeline that runs biomedical relation-extraction tasks (PPI, Chemical–Disease, etc.) through a local Ollama server, sweeping prompt × model combinations and producing classification metrics. Python ≥ 3.11; user-facing logs and most code comments are in Traditional Chinese.

## Common commands

```bash
# 1) Preprocess raw dataset → standard Task CSV (run whichever matches the config)
python preprocess/lll.py        # LLL (PPI, single-target)
python preprocess/bc5cdr.py     # BC5CDR (chemical-disease, multi-target)

# 2) Run the full pipeline against a config
python call_LLM.py --config configs/PPI_config.yaml
python call_LLM.py --config configs/BC5CDR_config.yaml
```

Prereqs: `ollama serve` running on `http://localhost:11434`, target models already `ollama pull`-ed, `pip install -r requirements.txt`. `call_LLM.py` returns exit code 0/1 for CI.

There is no test suite, linter, or build step configured — manual inspection of `data/<dataset>/output/eval/` is the validation path.

## Architecture

Six-stage pipeline orchestrated by [llm_modules/Pipeline.py](llm_modules/Pipeline.py) (`ExperimentPipeline.run`): **Load → Build → Inference → Parse → Process → Evaluate**. Stages communicate via files on disk, not return values — each stage reads what the previous wrote. The output chain is:

```
Task CSV + Prompt CSV ─▶ raw.csv ─▶ result.csv ─▶ partialInfo.csv / fullInfo.csv ─▶ eval/
```

### Single-target vs multi-target mode

The whole config behaves differently based on whether `pairColumns` is empty. This switch is enforced by `LLMAppConfig.validateTargetMode` in [llm_modules/schemas.py](llm_modules/schemas.py):

- **single-target** (`pairColumns: []`): Task CSV carries one label per row in `labelColumn`. `Pipeline._buildTaskBatches` synthesizes a single-element `pairs` list from that column. `maxPairsPerBatch` **must be 1**. `pairTemplate` must be absent. Ollama JSON schema is `{label}`.
- **multi-target** (`pairColumns: [...]`): Task CSV needs a `pairs` JSON column (list of dicts containing `itemID`, `label`, and the `pairColumns`). `pairTemplate` is required and rendered once per pair into `{pairs}`. Batches are sliced by `maxPairsPerBatch`. Ollama schema is `{answers: [{id, label}]}`.

When editing prompt rendering or batching, branch on `config.isSingleTarget`, not on `pairColumns` directly.

### Checkpoint / resume

[llm_modules/OllamaEngine.py](llm_modules/OllamaEngine.py) appends one row to `raw.csv` per task with `flush() + fsync()`, so `raw.csv` *is* the checkpoint — there is no separate state file. On rerun, `Pipeline.loadCompletedRunKeys` rebuilds the set of completed `(model, promptID, taskID)` tuples and `buildPendingTasks` filters them out. Consequences:

- The composite key columns (`RUN_KEY_COLUMNS` in OllamaEngine.py) and the column order (`RAW_CSV_SCHEMA`) are the single source of truth — both reader and writer reference these constants. If you change the schema, you must delete or migrate existing `raw.csv` files; mismatches raise `DataLoadError`.
- If inference fails for a task, the engine writes `"Error: …"` into `rawOutput` rather than raising. The row still counts as completed for checkpoint purposes; `OutputParser` then assigns `predLabel = -1`.
- If both `pendingTaskList` and `completedRunKeySet` are empty, the pipeline raises rather than silently skipping to eval.

### Concurrency model

Two-level semaphores in `LLMEngine`: `maxConcurrentModels` gates how many models load simultaneously (outer), `concurrencyPerModel` gates in-flight requests per model (inner, lazily created via `defaultdict`). All disk appends are serialized by a single `asyncio.Lock` (`fileLock`). `runInference` wraps the whole thing in `asyncio.run` and `close()`s the httpx pool in `finally`.

### Path resolution

`PathsConfig` (in schemas.py) normalizes paths in a `model_validator`: any of `rawOutputPath`/`resultPath`/`partialInfoPath`/`fullInfoPath`/`promptPreviewPath`/`singlePromptCmbOutputDir`/`evalDir` left as `null` defaults to `outputRoot/<DEFAULT_NAMES name>`; if set as a relative path, it resolves under `outputRoot`; absolute paths pass through. All parent directories are `mkdir(parents=True, exist_ok=True)`-ed at config-load time, so downstream code never creates directories.

### Label encoding

`Classification.classes` is the source of truth — the list **index** is the integer code (e.g. `["no","yes"]` → no=0, yes=1). Conversion is via `labelToCode`, which strips whitespace and compares case-insensitively; anything unknown becomes `-1`. The same `classes` list is also serialized into the Ollama `format` JSON schema (`buildResponseSchema`), so the model is constrained to return one of those strings.

This means preprocessors **must** emit labels whose lowercase/stripped form matches an entry in `labelSet.classes` (config key `labelSet`, typed as `Classification`). `LLMResultProcessor._convertTrueLabel` warns when this alignment fails — that warning is the canonical signal of a preprocess/config mismatch.

`-1` is excluded from metric calculations in `Evaluate.py` but counted as wrong in the correctness heatmap / "samples to review" analysis.

### Reserved pair fields

`RESERVED_PAIR_FIELDS = {'itemID', 'label'}` (schemas.py) are internal-only — `PromptFormatter._extractPairFields` strips them before rendering so they never leak into the LLM prompt. When adding metadata to pair dicts, either keep it out of `pairColumns` or update this frozenset.

### Output artifacts

- `raw.csv` — append-only log of every LLM call (schema = `RAW_CSV_SCHEMA`). Checkpoint source of truth.
- `result.csv` — long format, one row per pair × (model, promptID), with `predLabel` ∈ {-1, 0..N-1}.
- `partialInfo.csv` — wide pivot, one row per sample, one column per `model_promptID` (e.g. `llama3.2:1b_p01`). Consumed by Evaluate.
- `fullInfo.csv` — same shape as partialInfo plus `__raw` / `__sysPrompt` suffix columns for manual review.
- `eval/evalSummary.csv`, `eval/samplesToReview.csv`, `eval/correctnessHeatmap.png`, `eval/plots/CM_*.png`.

## Working with this codebase

- All errors flow through the `PipelineError` hierarchy in schemas.py (`DataLoadError`, `TaskBuildError`, `InferenceError`, `ParsingError`); `call_LLM.py` is the single top-level catch. Don't swallow these inside stages.
- Naming convention is `camelCase` for variables/methods (unusual for Python) and `PascalCase` for classes — match existing style when editing.
- The `data/` tree, `logs/`, and `data/prompt_output/` are gitignored output dirs; don't commit anything under them. `docs/` is also gitignored.
- When changing `raw.csv` columns, also update `RAW_CSV_SCHEMA` and `RUN_KEY_COLUMNS` in OllamaEngine.py — `Pipeline.loadCompletedRunKeys` validates against them.
