# Migrating from version 1

This guide is for users of the original tool: the single-folder `app.py` and `llm_batch.py`
("Corpus LLM Runner"). Version 2 is a rebuild of the same idea, with the same presets, the same
models and the same caching principle. You can bring your saved outputs and evaluation history
across, and a few behaviours have changed on purpose. All the changes are listed below.

## At a glance

| | Version 1 | Version 2 |
|---|---|---|
| Start | `streamlit run app.py` next to `llm_batch.py` | `batchtextreview ui`, or `streamlit run app.py` in the repository |
| Input | One CSV in a hard-coded folder, with a column named `TEXT` | Upload, open from any folder, or use the sample. CSV, Excel, Parquet, JSON and more. Any text column |
| Saved outputs | `results/<run_id>.parquet` + `<run_id>.json`, one pair per configuration | One SQLite database, `results/batchtextreview.db` |
| Evaluation history | `results/evals.csv`, appended on every rerun | History page, one line per evaluation, updated in place |
| Failed rows | `[ERROR: ...]` written into `OUTPUT` | `STATUS` = `error`, message in `ERROR`, `OUTPUT` empty |
| `LABEL` for blank or failed rows | `[invalid]` | Empty |
| Metrics | Accuracy | Accuracy, macro-F1, Cohen's kappa, per-class, by slice |
| CSV download | UTF-8 without BOM, so accents are garbled in Excel | UTF-8 with BOM, plus an Excel download |
| Weights (precision) | Always 4-bit | Auto: 4-bit on an NVIDIA GPU, fp16 on Apple silicon, fp32 on CPU |
| Long runs | Stopped by any click; no cancel | Run in the background, with progress, time remaining and Cancel |
| Who can open the app | Anyone on your network (Streamlit's default) | This computer only, unless you choose otherwise |
| Scripting | `llm_batch.run_batch(...)` | `batchtextreview run ...`, or `engine.execute_run(...)` from Python |

## Bringing your old results across

Version 1 wrote everything into a `results` folder inside the folder you launched it from. That
folder holds `<run_id>.parquet` and `<run_id>.json` files plus `evals.csv`. The import reads those
files. It never modifies or deletes them.

**In the app (one click).** Do either of these:

- copy the old `*.parquet`, `*.json` and `evals.csv` files into version 2's `results` folder (by
  default `results` inside the folder you launch version 2 from); or
- point version 2 at the old folder when you launch it:
  `batchtextreview ui --results-dir "C:\path\to\old\project\results"`, or set `BTR_RESULTS_DIR`
  to that folder. Version 2's database is then created in that same folder, next to the old
  files.

Then open the **History** page. It finds the old files and offers to import them. Click
**Import**. A summary shows what was imported, and lists anything that was skipped and why. The
History page offers the import only once. If you add more old files later, use the command
line.

**From the command line,** give the old folder:

```bash
batchtextreview import-legacy "C:\path\to\old\project\results"
```

This imports into version 2's current results database: `results` in the folder you run it from,
or `BTR_RESULTS_DIR` if you set it. Add `--results-dir DIR` to import into the database in `DIR`
instead. Use the same folder when you launch the app, or the imported history won't appear.

What the import does:

- **Each checkpoint becomes a version 2 run.** Its version 1 model name is mapped to the model's
  Hugging Face repo (see the table below), its precision is recorded as 4-bit (version 1 always
  loaded 4-bit), and its outputs are copied in. Outputs already produced by version 2 are never
  overwritten.
- **Each line of `evals.csv` becomes a History entry** marked as version 1. Where possible it is
  linked to the imported run. Macro-F1 and kappa are empty for these lines, because version 1 did
  not record them.
- **Running the import twice is safe.** Nothing is duplicated.
- **Some files are skipped**, each with a reason: unreadable files, checkpoints whose model is not
  in version 2's catalogue, and `.parquet` files without their `.json` manifest (the manifest is
  what says which model and prompt produced them).

### When imported outputs are reused

A version 2 run reuses imported outputs when its configuration matches the version 1 run:

- the same model;
- weights 4-bit, which is what *Auto* picks on an NVIDIA GPU with bitsandbytes installed, the
  setup version 1 required;
- the same final prompt. The built-in presets use exactly the same wording as in version 1, so
  preset runs match;
- the same max output tokens;
- max input characters 8000, the version 1 value.

Rows are matched by their text, exactly as before, so a different file containing the same texts
also benefits.

One subtle difference: version 2 trims spaces from the start and end of the `{var}` value. A
version 1 run whose variable began or ended with a space therefore does not match, and its rows
are generated again.

Run ids are different in version 2. Version 1 hashed the model's *display label*, so renaming a
model orphaned its outputs. Version 2 hashes the repo id, so labels can change freely. The import
translates the old ids.

## Model names

The model labels no longer carry hints such as "(fastest)" or "(gated)". These now appear as
badges next to the model. Imported history is mapped to the new names, and the command line's
`--model` still accepts the old labels.

| Version 1 label | Version 2 label |
|---|---|
| Granite-4.0-350M (tiny) | Granite-4.0-350M |
| Granite-4.0-1B | Granite-4.0-1B |
| Qwen2.5-1.5B-Instruct (fastest) | Qwen2.5-1.5B-Instruct |
| Granite-4.0-Micro 3B | Granite-4.0-Micro |
| Qwen2.5-3B-Instruct (faster) | Qwen2.5-3B-Instruct |
| Llama-3.2-3B-Instruct (gated) | Llama-3.2-3B-Instruct |
| Phi-3.5-mini-Instruct | Phi-3.5-mini-Instruct |
| Phi-4-mini-Instruct (3.8B) | Phi-4-mini-Instruct |
| Qwen3-4B-Instruct-2507 | Qwen3-4B-Instruct-2507 |
| Qwen3.5-4B (experimental) | Qwen3.5-4B |
| Qwen2.5-7B-Instruct (best quality) | Qwen2.5-7B-Instruct |
| Llama-3.1-8B-Instruct (gated) | Llama-3.1-8B-Instruct |
| Qwen3-8B | Qwen3-8B |
| Qwen3.5-9B (experimental) | Qwen3.5-9B |

## Scripts that used `llm_batch`

For a whole file, the command line is simplest: `batchtextreview run data.csv --preset Sentiment
--model Qwen2.5-3B-Instruct` writes a results file next to the input (see "Command line" in the
[README](../README.md#command-line)). Scripts that called `llm_batch.run_batch(label, texts,
template)` can call the engine directly instead:

```python
from batchtextreview import engine, models
from batchtextreview.backends.manager import ModelManager
from batchtextreview.config import Settings
from batchtextreview.store import Store

config = engine.RunConfig.build(
    models.resolve_model("Qwen2.5-3B-Instruct (faster)"),  # version 1 labels still work
    "Classify sentiment as positive/negative/neutral.\n\nTEXT:\n{text}",
    max_new_tokens=256,
    precision="4bit",  # version 1's precision; "auto" picks one for this machine
)
result = engine.execute_run(
    config,
    ["some text", "more text"],
    store=Store(Settings.from_env().db_path),  # shares saved outputs with the app
    manager=ModelManager(),  # holds the loaded model: keep one per script
    batch_size=8,
)
outputs, stats = result.outputs, result.stats
```

The differences from `run_batch`:

- `stats` is an object, so `stats["cached"]` becomes `stats.cached`. The run id is
  `result.run_id`.
- A failed row has an empty output. `result.statuses` says what happened to each row and
  `result.errors` holds the messages, as described below.
- A model that fails to load no longer raises: `result.aborted_reason` holds the message and the
  rows stay `pending`.
- Outputs are saved in version 2's database rather than as parquet files, so the app and the
  command line reuse them. Pass `store=None` to neither read nor save outputs.

The README's [Python section](../README.md#using-it-from-python) also shows how to load a file and
score the outputs.

## Behaviour changes you will notice

### OUTPUT no longer contains `[ERROR: ...]`

When a batch failed, version 1 wrote `[ERROR: <message>]` into `OUTPUT` for its rows. The
evaluation then scored those rows as invalid predictions, which made the model look worse than it
was. Version 2 keeps the two apart:

- `OUTPUT` is empty for a failed row;
- `STATUS` says what happened to every row: `generated` (by this run), `cached` (a saved output;
  shown as "Saved" in the app), `blank` (empty text, never sent to the model), `error`, or
  `pending` (not run because the run was cancelled or stopped early);
- `ERROR` holds the message. This column only appears when at least one row failed.

Failed, blank and not-run rows are **excluded** from scoring and reported as a separate count.
They are not counted as wrong. If a script of yours looked for `OUTPUT` starting with `[ERROR`,
test `STATUS == "error"` instead.

### LABEL is blank for rows that can't be scored

Version 1 put `[invalid]` in `LABEL` for every output that didn't resolve to an allowed answer,
including blank texts and failed rows. In version 2, `LABEL` is empty for blank, failed and
not-run rows. `[invalid]` now means only one thing: the model produced an answer that didn't
resolve to exactly one allowed label.

Answer matching has also improved. Version 2 ignores lead-ins such as "Answer:" and is
negation-aware ("not positive" no longer counts as positive). It matches "3.0" to 3 and reads
"4/5" as 4. Empty gold values are always skipped. For open-ended tasks, version 1 turned them into
the text "nan" and scored them. Scoring the same outputs can therefore give slightly different
numbers from your version 1 history.

### Evaluation history is deduplicated

Version 1 appended a line to `evals.csv` every time you clicked Run with a gold column selected,
even when the run only reloaded saved outputs. Repeating a run therefore duplicated the line.
Version 2 keeps one History line per combination of configuration (run id), data, gold column and
label set. Scoring it again updates that line. Lines
imported from version 1 are kept as they were, including any duplicates. You can delete those on
the History page.

### CSV export is UTF-8 with BOM

Downloads now open correctly in Excel on Windows, with £ signs, accents and curly quotes intact.
There is also an **Excel** download. Two related changes:

- In the CSV, text cells that a spreadsheet would run as a formula get a leading `'`. These are
  cells starting with `=`, `+`, `-` or `@` that aren't plain numbers. Numbers such as `-5` are left
  alone. In the Excel download, text always stays text.
- Reading a version 2 CSV with pandas: use `pd.read_csv(path, encoding="utf-8-sig")`, so that the
  BOM doesn't end up in the first column's name. Version 2 itself reads both kinds.

The download is no longer called `llm_results.csv`. Its name now records what it contains:
`<dataset>_<task>_<model>_<YYYYMMDD-HHMM>.csv`.

### The text column is selectable

You no longer need a column called `TEXT`. The app guesses the text column and you can change it.
`INDEX` is no longer special either: pick it (or any unique column) as the optional ID column,
which then identifies rows in the Results view. If you run on a file you exported earlier, the new
result columns get free names (`OUTPUT_2`, `LABEL_2`, …) instead of overwriting the old ones.

### No hard-coded default folder

Version 1 opened at one particular user's Windows folder. Version 2 starts in `BTR_DATA_DIR`
(default: the folder you launched from) and then remembers the last folder you used. Paths pasted
with quotes, as Windows' "Copy as path" produces them, are accepted.

### Smaller changes

- **Default row scope.** Version 1 processed the first 20 rows by default. Version 2 selects all
  rows for files up to 500 rows, and the first 20 for bigger files ("First rows"). "Random
  sample" is new.
- **Precision is chosen per device**, so the tool now also runs on CPU-only machines and Macs. The
  setting is called **Weights** (under **Advanced** on the Run page; `--precision` on the command
  line). Saved outputs are tied to it. Version 1 outputs (4-bit) are reused only by 4-bit runs.
- **The app only accepts connections from this computer.** Version 1 ran with Streamlit's
  default, which listens on every network interface. To share the app on your network as before,
  start it with `batchtextreview ui -- --server.address 0.0.0.0`, and only on a network you trust:
  anyone who can reach it can open data files on this computer's disk.
- **"Clear cache" is now "Discard saved outputs"**, on the Run page under the run plan. It deletes
  the saved outputs of the current configuration only. Every configuration's saved outputs are
  listed, and can be deleted, under **Models & settings → Saved outputs**.
- **A log file.** The app and the command line log to `results/logs/batchtextreview.log`. Model
  load failures are written there in full.
- **A `{var}` value containing `{text}` is rejected.** Version 1 expanded it once per row.
- **Models without a chat template** now work: the plain prompt is sent instead of every batch
  failing.
- **A failed model load** shows a clear message with a fix, instead of a traceback. It also no
  longer breaks every later run with that model.
- **Only one run uses the GPU at a time**, even with several browser tabs open. While a run is
  in progress, the Run button in the other tabs is disabled until it finishes.

## Afterwards

Once you have checked the History page, you can archive the old `*.parquet`, `*.json` and
`evals.csv` files. Version 2 no longer needs them. Version 1 itself still works side by side with
version 2 if you need it, but it does not see results produced by version 2.

To back up version 2's data, copy `results/batchtextreview.db` while the app is closed.
