# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [2.2.0] - 2026-10-17

### Added

- **Cloud models.** Anthropic's Claude and OpenAI's GPT models run through the providers' APIs,
  next to the local models, in the app and on the command line. Built in: Claude Haiku, Sonnet
  and Opus 5.5, and GPT-6 Luna, GPT-6.1 Sol and GPT-6 Astra. Any other model works as
  `anthropic:<model id>` or `openai:<model id>`, and so do OpenAI-compatible servers
  (`OPENAI_BASE_URL`). Install the SDKs with `pip install -e ".[api]"`.
  - **Before a run**, the Run page shows how many texts go to the provider, about how many tokens,
    and the most it should cost at list prices. A "Send these texts to …" tick, asked once per
    browser session, is needed before anything is sent.
  - **API keys** come from the session, then `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, then the
    system keyring. Manage them under **Models & settings → Cloud models**, which also has a
    connection test. Keys are never stored in the results database or the log.
  - **Each text is one request.** A batch's texts are sent at once (the batch size), with the
    SDKs' retries. A text the provider refuses or rejects fails on its own, and the rest of its
    batch is kept. A rejected key stops the run at once.
  - **Token counts** and the list-price cost appear in Results → Details and at the end of a
    command-line run.
- `engine.RunStats.input_tokens` / `output_tokens`, and the optional backend attributes
  `last_row_errors` and `last_usage`. `GenerationError(fatal=True)` stops a run without retrying
  its rows.

### Removed

- `environment.yml`. The README's NVIDIA GPU install now uses three commands instead:
  - `conda create -n batchtextreview -c conda-forge --override-channels python=3.12 pip`
  - `pip install torch --index-url https://download.pytorch.org/whl/cu130`
  - `pip install -e ".[llm]"`

  `environment.yml` also installed the `dev` tools. For development, use
  `pip install -e ".[llm,dev]"`.

### Fixed

- **CPU-only PyTorch on Windows.** The GPU install could install a CPU-only PyTorch on Windows.
  It searched both PyPI and the CUDA 12.8 index, whose newest torch was older than PyPI's, so pip
  took PyPI's build. That build is CPU-only on Windows. PyTorch now comes from the CUDA 13.0 index
  alone. For GTX 9xx and 10xx cards, use the `cu126` index instead.
- **HTTP 429 from Anaconda's channel.** The install failed on networks that Anaconda's channel
  refuses, because conda added that channel from the user's settings. `--override-channels` keeps
  conda on conda-forge. A `nodefaults` line in the environment file would not have been enough:
  conda 26.7 and newer ignore it in `conda env create` (conda issue #16816).
- **Certificate errors.** The README now lists the certificate errors that git, conda and pip give
  on networks with TLS inspection. It gives the one-off fix for each, to run before `git clone`.

## [2.1.0] - 2026-10-14

### Added

- A processing order setting: **Processing order** under **Advanced** on the Run page (the app
  remembers your choice), `--order longest` or `--order shortest` for the `run` command, and
  `order=` for `engine.execute_run` and `jobs.JobRequest`. The Results page's Details tab shows
  the order a run used, and retrying failed rows or generating the remaining rows keeps it. The
  order isn't part of the run id, so saved outputs are reused whichever order you pick.
- `engine.Progress.work_fraction` (with `generation_s` and `work_done`), the share of the work
  done rather than of rows.

### Changed

- The default order is now longest first (it was shortest first). The time estimate starts high
  and counts down instead of rising whenever the run reaches longer texts, and a model that runs
  out of memory fails in the first batch rather than near the end. Choose shortest first to see
  the first results sooner, or when only the longest texts run out of memory (the out-of-memory
  error card now says so).
- The progress bar (and the command line's percentage) measures the work done rather than rows,
  so a long text counts for more than a short one. It never goes backwards, and the rows of a
  failed batch retried one at a time count as work still to do.
- A steadier time estimate: each update moves only part of the way towards the newest
  calculation, so one unusually fast or slow batch no longer makes it jump.

### Fixed

- After **Retry failed rows** on the Results page, a retry that finished almost at once could
  leave the page on the original run instead of opening the retry.

## [2.0.0] - 2026-10-12

A ground-up rebuild of the version 1 tool as an installable package, with a new interface, a
results database, proper evaluation and a command line. See [docs/MIGRATING.md](docs/MIGRATING.md)
for upgrade steps.

### Added

- Installable package (`pip install -e .`) with a `batchtextreview` command. Its subcommands are
  `ui`, `run` (headless runs), `models`, `presets`, `history` (printed, or saved as CSV or Excel
  with `--output`), `import-legacy` and `device`.
- A documented Python API (`engine.RunConfig.build` and `engine.execute_run`) for scripts and
  notebooks, replacing `llm_batch.run_batch`.
- Four-page Streamlit app (Run, Results, History, Models & settings) with light and dark themes.
- Input from upload, any folder, or bundled sample data, in CSV, TSV, TXT, Excel (with sheet
  choice), Parquet, JSON and JSON Lines. Encoding and delimiter are detected. The text, ID and
  gold-label columns are selectable, and a sensible default is guessed for each.
- Row scope: All, First rows, or a repeatable Random sample.
- Task presets grouped by category, an editable prompt with live validation and preview, editable
  allowed answers, exact-match scoring for open-ended tasks, and saved custom presets.
- Any Hugging Face repo id or local model folder as a custom model. A demo model
  (`batchtextreview ui --demo` or `BTR_DEMO=1`) for trying the app without a GPU.
- **Weights** (precision) choice: Auto, 4-bit, fp16 or fp32 (and bf16 on the command line, as
  `--precision`). Auto picks what the device supports.
- Background runs with live progress, rows per second, time remaining and Cancel. A run plan
  shows how many rows are already saved before you start, and **Discard saved outputs** deletes
  them for that configuration.
- A `NOTE` column flags texts cut to the max input characters and outputs that reached the max
  output tokens ("output may be cut off"). The flag is saved with each output, so it also shows
  for saved results.
- Results view with filters (wrong, invalid, errors, blank), search, a row detail panel showing
  the exact prompt, a label distribution, and JSON outputs expanded into one column per key.
- Evaluation: accuracy, macro-F1, Cohen's kappa, valid-answer rate and coverage, a clickable
  confusion matrix, per-class precision, recall and F1, and accuracy by slice. Flagged rows (for
  example `CONTESTED`) can be excluded from the headline numbers.
- History leaderboard with filters and charts, and a change-since-last-run figure on the
  evaluation tiles.
- Excel export, alongside CSV.
- One-click (History page) and command-line import of version 1 checkpoints and `evals.csv`.
- Configuration through environment variables: `BTR_RESULTS_DIR`, `BTR_DATA_DIR`, `BTR_DEMO`,
  `BTR_VERBOSE` and `BTR_NO_TRUSTSTORE`.
- A log file, `results/logs/batchtextreview.log` (1 MB, three old copies kept), written by the app
  and the command line. Model-load failures are logged in full. `batchtextreview -v` or
  `BTR_VERBOSE=1` adds debug details and prints them in the terminal too.
- Command-line options `run --sheet` (Excel sheet), `ui --demo`, `ui --results-dir` and
  `import-legacy FOLDER` (the version 1 folder to import from).
- Gold-labelled sample dataset (`examples/sample_feedback.csv`), a conda environment file for
  CUDA, a test suite, and CI on Ubuntu and Windows.

### Changed

- The app only accepts connections from this computer (`server.address = "localhost"`).
  Version 1 used Streamlit's default, every network interface. To share it on a trusted network:
  `batchtextreview ui -- --server.address 0.0.0.0`.
- Saved outputs and evaluation history are kept in one SQLite database
  (`results/batchtextreview.db`). This replaces the per-configuration parquet and JSON files and
  `evals.csv`. Writes are incremental: version 1 rewrote the whole checkpoint after every batch.
  The database records its schema version (3). A database from an earlier build is upgraded in
  place when opened, and one written by a newer version is refused rather than changed.
- Run ids hash the model's repo id instead of its display label, so model names can change
  without orphaning saved outputs. For a local model folder they also include a fingerprint of
  its model files (names, sizes, modification times), so saving new weights into the same folder
  starts a fresh run. Model labels no longer carry hints such as "(fastest)"; these are shown as
  badges, and imported history is mapped to the new names.
- Failed rows are recorded in new `STATUS` and `ERROR` columns, and `OUTPUT` stays empty. Failed,
  blank and not-run rows are excluded from scoring instead of being counted as wrong.
- `LABEL` is empty for rows that can't be scored. `[invalid]` is reserved for real outputs that
  don't resolve to an allowed answer.
- Answer matching ignores lead-ins such as "Answer:", is negation-aware, handles scores such as
  "4/5" and numbers such as "3.0", and never guesses between two labels.
- Evaluation history has one line per configuration, data, gold column and label set. Re-scoring
  updates the line instead of appending a duplicate.
- CSV downloads are UTF-8 with BOM, so Excel shows non-ASCII text correctly. Download names
  describe their contents. The command line's default results file is
  `<input>_<preset>_<model>_results.csv`, so runs of different models don't replace each other.
- The default folder is no longer hard-coded. It is `BTR_DATA_DIR`, then the last folder used.
- Output columns never overwrite existing ones. Running on an exported file adds `OUTPUT_2` and so
  on.

### Fixed

- Results disappeared on any interaction (a download click, a widget change). They now persist.
- A long run was killed by any widget interaction and could not be cancelled.
- A failed model switch left the previous model's name pointing at freed weights, so every later
  run with that model failed.
- Model-load errors escaped as raw tracebacks. They are now a clean message with a suggested fix
  (licence and `HF_TOKEN`, out of memory, missing package, outdated `transformers`, network).
- 4-bit loading was forced on CPU and macOS, where it crashes.
- Tokenizers without a chat template made every batch fail.
- The end-of-sequence token was used for padding even when the tokenizer has its own pad token.
- `<think>` blocks from reasoning models leaked into outputs.
- Missing gold values became the text "nan" and were scored. Numeric gold values read as "3.0"
  matched "3" only by chance.
- Texts reading "NA" or "None" were loaded as missing values. Excel exports with a BOM, or
  Windows-1252 encoding, broke the first column name or failed to load.
- A `{var}` value containing `{text}` was expanded once per row.
- Two browser sessions shared one model without a lock.
- The results folder depended silently on the current directory, and the run manifest was never
  updated.
- Exported cells starting with `=`, `+`, `-` or `@` could run as spreadsheet formulas.

## [1.0.0]

The original single-folder tool (`app.py` + `llm_batch.py`, "Corpus LLM Runner"): run a prompt
over the `TEXT` column of a CSV with a local 4-bit model, checkpoint outputs per configuration,
and score the outputs against a gold column.

