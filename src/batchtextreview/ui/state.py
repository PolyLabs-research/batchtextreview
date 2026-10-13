"""Session-state keys and the per-run context the UI attaches to a job.

Nothing in this module imports Streamlit at module level, so the dataclass
can be used from the job worker thread and from tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from batchtextreview.ui.format import truncate

# ---- st.session_state keys -------------------------------------------------
ACTIVE_JOB = "btr.active_job"
"""Job id the Results page shows."""
SESSION_JOBS = "btr.session_jobs"
"""Job ids started from this browser session, newest first."""
LOADED_TABLE = "btr.table"
"""The LoadedTable currently selected on the Run page (or absent)."""
DATA_SOURCE = "btr.data_source"
"""How the table was loaded: "upload", "path" or "sample"."""

CONTEXT_KEY = "run"
"""Key of the RunContext inside ``JobRequest.context``."""
RETRY_OF = "retry_of"
"""``JobRequest.context`` key naming the job a retry resubmitted."""


@dataclass
class RunContext:
    """Everything the Results / Evaluation views need besides the RunResult."""

    name: str
    """Human run name, e.g. "Sentiment · feedback.csv · 14:03" (:func:`run_name`)."""
    dataset_name: str
    dataset_fingerprint: str
    """``data.frame_fingerprint`` of the *selected* rows."""
    frame: pd.DataFrame
    """Selected rows, index reset, all original columns."""
    text_column: str
    preset_key: str
    preset_title: str
    template_raw: str
    """Template as edited, before ``{var}`` substitution."""
    template_edited: bool
    var: str | None
    labels: list[str] | None
    output_kind: str
    selection: str
    """Human description, e.g. "First 20 of 60 rows"."""
    id_column: str | None = None
    gold_column: str | None = None
    group_columns: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def scorable_task(self) -> bool:
        return bool(self.labels)


def run_name(task_title: str, dataset_name: str, when: datetime | None = None) -> str:
    """ "Task · dataset · HH:MM" - short enough for a selectbox.

    Task and dataset come first: they are what people look for in a list of
    runs (the model is often the same for all of them, and is shown beside
    the name wherever it matters)."""
    when = when or datetime.now()
    return f"{task_title} · {truncate(dataset_name, 36)} · {when:%H:%M}"


# ---- Run page setup, to tell whether it changed during a run --------------
RUN_SETUP = "btr.run_setup"
"""Signature of what the Run page is set up to run right now (written by the
Run page on every full run)."""
JOB_SETUPS = "btr.job_setups"
"""job id -> the Run page's setup signature when that job was submitted: the
Run page's primary button opens a finished run (rather than starting an
identical one) only when its setup equals the page's."""
NOT_SET_UP_HERE = "not set up on the Run page"
"""First item of the setup recorded for a run the Run page didn't set up
(a retry of a run from another tab): it never equals the page's setup."""
