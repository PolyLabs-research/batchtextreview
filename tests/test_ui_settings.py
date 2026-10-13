"""Models & settings page: pure helpers plus AppTest coverage of the page."""

from __future__ import annotations

import sqlite3
import threading
import types
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from batchtextreview import models, presets
from batchtextreview.backends.base import DeviceInfo
from batchtextreview.backends.fake import FakeBackend
from batchtextreview.backends.manager import ModelManager
from batchtextreview.config import Settings
from batchtextreview.engine import RunConfig
from batchtextreview.jobs import JobRequest
from batchtextreview.store import Store
from batchtextreview.ui import nav, resources
from batchtextreview.ui.pages import settings as page

APP = Path(__file__).resolve().parents[1] / "app.py"
TIMEOUT = 30
GB = 1024**3

SENTIMENT = presets.get_preset("Sentiment")
YES_NO = presets.get_preset("Yes/No flag (set {var})")


@pytest.fixture(autouse=True)
def demo_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BTR_DEMO", "1")
    monkeypatch.setenv("BTR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    resources.get_model_manager().unload()
    st.cache_resource.clear()


def open_settings(**state: Any) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    at._page_hash = nav.page("settings")._script_hash
    return at.run()


def texts(at: AppTest, kind: str = "markdown") -> str:
    return "\n".join(str(e.value) for e in getattr(at, kind))


def every_text(at: AppTest) -> str:
    """Everything the page shows as text, tables included."""
    kinds = ("markdown", "caption", "code", "info", "warning", "error")
    parts = [texts(at, kind) for kind in kinds]
    parts += [df.value.to_csv() for df in [*at.dataframe, *at.table]]
    parts += [m.value for m in at.metric]
    return "\n".join(parts)


def table_with(at: AppTest, column: str):
    """The first table with ``column`` (read-only tables carry no key in AppTest)."""
    return next(df.value for df in [*at.dataframe, *at.table] if column in df.value.columns)


def no_exceptions(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def save_run(store, label: str, template: str, n: int, precision: str = "4bit") -> RunConfig:
    spec = models.find_model(label)
    assert spec is not None
    config = RunConfig.build(spec, template, 8, precision=precision)
    store.upsert_run(config)
    store.put_outputs(config.run_id, {f"{label}-{i}": "positive" for i in range(n)})
    return config


def runs_table(at: AppTest):
    (table,) = [df for df in at.dataframe if (df.key or "").startswith(page.K_RUNS_TABLE)]
    return table


def select_run(at: AppTest, run_id: str) -> None:
    """Select a saved-outputs row, as a click in the browser would."""
    table = runs_table(at)
    row = list(table.value["run_id"]).index(run_id)
    at.session_state[table.key] = {"selection": {"rows": [row], "columns": []}}


def delete_run_button(at: AppTest, run_id: str):
    return at.button(key=f"{page.K_DELETE_RUN}::{run_id}")


@pytest.fixture
def start_run(monkeypatch):
    """Starts a run that stays in progress until the test ends."""
    release = threading.Event()
    jobs = []

    def slow(prompt: str) -> str:
        release.wait(30)
        return "positive"

    monkeypatch.setattr(
        resources,
        "default_manager",
        lambda: ModelManager(lambda mid, p: FakeBackend(model_id=mid, responder=slow)),
    )

    def start():
        spec = models.find_model(models.DEMO_REPO)
        config = RunConfig.build(spec, SENTIMENT.template, 4)
        job = resources.get_job_manager().submit(JobRequest(config, ["a", "b"], batch_size=1))
        jobs.append(job)
        return job

    yield start
    release.set()
    for job in jobs:
        job.wait(30)


# ==========================================================================
# Pure helpers
# ==========================================================================


def test_device_badge_and_note():
    cuda = DeviceInfo("cuda", "NVIDIA RTX 4060", 8.0, bitsandbytes=True, cuda_version="12.4")
    assert page.device_badge(cuda)[0] == "GPU · 4-bit ready"
    assert page.device_note(cuda, demo=False) is None

    no_bnb = DeviceInfo("cuda", "NVIDIA RTX 4060", 8.0, bitsandbytes=False)
    assert page.device_badge(no_bnb)[2] == "orange"
    _, text, level = page.device_note(no_bnb, demo=False)
    assert "bitsandbytes" in text and level == "warning"

    cpu = DeviceInfo("cpu", "CPU (no CUDA GPU detected)", cuda_version="12.4")
    title, text, level = page.device_note(cpu, demo=False)
    assert title == "No NVIDIA GPU detected"
    assert "4-bit models need an NVIDIA GPU" in text
    assert "32-bit" in text
    assert "BTR_DEMO=1" in text and level == "warning"
    # The driver advice is kept out of the note (shown in an expander).
    assert "driver" not in text
    assert "check the NVIDIA driver" in page.driver_hint(cpu)  # CUDA build, no usable GPU
    assert "CUDA build of PyTorch" in page.driver_hint(DeviceInfo("cpu", "CPU"))
    assert page.driver_hint(DeviceInfo("none", "PyTorch not installed")) is None
    _, text, level = page.device_note(cpu, demo=True)
    assert "Demo mode is on" in text and level == "info"

    none = DeviceInfo("none", "PyTorch not installed")
    assert page.device_badge(none)[2] == "red"
    assert page.device_note(none, demo=False)[0] == "PyTorch isn't installed"
    assert "16-bit" in page.device_note(DeviceInfo("mps", "Apple GPU (MPS)"), demo=False)[1]


def test_catalogue_frame():
    specs = models.available_models()
    llama = "meta-llama/Llama-3.2-3B-Instruct"
    frame = page.catalogue_frame(specs, {llama: True}, token=False, loaded_repo=llama)
    assert len(frame) == len(specs)
    row = frame.set_index("Model").loc["Llama-3.2-3B-Instruct"]
    assert row["Repository"] == f"https://huggingface.co/{llama}"
    assert row["Notes"] == ["Loaded", "Gated · needs HF_TOKEN"]
    assert row["On disk"] == "Downloaded"
    other = frame.set_index("Model").loc["Qwen2.5-1.5B-Instruct"]
    assert other["Notes"] == ["Fastest"]
    assert other["On disk"] == "–"  # unknown
    missing = page.catalogue_frame(specs, {llama: False}, token=False).set_index("Model")
    assert missing.loc["Llama-3.2-3B-Instruct", "On disk"] == "≈6 GB download"
    with_token = page.catalogue_frame(specs, None, token=True)
    assert with_token.set_index("Model").loc["Llama-3.2-3B-Instruct", "Notes"] == ["Gated"]
    assert with_token.set_index("Model").loc["Qwen3.5-4B", "Notes"] == ["Experimental"]


def test_guess_task():
    candidates = presets.builtin_presets()
    assert page.guess_task(SENTIMENT.template, candidates) == "Sentiment"
    filled = YES_NO.template.replace("{var}", "Is it about parking?")
    assert page.guess_task(filled, candidates) == YES_NO.title
    assert page.guess_task("Something else entirely: {text}", candidates) is None


def test_runs_frame_prefers_history_task(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    config = save_run(store, "Qwen2.5-3B-Instruct", SENTIMENT.template, 3)
    edited = save_run(store, "Qwen2.5-3B-Instruct", "Rate this: {text}", 2)
    store.log_evaluation({"run_id": edited.run_id, "preset": "Intent", "accuracy": 0.5})
    frame = page.runs_frame(store.list_runs(), store.list_evaluations(), []).set_index("run_id")
    assert frame.loc[config.run_id, "Task"] == "Sentiment"  # matched by template
    assert frame.loc[edited.run_id, "Task"] == "Intent"  # from history
    assert frame.loc[config.run_id, "Saved outputs"] == 3
    assert frame.loc[config.run_id, "Weights"] == "4-bit"


def test_env_table_never_shows_the_token(tmp_path):
    settings = Settings(results_dir=tmp_path / "r", data_dir=tmp_path / "d")
    env = {"HF_TOKEN": "hf_supersecret", "BTR_DEMO": "1"}
    table = page.env_table(settings, env).set_index("Variable")
    assert table.loc["HF_TOKEN", "Current value"] == "Set (hidden)"
    assert "hf_supersecret" not in table.to_csv()
    assert table.loc["BTR_DEMO", "Current value"] == "1"
    # Paths aren't repeated here: the Results folder box shows the full one.
    assert str(tmp_path) not in table.to_csv()
    assert table.loc["BTR_RESULTS_DIR", "Current value"].startswith("Not set")
    assert page.env_table(settings, {}).loc[3, "Current value"] == "Not set"
    long_path = "C:\\Users\\someone\\Documents\\Projects\\batch text review\\results"
    custom = page.env_table(
        settings, {"BTR_RESULTS_DIR": long_path, "BTR_DATA_DIR": "D:\\data"}
    ).set_index("Variable")
    assert custom.loc["BTR_RESULTS_DIR", "Current value"] == "Set · see Results folder above"
    assert custom.loc["BTR_DATA_DIR", "Current value"] == "D:\\data"


# ==========================================================================
# Page
# ==========================================================================


@pytest.mark.ui
def test_renders_with_nothing_saved():
    at = open_settings()
    no_exceptions(at)
    shown = texts(at)
    assert "No model loaded" in shown
    assert "No saved outputs yet" in shown
    assert len(table_with(at, "Repository")) == len(models.MODELS)
    labels = [m.label for m in at.metric]
    assert labels == [
        "Saved output sets",
        "Saved outputs (unique texts)",
        "Evaluations",
        "Saved presets",
        "Database size",
    ]
    assert [m.value for m in at.metric][:4] == ["0", "0", "0", "0"]
    assert at.button(key="settings.unload").disabled
    env = table_with(at, "Variable")
    assert list(env["Variable"])[:3] == ["`BTR_RESULTS_DIR`", "`BTR_DATA_DIR`", "`BTR_DEMO`"]


@pytest.mark.ui
def test_token_value_is_never_rendered(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret_value")
    at = open_settings()
    no_exceptions(at)
    everything = every_text(at).replace("\\", "")  # unescape Markdown
    assert "hf_supersecret_value" not in everything
    assert "Set (hidden)" in everything


@pytest.mark.ui
def test_saved_outputs_select_and_delete():
    store = resources.get_store()
    keep = save_run(store, "Qwen2.5-7B-Instruct", SENTIMENT.template, 4)
    drop = save_run(store, "Qwen2.5-3B-Instruct", "Is this text about libraries?\n\n{text}", 6)
    at = open_settings()
    no_exceptions(at)
    table = runs_table(at).value
    assert set(table["Task"]) == {"Sentiment", "Custom prompt"}
    assert "Select a row" in texts(at, "caption")

    select_run(at, drop.run_id)
    at.run()
    no_exceptions(at)
    assert any("Is this text about libraries?" in c.value for c in at.code)
    # The configuration id is there for the curious, in a tooltip - not in the copy.
    assert drop.run_id not in texts(at, "caption")
    assert any(drop.run_id in (c.proto.help or "") for c in at.caption)

    delete_run_button(at, drop.run_id).click().run()
    no_exceptions(at)
    assert "Delete 6 saved outputs for Qwen2" in texts(at)
    confirm = at.button(key="settings.delete_run_confirm")
    assert confirm.label == "Delete 6 saved outputs"
    confirm.click().run()
    no_exceptions(at)
    assert list(store.list_runs()["run_id"]) == [keep.run_id]
    assert len(runs_table(at).value) == 1
    assert "Select a row" in texts(at, "caption")  # nothing selected any more


@pytest.mark.ui
def test_selection_follows_the_run_when_the_list_changes():
    """A run started elsewhere adds a row above the selected one: the card,
    and its delete button, stay on the configuration the user picked."""
    store = resources.get_store()
    picked = save_run(store, "Qwen2.5-3B-Instruct", SENTIMENT.template, 6)
    other = save_run(store, "Qwen2.5-7B-Instruct", SENTIMENT.template, 62)
    at = open_settings()
    assert list(runs_table(at).value["run_id"]) == [other.run_id, picked.run_id]
    select_run(at, picked.run_id)
    at.run()
    assert "Qwen2\\.5\\-3B\\-Instruct · Sentiment" in texts(at)

    later = (datetime.now() + timedelta(minutes=5)).isoformat(timespec="seconds")
    newcomer = save_run(store, "Phi-4-mini-Instruct", SENTIMENT.template, 9)
    with sqlite3.connect(store.path) as conn:  # make it the most recently used
        conn.execute("UPDATE runs SET updated_at = ? WHERE run_id = ?", (later, newcomer.run_id))
    conn.close()
    at.run()
    no_exceptions(at)
    ids = list(runs_table(at).value["run_id"])
    assert ids[0] == newcomer.run_id and ids.index(picked.run_id) == 2
    assert "Qwen2\\.5\\-3B\\-Instruct · Sentiment" in texts(at)
    delete_run_button(at, picked.run_id).click().run()
    no_exceptions(at)
    assert "Delete 6 saved outputs for Qwen2\\.5\\-3B\\-Instruct" in texts(at)
    at.button(key="settings.delete_run_confirm").click().run()
    assert set(store.list_runs()["run_id"]) == {other.run_id, newcomer.run_id}


@pytest.mark.ui
def test_actions_are_disabled_while_a_run_is_in_progress(start_run):
    store = resources.get_store()
    config = save_run(store, "Qwen2.5-3B-Instruct", SENTIMENT.template, 6)
    job = start_run()
    for _ in range(100):  # wait until the job holds the model
        if resources.get_model_manager().busy:
            break
        job.wait(0.05)
    at = open_settings()
    select_run(at, config.run_id)
    at.run()
    no_exceptions(at)
    unload = at.button(key="settings.unload")
    assert unload.disabled and unload.help == "Wait for the current run to finish."
    assert "In use" in texts(at)
    delete = delete_run_button(at, config.run_id)
    assert delete.disabled and delete.help == "Wait for the current run to finish."


@pytest.mark.ui
def test_unreadable_database_shows_one_error(monkeypatch):
    def broken(self, *args, **kwargs):
        raise RuntimeError("file is not a database")

    monkeypatch.setattr(Store, "list_runs", broken)
    at = open_settings()
    no_exceptions(at)
    assert [e.value.split("\n\n")[0] for e in at.error] == ["Couldn't read the results database."]
    assert "not open in another program" in at.error[0].value
    assert not at.warning
    assert "file is not a database" in texts(at, "code")  # under Technical details
    assert "Unavailable until the results database can be read." in texts(at, "caption")
    assert [m.value for m in at.metric] == ["–"] * 5
    # The rest of the page still works.
    assert len(table_with(at, "Repository")) == len(models.MODELS)


@pytest.mark.ui
def test_unload_model():
    manager = resources.get_model_manager()
    with manager.acquire(models.DEMO_REPO, "none"):
        pass
    at = open_settings()
    no_exceptions(at)
    assert "Demo \\(fake outputs\\)" in texts(at)
    unload = at.button(key="settings.unload")
    assert not unload.disabled
    unload.click().run()
    no_exceptions(at)
    assert manager.loaded is None
    assert "No model loaded" in texts(at)


@pytest.mark.ui
def test_live_gpu_memory_on_cuda(monkeypatch):
    gpu = DeviceInfo(
        "cuda",
        "NVIDIA GeForce RTX 4060 Laptop GPU",
        8.0,
        bitsandbytes=True,
        torch_version="2.5.1",
        cuda_version="12.4",
    )
    monkeypatch.setattr(resources, "device_info", lambda: gpu)
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(mem_get_info=lambda: (int(5.2 * GB), 8 * GB))
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    at = open_settings()
    no_exceptions(at)
    assert "GPU · 4-bit ready" in texts(at)
    assert "4-bit (NF4)" in texts(at)
    assert at.get("progress")[0].proto.text == "5.2 GB free of 8.0 GB"
    assert not at.warning and not at.info


# ---- final review fixes ----------------------------------------------------------


def test_download_range_follows_the_catalogue():
    specs = models.available_models()
    sizes = [resources.download_size_gb(s.params_b) for s in specs if s.params_b]
    text = page.download_range(specs)
    assert text == f"about {min(sizes)}-{max(sizes)} GB"
    assert text.endswith("-18 GB")  # Qwen3.5-9B, the largest built-in model
    assert page.download_range([]) == "several GB"


def test_env_table_lists_verbose_logging_and_a_saved_login(tmp_path):
    settings = Settings(results_dir=tmp_path / "r", data_dir=tmp_path / "d")
    table = page.env_table(settings, {}).set_index("Variable")
    assert "BTR_VERBOSE" in table.index
    assert "batchtextreview -v ui" in table.loc["BTR_VERBOSE", "What it does"]
    login = page.env_table(settings, {}, login_token=True).set_index("Variable")
    assert login.loc["HF_TOKEN", "Current value"] == (
        "Not set · the token saved by hf auth login is used"
    )
    env = page.env_table(settings, {"HF_TOKEN": "hf_x"}, login_token=True).set_index("Variable")
    assert env.loc["HF_TOKEN", "Current value"] == "Set (hidden)"


def test_catalogue_frame_in_offline_mode():
    specs = models.available_models()
    llama = "meta-llama/Llama-3.2-3B-Instruct"
    frame = page.catalogue_frame(specs, {llama: False}, token=True, offline=True)
    assert frame.set_index("Model").loc["Llama-3.2-3B-Instruct", "On disk"] == "Not downloaded"


def test_runs_frame_tells_same_named_folders_apart(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    for folder in ("/work/a/tinymodel", "/work/b/tinymodel"):
        spec = models.ModelSpec(label="tinymodel", repo=folder, builtin=False)
        config = RunConfig.build(spec, SENTIMENT.template, 8, precision="fp32")
        store.upsert_run(config)
        store.put_outputs(config.run_id, {"k": "positive"})
    frame = page.runs_frame(store.list_runs())
    assert sorted(frame["Model"]) == ["a/tinymodel", "b/tinymodel"]


@pytest.mark.ui
def test_storage_shows_the_log_file():
    at = open_settings()
    no_exceptions(at)
    assert "Log file" in texts(at)
    log = str(Path(resources.get_settings().results_dir) / "logs" / "batchtextreview.log")
    assert any(code.value == log for code in at.code)


@pytest.mark.ui
def test_loaded_model_card_says_memory_on_a_cpu(monkeypatch):
    manager = ModelManager(factory=lambda model_id, precision: FakeBackend(model_id, precision))
    monkeypatch.setattr(resources, "get_model_manager", lambda: manager)
    monkeypatch.setattr(
        resources, "device_info", lambda: DeviceInfo("cpu", "CPU", torch_version="2")
    )
    with manager.acquire("/work/a/tinymodel", "fp32"):
        pass
    at = open_settings()
    no_exceptions(at)
    captions = texts(at, "caption")
    assert "Unload it to free memory for other programs." in captions
    assert "GPU memory" not in captions
