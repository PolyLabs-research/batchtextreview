"""Tests for batchtextreview.models (the model catalogue)."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from batchtextreview.models import (
    DEMO_MODEL,
    DEMO_REPO,
    LEGACY_LABEL_TO_REPO,
    MODELS,
    ModelSpec,
    available_models,
    custom_model,
    find_model,
    model_revision,
    resolve_model,
)

# Version 1's llm_batch.MODELS, verbatim. Old run ids hashed these labels, so
# the legacy map must reproduce it exactly or old checkpoints won't import.
V1_MODELS = {
    "Granite-4.0-350M (tiny)": "ibm-granite/granite-4.0-350m",
    "Granite-4.0-1B": "ibm-granite/granite-4.0-1b",
    "Qwen2.5-1.5B-Instruct (fastest)": "Qwen/Qwen2.5-1.5B-Instruct",
    "Granite-4.0-Micro 3B": "ibm-granite/granite-4.0-micro",
    "Qwen2.5-3B-Instruct (faster)": "Qwen/Qwen2.5-3B-Instruct",
    "Llama-3.2-3B-Instruct (gated)": "meta-llama/Llama-3.2-3B-Instruct",
    "Phi-3.5-mini-Instruct": "microsoft/Phi-3.5-mini-instruct",
    "Phi-4-mini-Instruct (3.8B)": "microsoft/Phi-4-mini-instruct",
    "Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen3.5-4B (experimental)": "Qwen/Qwen3.5-4B",
    "Qwen2.5-7B-Instruct (best quality)": "Qwen/Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct (gated)": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3.5-9B (experimental)": "Qwen/Qwen3.5-9B",
}


def _by_repo(repo: str) -> ModelSpec:
    return next(s for s in MODELS if s.repo == repo)


# --------------------------------------------------------------------------
# Registry integrity
# --------------------------------------------------------------------------


def test_repos_are_unique() -> None:
    repos = [s.repo for s in MODELS]
    assert len(repos) == len(set(repos))


def test_labels_are_unique() -> None:
    labels = [s.label for s in (*MODELS, DEMO_MODEL)]
    assert len(labels) == len(set(labels))


def test_labels_never_collide_with_another_models_repo() -> None:
    # find_model accepts either; a label equal to a different model's repo
    # would make the lookup ambiguous.
    repos = {s.repo: s for s in MODELS}
    for spec in MODELS:
        assert repos.get(spec.label, spec) is spec


def test_registry_entries_are_well_formed() -> None:
    for spec in MODELS:
        assert spec.builtin is True
        assert not spec.is_demo
        assert spec.label.strip() == spec.label and spec.label
        assert spec.repo.count("/") == 1, spec.repo
        assert spec.params_b is not None and 0 < spec.params_b <= 9.5, spec


def test_legacy_map_matches_version_1_exactly() -> None:
    assert LEGACY_LABEL_TO_REPO == V1_MODELS
    # Order matters to nobody, but keep it as a sanity check on the copy.
    assert list(LEGACY_LABEL_TO_REPO) == list(V1_MODELS)


def test_every_legacy_repo_is_a_registry_repo() -> None:
    repos = {s.repo for s in MODELS}
    for label, repo in LEGACY_LABEL_TO_REPO.items():
        assert repo in repos, label


def test_every_registry_repo_has_a_legacy_label() -> None:
    legacy_repos = set(LEGACY_LABEL_TO_REPO.values())
    for spec in MODELS:
        assert spec.repo in legacy_repos, spec.repo


def test_legacy_map_is_one_to_one() -> None:
    assert len(set(LEGACY_LABEL_TO_REPO.values())) == len(LEGACY_LABEL_TO_REPO)


def test_badges_agree_with_version_1_labels() -> None:
    for label, repo in LEGACY_LABEL_TO_REPO.items():
        spec = _by_repo(repo)
        assert spec.gated == ("(gated)" in label), label
        assert spec.experimental == ("(experimental)" in label), label
        for tier in ("tiny", "fastest", "faster", "best quality"):
            if f"({tier})" in label:
                assert spec.tier == tier, label


def test_models_are_listed_smallest_first() -> None:
    sizes = [s.params_b for s in MODELS]
    assert sizes == sorted(sizes)  # type: ignore[type-var]


def test_demo_model() -> None:
    assert DEMO_MODEL.repo == DEMO_REPO == "demo://fake"
    assert DEMO_MODEL.is_demo
    assert not DEMO_MODEL.is_local
    assert DEMO_MODEL.hf_url is None
    assert DEMO_MODEL.size_label == ""
    assert DEMO_MODEL not in MODELS


def test_available_models() -> None:
    assert available_models() == list(MODELS)
    with_demo = available_models(include_demo=True)
    assert with_demo[0] == DEMO_MODEL
    assert with_demo[1:] == list(MODELS)
    # A fresh list each call: callers may mutate it.
    available_models().clear()
    assert len(available_models()) == len(MODELS)


def test_model_spec_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        MODELS[0].label = "renamed"  # type: ignore[misc]


# --------------------------------------------------------------------------
# find_model / resolve_model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", MODELS, ids=lambda s: s.repo)
def test_find_model_by_label_and_repo(spec: ModelSpec) -> None:
    assert find_model(spec.label) is spec
    assert find_model(spec.repo) is spec


@pytest.mark.parametrize("legacy", list(V1_MODELS))
def test_find_model_by_legacy_label(legacy: str) -> None:
    spec = find_model(legacy)
    assert spec is not None
    assert spec.repo == V1_MODELS[legacy]
    assert spec.builtin


def test_find_model_trims_whitespace() -> None:
    assert find_model("  Qwen/Qwen3-8B\n") is _by_repo("Qwen/Qwen3-8B")
    assert find_model(" Qwen3-8B ") is _by_repo("Qwen/Qwen3-8B")
    assert find_model(" Granite-4.0-350M (tiny) ") is _by_repo("ibm-granite/granite-4.0-350m")


def test_find_model_demo() -> None:
    assert find_model(DEMO_REPO) is DEMO_MODEL
    assert find_model(DEMO_MODEL.label) is DEMO_MODEL


@pytest.mark.parametrize("name", ["", "   ", "Qwen/Nope-1B", "qwen/qwen3-8b", "Qwen3-8B (gated)"])
def test_find_model_unknown(name: str) -> None:
    assert find_model(name) is None


def test_resolve_model_prefers_registry() -> None:
    assert resolve_model("Qwen2.5-7B-Instruct (best quality)") is _by_repo(
        "Qwen/Qwen2.5-7B-Instruct"
    )


def test_resolve_model_falls_back_to_custom() -> None:
    spec = resolve_model("someone/SomeModel-0.5B")
    assert spec.repo == "someone/SomeModel-0.5B"
    assert spec.builtin is False


def test_resolve_model_blank_raises() -> None:
    with pytest.raises(ValueError):
        resolve_model("  ")


# --------------------------------------------------------------------------
# custom_model
# --------------------------------------------------------------------------


def test_custom_model_repo_id() -> None:
    spec = custom_model("  HuggingFaceTB/SmolLM2-360M-Instruct  ")
    assert spec.repo == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert spec.label == "SmolLM2-360M-Instruct"
    assert spec.builtin is False
    assert spec.params_b is None and spec.size_label == ""
    assert not spec.gated and not spec.experimental
    assert not spec.is_local and not spec.is_demo
    assert spec.hf_url == "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct"


def test_custom_model_bare_name() -> None:
    spec = custom_model("gpt2")
    assert spec.repo == "gpt2"
    assert spec.label == "gpt2"


def test_custom_model_trailing_slash_is_dropped() -> None:
    spec = custom_model("org/name/")
    assert spec.label == "name"
    assert spec.repo == "org/name"  # the Hub rejects "org/name/"
    assert spec.hf_url == "https://huggingface.co/org/name"


def test_custom_model_trailing_slash_on_a_builtin_finds_it() -> None:
    # Same cache key and gated/experimental flags as picking it from the list.
    assert custom_model("Qwen/Qwen3-8B/") is _by_repo("Qwen/Qwen3-8B")
    assert custom_model("meta-llama/Llama-3.1-8B-Instruct/").gated


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/Qwen/Qwen3-8B",
        "https://huggingface.co/Qwen/Qwen3-8B/",
        "http://huggingface.co/Qwen/Qwen3-8B",
        "huggingface.co/Qwen/Qwen3-8B",
        "https://www.huggingface.co/Qwen/Qwen3-8B",
        "https://hf.co/Qwen/Qwen3-8B",
        "HTTPS://HuggingFace.co/Qwen/Qwen3-8B",
        "https://huggingface.co/Qwen/Qwen3-8B/tree/main",
        "https://huggingface.co/Qwen/Qwen3-8B?library=transformers",
        "https://huggingface.co/Qwen/Qwen3-8B#model-card",
    ],
)
def test_custom_model_pasted_page_url(url: str) -> None:
    assert custom_model(url) is _by_repo("Qwen/Qwen3-8B")


def test_custom_model_pasted_url_of_an_unlisted_model() -> None:
    spec = custom_model("https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/")
    assert spec.repo == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert spec.label == "SmolLM2-360M-Instruct"
    assert spec.hf_url == "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct"


@pytest.mark.parametrize("url", ["https://huggingface.co/", "https://huggingface.co/?q=x"])
def test_custom_model_url_without_a_model_raises(url: str) -> None:
    with pytest.raises(ValueError, match="no model"):
        custom_model(url)


@pytest.mark.parametrize(
    "bad",
    [
        "Qwen 3 8B",
        "Qwen/Qwen3 8B",
        "org/-name",
        "org/name-",
        ".org/name",
        "org/na--me",
        "org/na..me",
        "org:name",
        "org/" + "x" * 100,
    ],
)
def test_custom_model_invalid_repo_id_raises(bad: str) -> None:
    with pytest.raises(ValueError):
        custom_model(bad)


@pytest.mark.parametrize("ok", ["gpt2", "org/name", "org/name.v2", "a_b/c-d.e", "Org9/Model_1.5"])
def test_custom_model_valid_repo_ids_pass(ok: str) -> None:
    assert custom_model(ok).repo == ok


def test_custom_model_known_repo_returns_registry_entry() -> None:
    assert custom_model("Qwen/Qwen3-8B") is _by_repo("Qwen/Qwen3-8B")
    assert custom_model("Qwen3-8B") is _by_repo("Qwen/Qwen3-8B")
    assert custom_model(DEMO_REPO) is DEMO_MODEL


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_custom_model_blank_raises(blank: str) -> None:
    with pytest.raises(ValueError, match="repo id or a local model folder"):
        custom_model(blank)


def test_custom_model_local_directory(tmp_path: Path) -> None:
    folder = tmp_path / "my-finetune"
    folder.mkdir()
    spec = custom_model(f"  {folder}  ")
    assert spec.repo == str(folder.resolve())
    assert spec.label == "my-finetune"
    assert spec.builtin is False
    assert spec.is_local
    assert spec.hf_url is None


def test_custom_model_relative_directory_is_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models" / "tiny").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    spec = custom_model("models/tiny/")
    assert spec.repo == str((tmp_path / "models" / "tiny").resolve())
    assert Path(spec.repo).is_absolute()
    assert spec.label == "tiny"
    # Still local after the working directory changes.
    monkeypatch.chdir(tmp_path / "models")
    assert spec.is_local


def test_custom_model_current_directory_gets_a_real_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "checkpoint-500"
    folder.mkdir()
    monkeypatch.chdir(folder)
    spec = custom_model(".")
    assert spec.repo == str(folder.resolve())
    assert spec.label == "checkpoint-500"


def test_custom_model_expands_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    (home / "llm").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows
    spec = custom_model("~/llm")
    assert spec.repo == str((home / "llm").resolve())
    assert spec.is_local


def test_custom_model_missing_folder_raises(tmp_path: Path) -> None:
    # Reported at input time, not after Run as "model not found".
    missing = tmp_path / "not-here"
    with pytest.raises(ValueError, match="Folder not found"):
        custom_model(str(missing))


@pytest.mark.parametrize(
    "path",
    [
        r"C:\models\my-model",
        "C:/models/my-model",
        r"models\my-model",
        r"\\server\share\model",
        "/no/such/dir/model",
        "./no-such-model",
        "../no-such-model",
        "~/no-such-model-btr-test",
        "~nosuchuser-btr-test/model",
        "models/no/such-model",
    ],
)
def test_custom_model_missing_path_like_values_raise(
    path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    with pytest.raises(ValueError, match="Folder not found"):
        custom_model(path)


def test_custom_model_other_demo_ids_pass_through() -> None:
    spec = custom_model("demo://other")
    assert spec.is_demo
    assert spec.repo == "demo://other"


# --------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params_b", "expected"),
    [
        (None, ""),
        (0.35, "350M"),
        (0.5, "500M"),
        (0.135, "135M"),
        (1.0, "1B"),
        (1.5, "1.5B"),
        (3.0, "3B"),
        (3.8, "3.8B"),
        (7.6, "7.6B"),
        (8.2, "8.2B"),
        (70, "70B"),
    ],
)
def test_size_label(params_b: float | None, expected: str) -> None:
    assert ModelSpec("x", "org/x", params_b).size_label == expected


def test_builtin_size_labels() -> None:
    labels = {s.repo: s.size_label for s in MODELS}
    assert labels["ibm-granite/granite-4.0-350m"] == "350M"
    assert labels["ibm-granite/granite-4.0-1b"] == "1B"
    assert labels["Qwen/Qwen2.5-7B-Instruct"] == "7.6B"
    assert all(labels.values())


def test_hf_url_for_every_builtin() -> None:
    for spec in MODELS:
        assert spec.hf_url == f"https://huggingface.co/{spec.repo}"


def test_hf_url_none_for_local_directory(tmp_path: Path) -> None:
    assert ModelSpec("local", str(tmp_path), builtin=False).hf_url is None


def test_is_local_false_for_file_and_repo(tmp_path: Path) -> None:
    file = tmp_path / "weights.bin"
    file.write_bytes(b"")
    assert not ModelSpec("f", str(file)).is_local
    assert not ModelSpec("r", "org/definitely-not-a-local-folder").is_local


def test_is_demo_only_for_demo_scheme() -> None:
    assert ModelSpec("d", "demo://other").is_demo
    assert not ModelSpec("d", "org/demo").is_demo


# --------------------------------------------------------------------------
# model_revision
# --------------------------------------------------------------------------

T0 = 1_700_000_000_000_000_000


def _write(path: Path, text: str, mtime_ns: int = T0) -> None:
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def _model_folder(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja"):
        _write(folder / name, f"{name} v1")
    return folder


def test_model_revision_is_stable_for_an_untouched_folder(tmp_path: Path) -> None:
    folder = _model_folder(tmp_path / "m")
    revision = model_revision(str(folder))
    assert len(revision) == 12
    int(revision, 16)
    assert model_revision(str(folder)) == revision
    assert model_revision(str(folder) + os.sep) == revision


@pytest.mark.parametrize(
    "name", ["model.safetensors", "config.json", "tokenizer.json", "chat_template.jinja"]
)
def test_model_revision_changes_when_a_model_file_is_saved_again(tmp_path: Path, name: str) -> None:
    """Regression: a fine-tune saved again into the same folder kept the
    old model's identity. Same size, newer time: retrained weights."""
    folder = _model_folder(tmp_path / "m")
    before = model_revision(str(folder))
    # A second later: NTFS keeps 100 ns ticks, so a 1 ns step would vanish.
    _write(folder / name, f"{name} v2", mtime_ns=T0 + 1_000_000_000)
    assert model_revision(str(folder)) != before


def test_model_revision_changes_when_weight_files_are_added(tmp_path: Path) -> None:
    folder = _model_folder(tmp_path / "m")
    before = model_revision(str(folder))
    _write(folder / "model-00002-of-00002.safetensors", "more")
    assert model_revision(str(folder)) != before


def test_model_revision_ignores_docs_hidden_files_and_subfolders(tmp_path: Path) -> None:
    folder = _model_folder(tmp_path / "m")
    before = model_revision(str(folder))
    _write(folder / "README.md", "notes", mtime_ns=T0 + 5)
    _write(folder / ".gitattributes", "*.safetensors lfs", mtime_ns=T0 + 5)
    _write(folder / "card.png", "img", mtime_ns=T0 + 5)
    (folder / "checkpoint-500").mkdir()
    _write(folder / "checkpoint-500" / "model.safetensors", "old", mtime_ns=T0 + 5)
    assert model_revision(str(folder)) == before


def test_model_revision_is_empty_for_hub_ids_demo_and_missing_folders(tmp_path: Path) -> None:
    assert model_revision("Qwen/Qwen2.5-3B-Instruct") == ""
    assert model_revision(DEMO_REPO) == ""
    assert model_revision(str(tmp_path / "missing")) == ""
    file = tmp_path / "weights.bin"
    file.write_bytes(b"")
    assert model_revision(str(file)) == ""
