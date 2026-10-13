"""Packaging and repository hygiene: what the builds contain, what git ignores,
and that the documentation's links resolve.

The build tests need hatchling (in the "dev" extra) and skip without it.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote

import pytest

from batchtextreview import cli, models, presets

if sys.version_info >= (3, 11):
    import tomllib
else:  # pytest itself depends on tomli before 3.11
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_CONFIG = REPO_ROOT / ".streamlit" / "config.toml"
# Where the wheel puts it: next to the bundled entry point, ui/streamlit_app.py.
WHEEL_STREAMLIT_CONFIG = "batchtextreview/ui/.streamlit/config.toml"
SAMPLE = REPO_ROOT / "examples" / "sample_feedback.csv"


def _toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def hatch_targets() -> dict[str, Any]:
    return _toml(REPO_ROOT / "pyproject.toml")["tool"]["hatch"]["build"]["targets"]


# --------------------------------------------------------------------------
# Streamlit config
# --------------------------------------------------------------------------


def test_streamlit_listens_on_this_computer_only() -> None:
    # Streamlit's default is every network interface, and the app lists and
    # reads files on this disk for whoever can reach the port.
    assert _toml(STREAMLIT_CONFIG)["server"]["address"] == "localhost"


def test_wheel_ships_the_streamlit_config_next_to_the_bundled_entry_point(
    hatch_targets: dict[str, Any],
) -> None:
    # A regular install launches ui/streamlit_app.py, and Streamlit reads the
    # .streamlit/config.toml next to the script it runs.
    bundled = Path(cli.__file__).resolve().parent / "ui" / "streamlit_app.py"
    assert bundled.is_file()
    force_include = hatch_targets["wheel"]["force-include"]
    assert force_include[".streamlit/config.toml"] == WHEEL_STREAMLIT_CONFIG
    assert PurePosixPath(WHEEL_STREAMLIT_CONFIG).parent.parent == PurePosixPath(
        "batchtextreview", "ui"
    )
    # A wheel built from the sdist needs its source too.
    assert "/.streamlit/config.toml" in hatch_targets["sdist"]["include"]


def test_sdist_is_an_allow_list_of_existing_paths(hatch_targets: dict[str, Any]) -> None:
    include = hatch_targets["sdist"]["include"]
    for entry in include:
        assert entry.startswith("/") and not set(entry) & set("*?["), entry
        assert (REPO_ROOT / entry.lstrip("/")).exists(), f"{entry} is not in the checkout"


# --------------------------------------------------------------------------
# Lint reproducibility
# --------------------------------------------------------------------------


def test_ruff_is_pinned_exactly() -> None:
    # ruff's rules and formatter change between releases: CI and every
    # checkout must agree on one.
    dev = _toml(REPO_ROOT / "pyproject.toml")["project"]["optional-dependencies"]["dev"]
    [ruff] = [req for req in dev if re.match(r"ruff\b", req)]
    assert re.fullmatch(r"ruff==\d+\.\d+\.\d+", ruff.replace(" ", "")), ruff


# --------------------------------------------------------------------------
# .gitignore
# --------------------------------------------------------------------------


def _git_ignored(path: str, tmp_path: Path) -> bool:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git isn't installed")
    proc = subprocess.run(
        [
            git,
            # Only the repository's rules, not this machine's global excludes.
            "-c",
            f"core.excludesFile={tmp_path / 'no-global-excludes'}",
            "check-ignore",
            "--no-index",
            "--quiet",
            path,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        pytest.skip(f"not a git checkout: {proc.stderr.strip()}")
    return proc.returncode == 0


def _cli_default_output() -> str:
    """Where the README's example run writes its results."""
    preset = presets.get_preset("Sentiment")
    model = models.find_model("Qwen2.5-1.5B-Instruct")
    assert preset is not None and model is not None
    return cli.default_output_path(SAMPLE, preset, model).relative_to(REPO_ROOT).as_posix()


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        pytest.param(None, id="cli-default-output"),
        "customer_tickets_sentiment_results.xlsx",
        "results/batchtextreview.db",
    ],
)
def test_git_ignores_secrets_and_outputs(path: str | None, tmp_path: Path) -> None:
    assert _git_ignored(path or _cli_default_output(), tmp_path)


@pytest.mark.parametrize(
    "path",
    [
        "examples/sample_feedback.csv",
        "src/batchtextreview/results.py",
        "tests/test_results.py",
        ".streamlit/config.toml",
    ],
)
def test_git_keeps_project_files(path: str, tmp_path: Path) -> None:
    assert not _git_ignored(path, tmp_path)


# --------------------------------------------------------------------------
# Builds
# --------------------------------------------------------------------------

# Enough of the project to build it, without whatever else is in the checkout.
_PROJECT_FILES = (
    "pyproject.toml",
    ".gitignore",
    "README.md",
    "app.py",
    ".streamlit/config.toml",
    "examples/sample_feedback.csv",
)
_STRAY_FILES = (".env", "customer_tickets.xlsx", "examples/sample_feedback_sentiment_results.csv")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    pytest.importorskip("hatchling")
    root = tmp_path / "project"
    for name in _PROJECT_FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / name, root / name)
    shutil.copytree(
        REPO_ROOT / "src" / "batchtextreview",
        root / "src" / "batchtextreview",
        ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]", ".streamlit"),
    )
    for name in _STRAY_FILES:
        (root / name).write_text("HF_TOKEN=hf_not_a_real_token\n", encoding="utf-8")
    return root


def _build(target: str, root: Path, out: Path) -> Path:
    from hatchling.builders.sdist import SdistBuilder
    from hatchling.builders.wheel import WheelBuilder

    builder = SdistBuilder(str(root)) if target == "sdist" else WheelBuilder(str(root))
    return Path(next(iter(builder.build(directory=str(out), versions=["standard"]))))


def _sdist_members(sdist: Path) -> set[str]:
    with tarfile.open(sdist) as archive:
        # "batchtextreview-2.0.0/src/..." -> "src/..."
        return {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}


def test_sdist_leaves_out_stray_files(project: Path, tmp_path: Path) -> None:
    members = _sdist_members(_build("sdist", project, tmp_path / "dist"))
    assert members.isdisjoint(_STRAY_FILES)
    assert {
        "pyproject.toml",
        "README.md",
        "app.py",
        ".streamlit/config.toml",
        "examples/sample_feedback.csv",
        "src/batchtextreview/cli.py",
        "src/batchtextreview/ui/streamlit_app.py",
    } <= members


def test_wheel_built_from_the_sdist_has_the_streamlit_config(project: Path, tmp_path: Path) -> None:
    sdist = _build("sdist", project, tmp_path / "dist")
    unpacked = tmp_path / "unpacked"
    with tarfile.open(sdist) as archive:
        if hasattr(tarfile, "data_filter"):
            archive.extractall(unpacked, filter="data")
        else:  # Python 3.10 before 3.10.12; our own archive
            archive.extractall(unpacked)
    [source] = unpacked.iterdir()
    wheel = _build("wheel", source, tmp_path / "wheel")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert "batchtextreview/ui/streamlit_app.py" in names
        assert archive.read(WHEEL_STREAMLIT_CONFIG) == STREAMLIT_CONFIG.read_bytes()
    assert not any(name.startswith(("src/", ".env")) for name in names)


# --------------------------------------------------------------------------
# Documentation links (README.md is also the package's long description)
# --------------------------------------------------------------------------

_FENCED_BLOCK = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)
_CODE_SPAN = re.compile(r"`[^`\n]*`")
_INLINE_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", re.MULTILINE)
_HTML_LINK = re.compile(r"""\b(?:src|href)\s*=\s*["']([^"']+)["']""")
_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


def _relative_links(markdown: str) -> list[str]:
    text = _CODE_SPAN.sub("", _FENCED_BLOCK.sub("", markdown))
    links = [
        *_INLINE_LINK.findall(text),
        *_REFERENCE_LINK.findall(text),
        *_HTML_LINK.findall(text),
    ]
    return [link for link in links if not _SCHEME.match(link) and not link.startswith("#")]


def test_link_finder() -> None:
    markdown = (
        "![Shot](docs/shot.png) [Guide](docs/GUIDE.md#setup) [Web](https://example.com)\n"
        "[Top](#top) <img src='docs/logo.svg'> `[code](not/a/link.md)`\n"
        "[ref]: docs/REF.md\n```\n[fenced](not/a/link.md)\n```\n"
    )
    assert _relative_links(markdown) == [
        "docs/shot.png",
        "docs/GUIDE.md#setup",
        "docs/REF.md",
        "docs/logo.svg",
    ]


@pytest.mark.parametrize(
    "document",
    ["README.md", "CHANGELOG.md", *sorted(p.name for p in (REPO_ROOT / "docs").glob("*.md"))],
)
def test_documentation_links_resolve(document: str) -> None:
    path = REPO_ROOT / document
    if not path.is_file():
        path = REPO_ROOT / "docs" / document
    missing = [
        link
        for link in _relative_links(path.read_text(encoding="utf-8"))
        if not (path.parent / unquote(link.split("#", 1)[0])).exists()
    ]
    assert not missing, f"{path.relative_to(REPO_ROOT)} links to missing files: {missing}"


# --------------------------------------------------------------------------
# Documentation content: what the docs say matches what the code does
# --------------------------------------------------------------------------

README = REPO_ROOT / "README.md"
MIGRATING = REPO_ROOT / "docs" / "MIGRATING.md"
_DOCS_WITH_COMMANDS = (README, MIGRATING, REPO_ROOT / "CHANGELOG.md")
_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
_PYTHON_FENCE = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _section(markdown: str, heading: str) -> str:
    """The text under ``## heading``, up to the next ``## `` heading."""
    start = markdown.index(f"\n## {heading}\n")
    end = markdown.find("\n## ", start + 1)
    return markdown[start : end if end != -1 else None]


def _subcommands() -> dict[str, argparse.ArgumentParser]:
    parser = cli.build_parser()
    [commands] = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    return dict(commands.choices)


def _option_spellings(parser: argparse.ArgumentParser) -> list[tuple[str, ...]]:
    """Each option's spellings, longest first (``("--verbose", "-v")``), or a
    positional's metavar."""
    return [
        tuple(sorted(action.option_strings, key=len, reverse=True))
        if action.option_strings
        else (action.metavar or action.dest.upper(),)
        for action in parser._actions
        if not isinstance(action, argparse._HelpAction)
    ]


def _documented_commands(markdown: str) -> list[str]:
    """Every complete ``batchtextreview ...`` command line in code blocks and
    inline code (placeholders such as ``<command>`` and ``...`` skipped)."""
    commands = []
    for block in _FENCE.findall(markdown):
        # Continuation lines: bash "\", PowerShell "`".
        for line in re.sub(r"[\\`]\n\s*", " ", block).splitlines():
            match = re.match(r"\s*(?:python -m )?batchtextreview\s(.*)", line)
            if match:
                commands.append(match.group(1))
    for span in re.findall(r"`batchtextreview\s([^`]*)`", _FENCED_BLOCK.sub("", markdown)):
        commands.append(" ".join(span.split()))
    return [c for c in commands if "<" not in c and "..." not in c]


def _parse_like_main(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """``cli.main``'s parsing: after ``ui``, "--" and unknown options go to
    Streamlit; anywhere else an unknown option is an error."""
    lead = 0
    while argv[lead : lead + 1] in (["-v"], ["--verbose"]):
        lead += 1
    if argv[lead : lead + 1] == ["ui"] and "--" in argv:
        argv = argv[: argv.index("--")]
    args, unknown = cli.build_parser().parse_known_args(argv)
    return args, unknown if args.command != "ui" else []


def test_documented_command_finder() -> None:
    markdown = (
        "Run `batchtextreview ui --demo` or `batchtextreview -v <command>`.\n"
        "```bash\nbatchtextreview run a.csv \\\n    --model x  # comment\n"
        "python -m batchtextreview ui\nfrom batchtextreview import engine\n```\n"
    )
    assert _documented_commands(markdown) == ["run a.csv  --model x  # comment", "ui", "ui --demo"]


@pytest.mark.parametrize("document", _DOCS_WITH_COMMANDS, ids=lambda p: p.name)
def test_documented_commands_parse(document: Path) -> None:
    commands = _documented_commands(document.read_text(encoding="utf-8"))
    if document != _DOCS_WITH_COMMANDS[-1]:
        assert commands, f"no batchtextreview commands found in {document.name}"
    for command in commands:
        argv = shlex.split(command, comments=True)
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                _args, unknown = _parse_like_main(argv)
        except SystemExit:
            pytest.fail(f"{document.name}: `batchtextreview {command}` - {stderr.getvalue()}")
        assert not unknown, f"{document.name}: `batchtextreview {command}` - unknown {unknown}"


def test_migrating_imports_from_the_old_folder_into_the_current_database() -> None:
    # --results-dir picks the destination database, not the version-1 folder.
    imports = [
        _parse_like_main(shlex.split(command, comments=True))[0]
        for command in _documented_commands(MIGRATING.read_text(encoding="utf-8"))
        if command.startswith("import-legacy")
    ]
    assert imports
    assert all(args.folder and not args.results_dir for args in imports)


def test_readme_run_options_table_matches_the_parser() -> None:
    table = _readme().split("`run` options:", 1)[1].split("Exit codes", 1)[0]
    documented = set()
    for row in table.splitlines():
        if row.startswith("| `"):
            first_cell = row.split("|")[1]
            documented |= set(re.findall(r"`(--[a-z-]+|[A-Z]+)\b", first_cell))
    assert documented == {names[0] for names in _option_spellings(_subcommands()["run"])}


def test_readme_mentions_every_other_command_option() -> None:
    markdown = _readme()
    # Paragraphs outside code blocks, and each code-block line on its own.
    chunks = _FENCED_BLOCK.sub("", markdown).split("\n\n")
    for block in _FENCE.findall(markdown):
        chunks += block.splitlines()

    def mentioned(command: str, spellings: tuple[str, ...]) -> bool:
        options = "|".join(re.escape(spelling) for spelling in spellings)
        return any(
            re.search(rf"\b{command}\b", chunk)
            and re.search(rf"(?<![\w-])({options})(?![\w-])", chunk)
            for chunk in chunks
        )

    missing = [
        f"{name} {spellings[0]}"
        for name, parser in _subcommands().items()
        if name != "run"
        for spellings in _option_spellings(parser)
        if not mentioned(name, spellings)
    ]
    assert not missing, f"README.md doesn't mention: {missing}"
    assert "batchtextreview -v" in markdown


def test_readme_documents_the_default_output_name() -> None:
    [output] = [a for a in _subcommands()["run"]._actions if "--output" in a.option_strings]
    default = re.search(r"default:\s*(\S+)\)", output.help or "")
    assert default, output.help
    assert f"| `{default.group(1)}` |" in _readme()


def test_readme_lists_every_exit_code() -> None:
    sentence = _readme().split("Exit codes:", 1)[1].split("\n\n", 1)[0]
    for code in (cli.EXIT_OK, cli.EXIT_ERROR, cli.EXIT_USAGE, cli.EXIT_INTERRUPTED):
        assert f"`{code}`" in sentence, code


def test_readme_configuration_lists_every_environment_variable() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REPO_ROOT / "src" / "batchtextreview").rglob("*.py")
    )
    variables = set(re.findall(r"\"(BTR_[A-Z_]+)\"", source))
    assert {"BTR_RESULTS_DIR", "BTR_VERBOSE"} <= variables
    configuration = _section(_readme(), "Configuration")
    for variable in variables:
        assert f"| `{variable}` |" in configuration, variable
    # The ones set at startup are explained too.
    env_source = (REPO_ROOT / "src" / "batchtextreview" / "_env.py").read_text(encoding="utf-8")
    for variable in re.findall(r"setdefault\(\"([A-Z_]+)\"", env_source):
        assert f"`{variable}=" in configuration, variable


def test_readme_project_layout_lists_every_module() -> None:
    layout = _section(_readme(), "Project layout")
    package = REPO_ROOT / "src" / "batchtextreview"
    for path in package.glob("*.py"):
        if path.name not in {"__init__.py", "__main__.py"}:
            assert re.search(rf"\b{re.escape(path.name)}\b", layout), path.name
    for folder in ("backends/", "ui/"):
        assert folder in layout


def test_readme_shows_the_screenshot() -> None:
    match = re.search(r"!\[([^\]]+)\]\(docs/screenshot\.png\)", _readme())
    assert match and len(match.group(1)) > 20, "README.md should embed docs/screenshot.png"
    assert (REPO_ROOT / "docs" / "screenshot.png").is_file()


# Labels the README's walkthrough tells you to look for, as the app spells them.
_UI_LABELS = (
    "Sample data",
    "Allowed answers",
    "Score by exact match",
    "Preview prompt",
    "Save as preset",
    "Reset to preset",
    "Custom model…",
    "First rows",
    "Random sample",
    "Advanced",
    "Weights",
    "Discard saved outputs",
    "Show saved results",
    "Export",
    "Unload",
)


@pytest.mark.parametrize("label", _UI_LABELS)
def test_readme_ui_labels_exist_in_the_app(label: str) -> None:
    assert f"**{label}**" in " ".join(_readme().split())
    ui_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REPO_ROOT / "src" / "batchtextreview" / "ui").rglob("*.py")
    )
    assert f'"{label}' in ui_source, f"the app has no {label!r} label any more"


@pytest.mark.parametrize("document", [README, MIGRATING], ids=lambda p: p.name)
def test_python_examples_run(
    document: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As written, with the demo model in place of a real one.
    [code] = _PYTHON_FENCE.findall(document.read_text(encoding="utf-8"))
    code, swapped = re.subn(r"resolve_model\(\"[^\"]+\"\)", 'resolve_model("demo://fake")', code)
    assert swapped == 1
    (tmp_path / "examples").mkdir()
    shutil.copyfile(SAMPLE, tmp_path / "examples" / SAMPLE.name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BTR_RESULTS_DIR", str(tmp_path / "results"))
    namespace: dict[str, Any] = {}
    exec(compile(code, str(document), "exec"), namespace)
    result = namespace["result"]
    assert result.complete and result.outputs
    assert (tmp_path / "results" / "batchtextreview.db").is_file()
