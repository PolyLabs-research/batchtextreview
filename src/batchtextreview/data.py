"""Loading tabular input files and guessing their roles."""

from __future__ import annotations

import codecs
import csv
import hashlib
import io
import json
import logging
import os
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
    ".xlsm",
    ".parquet",
    ".jsonl",
    ".json",
)
"""Extensions offered in the file picker / uploader."""

ENCODINGS: tuple[str, ...] = ("utf-8-sig", "cp1252", "latin-1")
"""Tried in order for delimited text. utf-8-sig also reads plain UTF-8 and
strips an Excel BOM; latin-1 never fails, so it is the last resort. (A
UTF-16 or UTF-32 byte-order mark is checked for first.)"""

MIXED_ENCODING = "utf-8+cp1252"
"""``LoadedTable.encoding`` of a UTF-8 file with some Windows-1252 bytes in
it (rows pasted or appended from an ANSI file): each part is decoded with
its own codec. Not a Python codec name."""

ENCODING_LABELS: dict[str, str] = {
    "utf-8": "UTF-8",
    "utf-8-sig": "UTF-8",
    MIXED_ENCODING: "UTF-8 + Windows-1252",
    "cp1252": "Windows-1252",
    "latin-1": "Latin-1",
    "utf-16-le": "UTF-16 LE",
    "utf-16-be": "UTF-16 BE",
    "utf-32-le": "UTF-32 LE",
    "utf-32-be": "UTF-32 BE",
}
"""Display names for every ``LoadedTable.encoding`` value."""

_log = logging.getLogger(__name__)

_DELIMITED = (".csv", ".tsv", ".txt")
_EXCEL = (".xlsx", ".xlsm")
_DELIMITERS = (",", "\t", ";", "|")
_SNIFF_CHARS = 64 * 1024
_SNIFF_ROWS = 200
# Used to read a file with no delimiter in its header as one column of lines.
_NO_DELIMITER = "\x1f"
_UNNAMED = re.compile(r"Unnamed: \d+")
_DELIMITER_NAMES = {",": "comma", "\t": "tab", ";": "semicolon", "|": "pipe"}
_FIELD_COUNT = re.compile(r"Expected (\d+) fields in line \d+, saw (\d+)")
_LEADING_BLANK_LINES = re.compile(r"\A(?:[^\S\r\n]*\r?\n)+")
_SEP_LINE = re.compile(r"\Asep=(\t|[^\s\"])[^\S\r\n]*(?=\r?\n|\Z)", re.IGNORECASE)
# Windows PowerShell's Export-Csv writes "#TYPE <.NET type name>" first.
_TYPE_LINE = re.compile(r"\A#TYPE [^\r\n]*(?=[\r\n]|\Z)")
# Sniffing: a delimiter that most rows don't split into the header's width
# is only accepted if the rows still look like a table (_splits_like_a_table).
_MIN_CONSISTENCY = 0.8
_MAX_UNSPLIT_SHARE = 0.3
_MIN_SPACED_SHARE = 0.8
# At most this many title / note lines (or blank and title rows in Excel)
# above a table's header; in Excel at most _MAX_TITLE_ROWS of them filled.
_MAX_PREAMBLE_ROWS = 10
_MAX_TITLE_ROWS = 3
_MIN_PREAMBLE_CONSISTENCY = 0.9
# Byte-order mark -> codec for what follows it. UTF-32 first: its
# little-endian BOM starts with the UTF-16 one.
_UNICODE_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)


class DataLoadError(ValueError):
    """The file couldn't be read as a table. Message is user-facing."""


@dataclass
class LoadedTable:
    frame: pd.DataFrame
    name: str
    """File name (no directory)."""
    fmt: str
    """"csv", "tsv", "excel", "parquet", "json" or "jsonl"."""
    encoding: str | None = None
    """Codec the text was decoded with: "utf-8", "utf-8-sig" (UTF-8 with a
    BOM), "cp1252", "latin-1", "utf-16-le", "utf-16-be", "utf-32-le" or
    "utf-32-be", or :data:`MIXED_ENCODING` (see :data:`ENCODING_LABELS`);
    None for Excel and Parquet."""
    delimiter: str | None = None
    sheet: str | None = None
    size_bytes: int | None = None


def read_table(
    source: str | Path | bytes | BinaryIO,
    name: str | None = None,
    sheet: str | None = None,
) -> LoadedTable:
    """Read a table from a path, raw bytes, or a binary file-like object.

    * ``name`` is required for bytes / file objects (its extension picks the
      format; Streamlit's UploadedFile has ``.name``, use it if present).
    * Delimited text (.csv/.tsv/.txt): sniff the delimiter (``,`` ``\\t``
      ``;`` ``|``) from the first ~64 KB, falling back to "," (".tsv" ->
      tab). Try :data:`ENCODINGS` in order.
    * Excel: first sheet unless ``sheet`` given (openpyxl). Hidden sheets
      are skipped for the default, and the table may start below empty rows
      or a title row (see ``_excel_header_row``).
    * Every column is read as *string* (``dtype=str``) so IDs keep leading
      zeros and numeric-looking labels compare as text. Only truly empty
      cells become missing: ``keep_default_na=False, na_values=[""]`` (so a
      text of "NA" or "None" is kept as text). Parquet/JSON are converted
      the same way after reading (``astype("string")`` then to object with
      missing preserved).
    * Column names are converted to ``str`` and stripped of whitespace and
      a BOM; duplicate names get " (2)", " (3)" suffixes; fully empty
      columns named "Unnamed: N" are dropped.
    * A table with no columns or no rows raises DataLoadError; so does any
      parser failure (with the original reason in the message).

    Also: a UTF-16 or UTF-32 file with a byte-order mark (Excel's "Unicode
    Text" export is tab-separated UTF-16 LE) is decoded as such, and
    ``encoding`` says which ("utf-16-le", "utf-16-be", "utf-32-le",
    "utf-32-be"); a UTF-8 file with a few Windows-1252 bytes in it is read
    as both (``encoding`` :data:`MIXED_ENCODING`); a text file whose header
    has no delimiter - or whose lines don't split like a table, as one text
    per line with commas in it doesn't - is read as a single column, one
    row per line (quote marks kept as written unless the file is valid
    quoted CSV); whitespace-only lines before the header are skipped, as
    are PowerShell's "#TYPE" line and title lines above a table whose rows
    all split alike; an Excel "sep=;" first line sets the delimiter; rows
    where every cell is empty are dropped.
    Every column ends up with the dtype ``read_csv(dtype=str)`` gives on the
    running pandas (object on 2.x, ``str`` on 3.x) and NaN for missing.
    """
    raw, inferred = _read_bytes(source)
    name = Path(name or inferred or "").name
    if not name:
        raise DataLoadError(
            "A file name with an extension is needed to tell the format of uploaded data."
        )
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(_unsupported_message(name, ext))
    if not raw.strip():
        raise DataLoadError(f"{name} is empty.")

    if ext in _DELIMITED:
        table = _read_delimited(raw, name, prefer_tab=ext == ".tsv")
    elif ext in _EXCEL:
        table = _read_excel(raw, name, sheet)
    elif ext == ".parquet":
        table = _read_parquet(raw, name)
    elif ext == ".jsonl":
        table = _read_json(raw, name, lines=True)
    else:
        table = _read_json(raw, name, lines=False)

    table.frame = _tidy(table.frame, name)
    table.size_bytes = len(raw)
    return table


def excel_sheets(source: str | Path | bytes | BinaryIO) -> list[str]:
    """Sheet names of an Excel workbook (empty list if unreadable).

    Worksheets in workbook order, except that hidden ones come last, so the
    first name is the sheet :func:`read_table` reads by default. Chart
    sheets hold no table and are left out."""
    try:
        raw, _ = _read_bytes(source)
        import openpyxl

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            book = openpyxl.load_workbook(io.BytesIO(raw), read_only=True)
        try:
            return _visible_first(book, [str(sheet.title) for sheet in book.worksheets])
        finally:
            book.close()
    except Exception:
        return []


TEXT_COLUMN_NAMES = (
    "text",
    "body",
    "content",
    "message",
    "comment",
    "comments",
    "review",
    "description",
    "note",
    "notes",
)
ID_COLUMN_NAMES = ("index", "id", "row_id", "uid", "key", "reference", "ref")

_GOLD_PREFIXES = ("gold", "label", "target", "truth", "expected", "y_")
_MAX_CATEGORIES = 20
_GUESS_ROWS = 200
_NAME_MIN_FILLED = 0.5
_NAME_MIN_LENGTH_SHARE = 0.25
# The columns results.build_results_frame (and the CORRECT column of a
# scored export) add, with the "_2" suffixes a re-run on an export gets.
_RESULT_STATUSES = frozenset({"generated", "cached", "blank", "error", "pending"})
_RESULT_NAME = re.compile(
    r"(?:OUTPUT|LABEL|STATUS|ERROR|NOTE|CORRECT)(?:_\d+)?|OUTPUT(?:_\d+)?\..+"
)
_STATUS_NAME = re.compile(r"STATUS(?:_\d+)?")
_OUTPUT_NAME = re.compile(r"OUTPUT(?:_\d+)?")
_PREVIOUS_LABEL = re.compile(r"LABEL(?:_\d+)?")


def guess_text_column(frame: pd.DataFrame) -> str | None:
    """A column whose name matches TEXT_COLUMN_NAMES case-insensitively
    (first in that priority order), else the column with the longest mean
    string length among the first 200 rows (ties -> leftmost). None for an
    empty frame.

    Missing cells count as length 0 in the mean, so a mostly-empty notes
    column doesn't beat the real text. A name match only counts when the
    column is filled in at least half of those rows and its mean length is
    at least a quarter of the longest column's: a sparse "Notes" column or a
    German "Note" (grade) column doesn't beat the real text either. In a
    results file exported by this app, the result columns (OUTPUT, STATUS,
    ...) are never guessed.
    """
    if frame.empty:
        return None
    skip = _result_columns(frame)
    positions = [p for p, column in enumerate(frame.columns) if column not in skip]
    positions = positions or list(range(frame.shape[1]))
    head = frame.head(_GUESS_ROWS)
    lengths = {position: _mean_length(head.iloc[:, position]) for position in positions}
    longest = max(lengths.values())
    for column in _match_names(frame, TEXT_COLUMN_NAMES):
        position = list(frame.columns).index(column)
        if position not in lengths:
            continue
        values = head.iloc[:, position]
        filled = int(values.map(_is_filled).sum())
        if (
            filled >= _NAME_MIN_FILLED * len(values)
            and lengths[position] >= _NAME_MIN_LENGTH_SHARE * longest
        ):
            return column
    best: Any = None
    best_length = -1.0
    for position in positions:
        if lengths[position] > best_length:
            best, best_length = frame.columns[position], lengths[position]
    return best


def guess_id_column(frame: pd.DataFrame) -> str | None:
    """A column named like ID_COLUMN_NAMES (case-insensitive) whose values
    are unique, else None.

    Exact names win (in ID_COLUMN_NAMES order); after them, names ending in
    "_id" / " id" / "-id" or a camel-case "ID" ("CaseID"). A column with any
    missing value is not an ID.
    """
    if len(frame) == 0:
        return None
    candidates = _match_names(frame, ID_COLUMN_NAMES)
    candidates += [
        column
        for column in frame.columns
        if column not in candidates and _looks_like_id_name(column)
    ]
    for column in candidates:
        series = _column(frame, column)
        try:
            if series.notna().all() and series.is_unique:
                return column
        except TypeError:  # unhashable cells
            continue
    return None


def guess_gold_columns(frame: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """Plausible gold-label columns, best first: names starting with "gold"
    / "label" / "target" / "truth" / "expected" / "y_" (case-insensitive),
    then other low-cardinality columns (<= 20 distinct non-missing values
    and fewer distinct values than half the rows). Never includes
    ``exclude``.

    Columns with no values at all are never offered. In a results file
    exported by this app, the earlier run's LABEL comes last and its other
    result columns (OUTPUT, STATUS, CORRECT, ...) are left out.
    """
    skip = set(exclude)
    results = _result_columns(frame)
    named: list[str] = []
    low_cardinality: list[str] = []
    previous: list[str] = []
    for position, column in enumerate(frame.columns):
        if column in skip:
            continue
        distinct = _distinct_count(frame.iloc[:, position])
        if not distinct:
            continue
        if column in results:
            if _PREVIOUS_LABEL.fullmatch(str(column)):
                previous.append(column)
        elif _normalise_name(column).startswith(_GOLD_PREFIXES):
            named.append(column)
        elif distinct <= _MAX_CATEGORIES and distinct < len(frame) / 2:
            low_cardinality.append(column)
    return named + low_cardinality + previous


def guess_group_columns(frame: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """Columns suitable for slicing evaluation (2..20 distinct values),
    excluding ``exclude``.

    A column where every value is different (e.g. the text itself in a tiny
    file) is not a grouping, so it is left out.
    """
    skip = set(exclude)
    out: list[str] = []
    for position, column in enumerate(frame.columns):
        if column in skip:
            continue
        series = frame.iloc[:, position]
        distinct = _distinct_count(series)
        if distinct is None or not 2 <= distinct <= _MAX_CATEGORIES:
            continue
        if distinct < int(series.notna().sum()):
            out.append(column)
    return out


def list_data_files(folder: str | Path) -> list[Path]:
    """Supported files directly inside ``folder``, newest first. Empty list
    if the folder doesn't exist or can't be listed.

    Hidden files and Office lock files ("~$book.xlsx") are skipped; equal
    modification times are ordered by name.
    """
    try:
        entries = list(Path(folder).expanduser().iterdir())
    except (OSError, TypeError, ValueError):
        return []
    found: list[tuple[float, Path]] = []
    for path in entries:
        if path.name.startswith((".", "~$")) or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            if not path.is_file():
                continue
            modified = path.stat().st_mtime
        except OSError:
            continue
        found.append((modified, path))
    found.sort(key=lambda item: (-item[0], item[1].name.lower(), item[1].name))
    return [path for _, path in found]


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """16-hex sha256 of the frame's content (via ``pd.util.hash_pandas_object``
    with index), for cache keys in the UI.

    Column names are part of the content, so renaming a column changes it.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps([str(column) for column in frame.columns]).encode("utf-8"))
    try:
        hashed = pd.util.hash_pandas_object(frame, index=True)
    except TypeError:  # unhashable cells such as lists
        hashed = pd.util.hash_pandas_object(frame.astype(str), index=True)
    digest.update(np.ascontiguousarray(hashed.to_numpy()).tobytes())
    return digest.hexdigest()[:16]


# --- reading -----------------------------------------------------------------


def _read_bytes(source: str | Path | bytes | BinaryIO) -> tuple[bytes, str | None]:
    """The raw content of ``source`` plus the file name it implies."""
    if isinstance(source, (str, os.PathLike)):
        path = Path(source).expanduser()
        if path.is_dir():
            raise DataLoadError(f"{path} is a folder, not a file.")
        try:
            return path.read_bytes(), path.name
        except FileNotFoundError:
            raise DataLoadError(f"File not found: {path}") from None
        except PermissionError:
            raise DataLoadError(
                f"Permission denied reading {path} - is it open in another program?"
            ) from None
        except OSError as exc:
            raise DataLoadError(f"Couldn't open {path}: {exc.strerror or exc}") from None
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), None

    label = getattr(source, "name", None)
    inferred = Path(label).name if isinstance(label, str) and label else None
    try:
        # The UI hands over the same upload object on every rerun, so read
        # from the start rather than wherever the last reader left off.
        if hasattr(source, "getvalue"):
            data = source.getvalue()
        else:
            if getattr(source, "seekable", lambda: False)():
                source.seek(0)
            data = source.read()
    except (OSError, ValueError) as exc:
        raise DataLoadError(f"Couldn't read {inferred or 'the upload'}: {exc}") from exc
    if isinstance(data, str):
        data = data.encode("utf-8")
    return bytes(data), inferred


def _unsupported_message(name: str, ext: str) -> str:
    supported = ", ".join(SUPPORTED_EXTENSIONS)
    if ext == ".xls":
        return (
            f"{name} is a legacy .xls workbook, which isn't supported. Open it in Excel "
            "and save it as .xlsx or CSV."
        )
    if not ext:
        return f"Can't tell the format of {name} (no file extension). Supported: {supported}."
    return f"Unsupported file type '{ext}' ({name}). Supported: {supported}."


def _decode(raw: bytes, name: str) -> tuple[str, str]:
    """Decoded text and the encoding that worked."""
    for bom, encoding in _UNICODE_BOMS:
        if not raw.startswith(bom):
            continue
        try:
            text = raw[len(bom) :].decode(encoding)
        except UnicodeDecodeError as exc:
            raise DataLoadError(
                f"Couldn't decode {name} as {ENCODING_LABELS[encoding]}: {exc.reason}."
            ) from exc
        _log.debug("%s: byte-order mark found, decoded as %s", name, encoding)
        return text, encoding
    if raw.startswith(b"PK\x03\x04"):
        raise DataLoadError(
            f"{name} looks like an Excel workbook (or another zip file), not a text "
            "file. Rename it to .xlsx, or save it from Excel as CSV."
        )
    if raw.startswith(b"PAR1"):
        raise DataLoadError(f"{name} looks like a Parquet file. Rename it to .parquet.")
    if b"\x00" in raw[:8192]:
        raise DataLoadError(f"{name} contains binary data - it isn't a text file.")
    has_bom = raw.startswith(codecs.BOM_UTF8)
    for encoding in ENCODINGS:
        body = raw if encoding.startswith("utf-8") or not has_bom else raw[len(codecs.BOM_UTF8) :]
        try:
            text = body.decode(encoding)
        except UnicodeDecodeError:
            if encoding == "utf-8-sig" and _has_utf8_text(
                raw[len(codecs.BOM_UTF8) :] if has_bom else raw
            ):
                # Real UTF-8 with a few stray Windows-1252 bytes: decoding it
                # all as cp1252 would garble every other accented character.
                text = body.decode("utf-8-sig", errors=_CP1252_FALLBACK)
                _log.debug("%s: decoded as UTF-8 with some Windows-1252 bytes", name)
                return text, MIXED_ENCODING
            continue
        if encoding == "utf-8-sig" and not has_bom:
            encoding = "utf-8"
        _log.debug("%s: decoded as %s", name, encoding)
        return text, encoding
    raise DataLoadError(f"Couldn't decode {name} as text.")  # pragma: no cover - latin-1


def _has_utf8_text(raw: bytes) -> bool:
    """Whether some line with non-ASCII bytes is valid UTF-8. Windows-1252
    text practically never is: its accented letters are single bytes, which
    UTF-8 only allows in multi-byte sequences."""
    for line in raw.split(b"\n"):
        if line.isascii():
            continue
        try:
            line.decode("utf-8")
        except UnicodeDecodeError:
            continue
        return True
    return False


def _cp1252_fallback(error: UnicodeError) -> tuple[str, int]:
    """Codec error handler: bytes that aren't valid UTF-8 are read as
    Windows-1252 (Latin-1 for the five bytes it leaves undefined)."""
    if not isinstance(error, UnicodeDecodeError):
        raise error
    chunk = error.object[error.start : error.end]
    chars = []
    for byte in chunk:
        try:
            chars.append(bytes([byte]).decode("cp1252"))
        except UnicodeDecodeError:
            chars.append(chr(byte))
    return "".join(chars), error.end


_CP1252_FALLBACK = "batchtextreview.cp1252-fallback"
codecs.register_error(_CP1252_FALLBACK, _cp1252_fallback)


def _sniff_delimiter(text: str, prefer_tab: bool) -> str | None:
    """The candidate delimiter that splits the header into the most
    consistent columns, or None if no candidate splits the header at all
    or the rows don't split like a table (see :func:`_splits_like_a_table`).

    csv.Sniffer is easily fooled by punctuation in free text; scoring on
    "every row has as many fields as the header" isn't.
    """
    sample, truncated = _sniff_sample(text)
    best: str | None = None
    best_score: tuple[float, int, int] | None = None
    for rank, delimiter in enumerate(_delimiter_order(prefer_tab)):
        rows = [row for _, row in _sample_rows(sample, delimiter, truncated)]
        if not rows or len(rows[0]) < 2:
            continue
        if prefer_tab and delimiter == "\t":
            return delimiter
        width = len(rows[0])
        body = [row for row in rows[1:] if any(cell.strip() for cell in row)]
        if not _splits_like_a_table(rows[0], body, delimiter):
            continue
        consistency = sum(len(row) == width for row in body) / len(body) if body else 1.0
        score = (consistency, width, -rank)
        if best_score is None or score > best_score:
            best, best_score = delimiter, score
    return best


def _splits_like_a_table(header: list[str], body: list[list[str]], delimiter: str) -> bool:
    """Whether rows split by ``delimiter`` look like a table rather than
    lines of text that happen to contain it ("Okay, I guess not").

    Most rows having the header's width settles it. Otherwise a table is
    ragged (short rows get padded; a long one is a mistake worth an error),
    while text - with commas or semicolons, the ones prose uses - shows
    itself by a space after nearly every one, or by a first line written
    that way above many lines with none at all. (A hand-typed table whose
    ID-only rows leave out the trailing commas has a plain header.)
    """
    if not body or delimiter not in ",;":
        return True
    width = len(header)
    if sum(len(row) == width for row in body) >= _MIN_CONSISTENCY * len(body):
        return True
    prose_header = any(cell[:1].isspace() for cell in header[1:])
    if prose_header and sum(len(row) == 1 for row in body) >= _MAX_UNSPLIT_SHARE * len(body):
        return False
    following = [cell for row in (header, *body) for cell in row[1:]]
    spaced = sum(cell[:1].isspace() for cell in following)
    return not following or spaced < _MIN_SPACED_SHARE * len(following)


def _find_preamble(text: str, prefer_tab: bool) -> tuple[str, int] | None:
    """(delimiter, lines to skip) when the table starts below a few lines
    with no delimiter - a report title, an export note - else None.

    Only when nearly every sampled row from the header on has the header's
    width, with at least two data rows and three columns (two for tab or
    pipe): one text per line that happens to hold commas doesn't look like
    that.
    """
    sample, truncated = _sniff_sample(text)
    for delimiter in _delimiter_order(prefer_tab):
        rows = _sample_rows(sample, delimiter, truncated)
        min_width = 3 if delimiter in ",;" else 2
        for position, (line, header) in enumerate(rows[: _MAX_PREAMBLE_ROWS + 1]):
            if len(header) == 1:
                continue  # still above the table
            body = [row for _, row in rows[position + 1 :] if any(cell.strip() for cell in row)]
            alike = sum(len(row) == len(header) for row in body)
            if (
                position
                and len(header) >= min_width
                and len(body) >= 2
                and alike >= _MIN_PREAMBLE_CONSISTENCY * len(body)
            ):
                return delimiter, line - 1
            break
    return None


def _sniff_sample(text: str) -> tuple[str, bool]:
    """The start of ``text`` the sniffers look at, cut at a line break, and
    whether it is shorter than the text."""
    sample = text[:_SNIFF_CHARS]
    truncated = len(text) > len(sample)
    if truncated and "\n" in sample:
        sample = sample[: sample.rfind("\n")]
    return sample, truncated


def _delimiter_order(prefer_tab: bool) -> tuple[str, ...]:
    return ("\t", ",", ";", "|") if prefer_tab else _DELIMITERS


def _sample_rows(sample: str, delimiter: str, truncated: bool) -> list[tuple[int, list[str]]]:
    """Non-empty rows of ``sample`` with the line each starts on."""
    rows: list[tuple[int, list[str]]] = []
    reader = csv.reader(io.StringIO(sample, newline=""), delimiter=delimiter)
    start = 1
    try:
        for row in reader:
            if row:
                rows.append((start, row))
                if len(rows) >= _SNIFF_ROWS:
                    break
            start = reader.line_num + 1
    except csv.Error:
        pass
    if truncated and len(rows) > 2:
        rows = rows[:-1]  # may have been cut off mid-record
    return rows


def _blank_lines(text: str, count: int) -> str:
    """``text`` with its first ``count`` lines emptied (line numbers kept)."""
    found = re.match(rf"(?:[^\r\n]*(?:\r\n|\r|\n)){{{count}}}", text)
    return "\n" * count + text[found.end() :] if found else text


def _read_delimited(raw: bytes, name: str, prefer_tab: bool) -> LoadedTable:
    text, encoding = _decode(raw, name)
    # Lines are emptied rather than removed so error messages keep the
    # file's line numbers; the parser skips empty lines.
    text = _LEADING_BLANK_LINES.sub(lambda match: "\n" * match.group().count("\n"), text)
    declared = _SEP_LINE.match(text)  # Excel's "sep=;" first line
    if declared:
        text = text[declared.end() :]
    type_line = _TYPE_LINE.match(text)
    if type_line:
        _log.debug("%s: skipped PowerShell's #TYPE line", name)
        text = text[type_line.end() :]
    delimiter = declared[1] if declared else _sniff_delimiter(text, prefer_tab)
    how = "from its sep= line" if declared else "sniffed"
    if delimiter is None:
        preamble = _find_preamble(text, prefer_tab)
        if preamble:
            delimiter, skipped = preamble
            text = _blank_lines(text, skipped)
            how = f"sniffed below {skipped} line{'s' if skipped != 1 else ''} above the header"
        else:
            how = "none fits, so one column"
    reported = delimiter or ("\t" if prefer_tab else ",")
    _log.debug("%s: delimiter %r (%s)", name, reported, how)
    separator = delimiter
    quoting = csv.QUOTE_MINIMAL
    if separator is None:
        separator = _NO_DELIMITER if _NO_DELIMITER not in text else reported
        if separator == _NO_DELIMITER and not _quotes_follow_csv(text, separator):
            # Plain lines of prose: keep quote marks exactly as written.
            quoting = csv.QUOTE_NONE
    frame = _parse_delimited(text, separator, name, reported, quoting)
    return LoadedTable(
        frame=_promote_header(frame, name),
        name=name,
        fmt="tsv" if reported == "\t" else "csv",
        encoding=encoding,
        delimiter=reported,
    )


def _quotes_follow_csv(text: str, separator: str) -> bool:
    """Whether every double quote in ``text`` is CSV quoting - a quoted
    field that closes right before a delimiter or line end - rather than
    quote marks in prose such as ``"Best" product ever``."""
    if '"' not in text:
        return True
    try:
        for _ in csv.reader(io.StringIO(text, newline=""), delimiter=separator, strict=True):
            pass
    except csv.Error:
        return False
    return True


def _parse_delimited(
    text: str, separator: str, name: str, reported: str, quoting: int = csv.QUOTE_MINIMAL
) -> pd.DataFrame:
    """Parse with the C engine only: pandas' python engine silently stops
    at a quote it can't parse, losing every row after it."""
    options: dict[str, Any] = {
        "sep": separator,
        "header": None,
        "dtype": str,
        "keep_default_na": False,
        "na_values": [""],
        "skip_blank_lines": True,
        "quoting": quoting,
    }
    width = 0  # the header's field count, once a longer row turned up
    names: list[int] | None = None
    while True:
        try:
            frame = pd.read_csv(io.StringIO(text, newline=""), engine="c", names=names, **options)
            break
        except pd.errors.EmptyDataError:
            raise DataLoadError(f"{name} is empty.") from None
        except pd.errors.ParserError as exc:
            found = _FIELD_COUNT.search(str(exc))
            if found is None or int(found[2]) <= len(names or ()):
                raise _parse_error(text, separator, name, reported, _reason(exc)) from exc
            if not width:
                width = int(found[1])
                located = _first_overflow(text, separator, width)
                if located:
                    line, fields = located
                    reason = f"line {line} has {fields} fields but the header has {width}"
                    raise _parse_error(text, separator, name, reported, reason) from exc
            # Rows with extra *empty* fields ("1,a,,") are harmless - hand-
            # edited files often have them - so read wide enough for the
            # longest row and drop the extra columns if they hold nothing.
            names = list(range(int(found[2])))
        except Exception as exc:
            raise DataLoadError(f"Couldn't parse {name}: {_reason(exc)}.") from exc
    if not width:
        return frame
    extra = frame.iloc[:, width:].map(lambda cell: isinstance(cell, str) and bool(cell.strip()))
    if extra.to_numpy().any():  # csv and the C parser disagree on where a row ends
        reason = f"some rows have more fields than the header ({width})"
        raise _parse_error(text, separator, name, reported, reason)
    return frame.iloc[:, :width]


def _parse_error(text: str, separator: str, name: str, reported: str, reason: str) -> DataLoadError:
    """A user-facing parse error with a hint on how to fix the file."""
    hint = (
        f" Check that it is {_DELIMITER_NAMES.get(reported, repr(reported))}-delimited and "
        "that any text containing the delimiter or line breaks is wrapped in double quotes."
    )
    if "EOF inside string" in reason:
        line = _last_row_start(text, separator)
        reason = f"a double quote in the row starting on line {line} is never closed"
        hint = (
            " Add the missing closing quote; a double quote inside quoted text must be "
            'written twice ("").'
        )
    return DataLoadError(f"Couldn't parse {name}: {reason}.{hint}")


def _last_row_start(text: str, separator: str) -> int:
    """Line where the last row starts. A quoted field that is never closed
    runs to the end of the file, so this is the line it starts on."""
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=separator)
    start = next_start = 1
    try:
        for row in reader:
            if row:
                start = next_start
            next_start = reader.line_num + 1
    except csv.Error:  # the unclosed field outgrew csv's field size limit
        return next_start
    return start


def _first_overflow(text: str, separator: str, width: int) -> tuple[int, int] | None:
    """(line number, field count) of the first row with values beyond the
    header's width."""
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=separator)
    start = 1
    try:
        for row in reader:
            if len(row) > width and any(field.strip() for field in row[width:]):
                return start, len(row)
            start = reader.line_num + 1
    except csv.Error:
        pass
    return None


def _read_excel(raw: bytes, name: str, sheet: str | None) -> LoadedTable:
    try:
        with warnings.catch_warnings():
            # openpyxl warns about styles/extensions it ignores; not the user's problem.
            warnings.simplefilter("ignore")
            with pd.ExcelFile(io.BytesIO(raw), engine="openpyxl") as book:
                sheets = [str(title) for title in book.sheet_names]
                if not sheets:
                    raise DataLoadError(f"{name} has no worksheets.")
                if not sheet:
                    target = _visible_first(book.book, sheets)[0]
                elif sheet in sheets:
                    target = sheet
                else:
                    raise DataLoadError(
                        f"Sheet '{sheet}' not found in {name}. Sheets: {', '.join(sheets)}."
                    )
                frame = book.parse(
                    target,
                    header=None,
                    dtype=str,
                    keep_default_na=False,
                    na_values=[""],
                )
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"Couldn't read {name} as an Excel workbook: {_reason(exc)}.") from exc
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise DataLoadError(f"Sheet '{target}' in {name} is empty.")
    start = _excel_header_row(frame)
    if start:
        _log.debug("%s: the table on sheet %r starts at row %d", name, target, start + 1)
        frame = frame.iloc[start:]
    return LoadedTable(
        frame=_promote_header(frame, name),
        name=name,
        fmt="excel",
        sheet=target,
    )


def _visible_first(book: Any, sheets: list[str]) -> list[str]:
    """``sheets`` with the hidden ones (lookup lists and the like, which
    the user never sees in Excel) moved to the end."""
    hidden = set()
    for worksheet in getattr(book, "worksheets", ()):
        if getattr(worksheet, "sheet_state", "visible") != "visible":
            hidden.add(str(worksheet.title))
    visible = [title for title in sheets if title not in hidden]
    return visible + [title for title in sheets if title in hidden]


def _excel_header_row(frame: pd.DataFrame) -> int:
    """Position of the header row in a ``header=None`` sheet: past empty
    rows, and past up to three title rows (a single filled cell each,
    merged or not) when a table follows them (see
    :func:`_looks_like_header`)."""
    filled = frame.head(_MAX_PREAMBLE_ROWS + 1).map(_is_filled).sum(axis=1).tolist()
    titles = 0
    for position, count in enumerate(filled):
        if count == 0:
            continue
        if count == 1 and titles < _MAX_TITLE_ROWS:
            titles += 1
            continue
        if count > 1 and (not titles or _looks_like_header(frame, position)):
            return position
        break
    # No title: the first row with anything in it.
    return next((position for position, count in enumerate(filled) if count), 0)


def _looks_like_header(frame: pd.DataFrame, position: int) -> bool:
    """At least three distinct names that don't turn up again in their own
    column (as data would), above rows that mostly fill two cells or more -
    not a stray wide row in a one-column list."""
    row = frame.iloc[position]
    names = [str(value).strip() for value in row if _is_filled(value)]
    if len(names) < 3 or len(set(names)) < len(names):
        return False
    below = frame.iloc[position + 1 : position + 1 + _GUESS_ROWS]
    counts = below.map(_is_filled).sum(axis=1)
    counts = counts[counts > 0]
    if counts.empty or (counts >= 2).sum() * 2 < len(counts):
        return False
    for column, value in enumerate(row):
        if _is_filled(value):
            values = below.iloc[:, column]
            if (values[values.notna()].astype(str).str.strip() == str(value).strip()).any():
                return False
    return True


def _is_filled(value: Any) -> bool:
    return not _is_missing(value) and bool(str(value).strip())


def _read_parquet(raw: bytes, name: str) -> LoadedTable:
    try:
        # numpy_nullable keeps integer columns with gaps as integers, so they
        # become "3" rather than "3.0".
        frame = pd.read_parquet(io.BytesIO(raw), dtype_backend="numpy_nullable")
    except Exception as exc:
        raise DataLoadError(f"Couldn't read {name} as a Parquet file: {_reason(exc)}.") from exc
    return LoadedTable(frame=_index_as_columns(frame), name=name, fmt="parquet")


def _index_as_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Named index levels (e.g. an ID) are data, so they become columns -
    unless a column of that name already holds the same values, as
    ``set_index("id", drop=False)`` writes."""
    keep = [
        level
        for level, label in enumerate(frame.index.names)
        if label is not None
        and not _is_copy_of_column(frame, label, frame.index.get_level_values(level))
    ]
    if not keep:
        return frame
    return frame.reset_index(level=keep, allow_duplicates=True)


def _is_copy_of_column(frame: pd.DataFrame, label: Any, values: pd.Index) -> bool:
    matches = [position for position, column in enumerate(frame.columns) if column == label]
    if len(matches) != 1:
        return False
    column = frame.iloc[:, matches[0]].reset_index(drop=True).astype("string")
    return bool(column.equals(pd.Series(values).astype("string")))


def _read_json(raw: bytes, name: str, lines: bool) -> LoadedTable:
    text, encoding = _decode(raw, name)
    fmt = "jsonl" if lines else "json"
    records: list[dict[str, Any]] | dict[str, list[Any]]
    if lines:
        records = _json_lines(text, name)
    else:
        try:
            records = _json_records(json.loads(text), name)
        except json.JSONDecodeError as exc:
            if not exc.msg.startswith("Extra data"):
                raise DataLoadError(f"Couldn't parse {name} as JSON: {exc}.") from exc
            # One object per line, saved with a .json extension.
            try:
                records = _json_lines(text, name)
            except DataLoadError:
                raise DataLoadError(f"Couldn't parse {name} as JSON: {exc}.") from exc
            fmt = "jsonl"
    try:
        frame = pd.DataFrame(records)
    except Exception as exc:
        raise DataLoadError(f"Couldn't make a table from {name}: {_reason(exc)}.") from exc
    return LoadedTable(frame=frame, name=name, fmt=fmt, encoding=encoding)


def _json_lines(text: str, name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # Only "\n" separates records: splitlines() would also split on U+2028,
    # U+0085 etc., which JSON allows raw inside strings.
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            where = f"{exc.msg.removesuffix(' at')} at column {exc.colno}"
            raise DataLoadError(f"Couldn't parse line {number} of {name}: {where}.") from exc
        if not isinstance(value, dict):
            raise DataLoadError(
                f"Line {number} of {name} is not a JSON object - each line must be "
                '{"column": value, ...}.'
            )
        records.append(_json_row(value))
    return records


def _json_records(value: Any, name: str) -> list[dict[str, Any]] | dict[str, list[Any]]:
    """Rows from the usual pandas JSON orients: records, split, table,
    columns (dict of lists / dict of dicts)."""
    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            return [_json_row(item) for item in value]
    elif isinstance(value, dict):
        data = value.get("data")
        if isinstance(data, list) and isinstance(value.get("columns"), list):  # split
            columns = [str(column) for column in value["columns"]]
            rows: list[dict[str, Any]] = []
            for item in data:
                if not isinstance(item, list) or len(item) != len(columns):
                    break
                rows.append(_json_row(dict(zip(columns, item, strict=True))))
            else:
                return rows
        elif isinstance(data, list) and "schema" in value:  # table
            if all(isinstance(item, dict) for item in data):
                rows = [_json_row(item) for item in data]
                schema = value["schema"] if isinstance(value["schema"], dict) else {}
                if schema.get("primaryKey") == ["index"]:  # pandas' default index
                    for row in rows:
                        row.pop("index", None)
                return rows
        elif len(value) == 1 and _is_record_list(next(iter(value.values()))):
            return _json_records(next(iter(value.values())), name)  # {"items": [...]}
        elif value and all(isinstance(item, dict) for item in value.values()):
            # Inner keys are row labels: line rows up by them, not by order.
            keys = list(dict.fromkeys(key for cells in value.values() for key in cells))
            return {
                str(column): [_json_cell(cells.get(key)) for key in keys]
                for column, cells in value.items()
            }
        elif value and all(isinstance(item, list) for item in value.values()):
            if len({len(cells) for cells in value.values()}) == 1:
                return {
                    str(column): [_json_cell(cell) for cell in cells]
                    for column, cells in value.items()
                }
    raise DataLoadError(
        f"{name} isn't a table: expected a JSON array of objects, e.g. "
        '[{"id": 1, "text": "..."}, ...].'
    )


def _is_record_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(i, dict) for i in value)


def _json_row(record: dict[Any, Any]) -> dict[str, Any]:
    return {str(key): _json_cell(cell) for key, cell in record.items()}


def _json_cell(value: Any) -> str | None:
    """The cell as text, close to how it is written in the file."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# --- tidying -----------------------------------------------------------------


def _promote_header(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """First row of a ``header=None`` read becomes the column names, so
    duplicates are ours to name (pandas would make them "a.1")."""
    if frame.shape[0] == 0:
        raise DataLoadError(f"No columns found in {name}.")
    body = frame.iloc[1:].reset_index(drop=True)
    body.columns = list(frame.iloc[0])
    return body


def _tidy(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = _clean_names(list(frame.columns))
    empty_unnamed = [
        column
        for column in frame.columns
        if _UNNAMED.fullmatch(column) and frame[column].isna().all()
    ]
    frame = frame.drop(columns=empty_unnamed)
    if frame.shape[1] == 0:
        raise DataLoadError(f"No columns found in {name}.")
    frame = frame.dropna(how="all").reset_index(drop=True)
    if len(frame) == 0:
        raise DataLoadError(f"No data rows found in {name} - only column headers.")
    return pd.DataFrame(
        {column: _as_text(frame[column]) for column in frame.columns},
        index=pd.RangeIndex(len(frame)),
    )


def _clean_names(raw: Sequence[Any]) -> list[str]:
    """str, stripped, BOM-free, unique. Blank names become "Unnamed: N"
    (pandas' convention, N = position)."""
    names = []
    for position, value in enumerate(raw):
        text = "" if _is_missing(value) else str(value)
        text = text.replace("﻿", "").strip()
        names.append(text or f"Unnamed: {position}")
    taken = set(names)
    seen: set[str] = set()
    out: list[str] = []
    for text in names:
        if text in seen:
            suffix = 2
            while f"{text} ({suffix})" in taken:
                suffix += 1
            text = f"{text} ({suffix})"
            taken.add(text)
        seen.add(text)
        out.append(text)
    return out


def _as_text(series: pd.Series) -> pd.Series:
    """Values as strings, missing as NaN, in the dtype read_csv(dtype=str)
    produces on this pandas version."""
    values = series.astype("string").to_numpy(dtype=object, na_value=np.nan)
    return pd.Series(values, index=series.index, dtype=str, name=series.name)


def _is_missing(value: Any) -> bool:
    if isinstance(value, str):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # array-like
        return False


def _reason(exc: BaseException) -> str:
    """The exception's message on one line, without pandas' boilerplate."""
    text = " ".join(str(exc).split()) or type(exc).__name__
    text = text.replace("Error tokenizing data. C error: ", "")
    return text.rstrip(".")


# --- guessing ----------------------------------------------------------------


def _normalise_name(column: Any) -> str:
    return str(column).strip().lower()


def _match_names(frame: pd.DataFrame, wanted: Sequence[str]) -> list[Any]:
    """Columns whose names equal one of ``wanted`` (case-insensitive), in
    ``wanted`` order, then left to right."""
    out: list[Any] = []
    for target in wanted:
        out.extend(
            column
            for column in frame.columns
            if _normalise_name(column) == target and column not in out
        )
    return out


def _result_columns(frame: pd.DataFrame) -> set[Any]:
    """The columns this app added, when ``frame`` is an exported results
    file: per run, OUTPUT, LABEL (optional) and a STATUS column holding row
    statuses, then the ERROR / NOTE / OUTPUT.<key> / CORRECT columns that
    follow them. Empty for any other table, so a user's own "Label" or
    "Status" column is never affected."""
    names = [str(column) for column in frame.columns]
    found: set[Any] = set()
    for position, name in enumerate(names):
        if not _STATUS_NAME.fullmatch(name) or not _holds_statuses(frame.iloc[:, position]):
            continue
        start = position - 1
        if start >= 1 and _PREVIOUS_LABEL.fullmatch(names[start]):
            start -= 1
        if start < 0 or not _OUTPUT_NAME.fullmatch(names[start]):
            continue
        end = position + 1
        while (
            end < len(names)
            and _RESULT_NAME.fullmatch(names[end])
            and not _OUTPUT_NAME.fullmatch(names[end])  # the next run's block
        ):
            end += 1
        found.update(frame.columns[start:end])
    return found


def _holds_statuses(series: pd.Series) -> bool:
    values = series[series.notna()]
    try:
        return not values.empty and bool(values.isin(list(_RESULT_STATUSES)).all())
    except TypeError:  # unhashable cells
        return False


def _looks_like_id_name(column: Any) -> bool:
    text = str(column).strip()
    if text.lower().endswith(("_id", " id", "-id", ".id")):
        return True
    return len(text) > 2 and text.endswith("ID") and text[-3].islower()


def _column(frame: pd.DataFrame, column: Any) -> pd.Series:
    position = list(frame.columns).index(column)
    return frame.iloc[:, position]


def _mean_length(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    present = series[series.notna()]
    if present.empty:
        return 0.0
    return float(present.astype(str).str.len().sum()) / len(series)


def _distinct_count(series: pd.Series) -> int | None:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:  # unhashable cells
        return None
