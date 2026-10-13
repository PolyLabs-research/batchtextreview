"""Deterministic stand-in for a real model, for tests and demo mode."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence

from batchtextreview.backends.base import GenerationError
from batchtextreview.models import DEMO_REPO
from batchtextreview.presets import parse_label_list

Responder = Callable[[str], str]

_TEXT_MARKER = "TEXT:\n"


def demo_responder(prompt: str) -> str:
    """Plausible fake answers so the demo UI looks meaningful.

    Recognises the built-in presets by their instructions (sentiment,
    emotion, intent, urgency, yes/no, categorise, language, JSON-returning
    extraction tasks, summaries...) and answers from a tiny keyword lexicon
    applied to the TEXT part of the prompt (everything after the last
    "TEXT:\\n" marker, or the whole prompt). Unknown prompts get the first 12
    words of the text. Deterministic: same prompt -> same answer.
    """
    prompt = str(prompt)
    cut = prompt.rfind(_TEXT_MARKER)
    if cut >= 0:
        instruction, text = prompt[:cut], prompt[cut + len(_TEXT_MARKER) :]
    else:
        instruction, text = prompt, prompt
    # Loose keyword cues only apply to a template-shaped prompt: with the
    # bare "{text}" template the "instruction" is the row itself, and a
    # review that mentions "language" is not a language-detection task.
    task = _recognise(instruction.lower(), loose=cut >= 0)
    if task is None:
        return _first_words(text, 12)
    return _HANDLERS[task](text.strip(), instruction)


class FakeBackend:
    """Implements the Backend protocol without any ML dependencies."""

    def __init__(
        self,
        model_id: str = DEMO_REPO,
        precision: str = "none",
        responder: Responder | None = None,
        delay_s: float = 0.0,
        fail_when: Callable[[Sequence[str]], bool] | None = None,
    ) -> None:
        """
        responder   maps one prompt to one output (default: demo_responder)
        delay_s     sleep per generate() call (simulates a slow model)
        fail_when   if it returns True for a batch, generate() raises
                    RuntimeError("fake failure") - for error-path tests
        """
        self.model_id = model_id
        self.precision = precision
        self.responder: Responder = responder or demo_responder
        self.delay_s = max(0.0, float(delay_s))
        self.fail_when = fail_when
        self.calls = []
        self.closed = False
        self.last_hit_limit = []
        self._record_calls = True

    calls: list[list[str]]
    """Every batch of prompts received, for assertions in tests (stays empty
    for the long-lived backend made by :func:`demo_backend`)."""
    closed: bool
    last_hit_limit: list[bool]
    """One flag per prompt of the latest generate() call: True where the
    output has at least ``max_new_tokens`` words - the fake's stand-in for
    "cut off at the limit" (outputs aren't actually shortened). Empty after
    a call that raised."""

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        self.last_hit_limit = []
        if self.closed:
            # A real backend has freed its weights by now; using it is a bug
            # in the caller (v1 kept a label pointing at a freed model).
            raise GenerationError(f"The model {self.model_id} has been unloaded.")
        batch = list(prompts)
        if self._record_calls:
            self.calls.append(batch)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail_when is not None and self.fail_when(batch):
            raise RuntimeError("fake failure")
        outputs = [self.responder(p) for p in batch]
        limit = max(1, int(max_new_tokens))
        self.last_hit_limit = [len(str(text).split()) >= limit for text in outputs]
        return outputs

    def close(self) -> None:
        self.closed = True


def demo_backend(model_id: str = DEMO_REPO) -> FakeBackend:
    """The FakeBackend that demo mode serves.

    It answers like any FakeBackend but doesn't record ``calls``: the model
    manager keeps it for the life of the server, so recording would hold
    every prompt of every run in memory.
    """
    backend = FakeBackend(model_id=model_id, precision="none")
    backend._record_calls = False
    return backend


# ----------------------------------------------------------------------
# Task recognition
# ----------------------------------------------------------------------

# Exact phrases from the built-in templates, most specific first. The
# variable part of a template (a question, a list of categories...) sits
# inside the instruction too, which is why these are checked before the
# loose keywords below.
_PRESET_PHRASES: tuple[tuple[str, str], ...] = (
    ("for each aspect in this list", "aspects"),
    ("answer the question about the text below with only yes or no", "yesno"),
    ("into exactly one of these categories", "categorise"),
    ("extract the named entities", "entities"),
    ("list any personal or sensitive details", "pii"),
    ("replaced by [redacted]", "redact"),
    ("extract the following from the text below", "field"),
    ("most important keywords", "keywords"),
    ("classify the sentiment", "sentiment"),
    ("identify the dominant emotion", "emotion"),
    ("classify the primary intent", "intent"),
    ("rate how urgent", "urgency"),
    ("identify the language", "language"),
    ("topic labels", "topics"),
    ("summarise the text below", "summary"),
)

# Looser cues for user-written templates.
_LOOSE_CUES: tuple[tuple[str, str], ...] = (
    ("aspect", "aspects"),
    ("yes or no", "yesno"),
    ("categor", "categorise"),
    ("redact", "redact"),
    ("entities", "entities"),
    ("personal data", "pii"),
    ("pii", "pii"),
    ("keyword", "keywords"),
    ("sentiment", "sentiment"),
    ("emotion", "emotion"),
    ("intent", "intent"),
    ("urgen", "urgency"),
    ("language", "language"),
    ("topic", "topics"),
    ("summar", "summary"),
)


def _recognise(instruction: str, loose: bool) -> str | None:
    for phrase, task in _PRESET_PHRASES:
        if phrase in instruction:
            return task
    if loose:
        for cue, task in _LOOSE_CUES:
            if cue in instruction:
                return task
    return None


# ----------------------------------------------------------------------
# Lexicons. A trailing "*" means "any word starting with this".
# ----------------------------------------------------------------------


def _wordset(words: str) -> frozenset[str]:
    return frozenset(words.split())


class _Lexicon:
    def __init__(self, words: str) -> None:
        items = words.split()
        self.exact = frozenset(w for w in items if not w.endswith("*"))
        self.prefixes = tuple(w[:-1] for w in items if w.endswith("*"))

    def __contains__(self, token: object) -> bool:
        return isinstance(token, str) and (token in self.exact or token.startswith(self.prefixes))

    def count(self, tokens: Sequence[str]) -> int:
        return sum(1 for t in tokens if t in self)


_POSITIVE = _Lexicon(
    "thank* excellent great good delight* happy pleased love* lovely wonderful fantastic "
    "helpful improve* impress* brilliant perfect* appreciat* easy satisf* recommend* amazing "
    "outstanding superb best glad grateful kind friendly smooth* efficient* resolved quick* "
    "well nice efficient"
)
_NEGATIVE = _Lexicon(
    "unacceptable lost disgrace* complain* rude unprofessional poor* terrible awful bad worst "
    "disappoint* frustrat* angry furious delay* late wrong error* fail* broken problem* "
    "nobody ignor* worse horrible useless hate* upset unhappy slow* mistake* refus* "
    "confus* difficult nightmare ridiculous shocking appalling incorrect missing stress* "
    "worr* outrag* unfair dissatisf* incompetent hardship* underpa* overcharg*"
)
_NEGATORS = _wordset("not no never isn't wasn't aren't don't didn't doesn't cannot can't")

_EMOTIONS: tuple[tuple[str, _Lexicon], ...] = (
    (
        "anger",
        _Lexicon(
            "unacceptable disgrace* furious angry outrag* rude ridiculous appalling livid "
            "annoy* infuriat* complain* fed unprofessional shocking"
        ),
    ),
    (
        "joy",
        _Lexicon(
            "delight* happy pleased glad thank* excellent great wonderful love* thrilled "
            "excit* fantastic brilliant grateful"
        ),
    ),
    (
        "sadness",
        _Lexicon(
            "sad disappoint* unhappy upset heartbroken regret* depress* lonely devastat* lost"
        ),
    ),
    (
        "fear",
        _Lexicon(
            "worr* afraid scared anxious anxiety fear* concern* panic* nervous terrified threat*"
        ),
    ),
    ("surprise", _Lexicon("surpris* unexpected* astonish* amazed suddenly wow shock*")),
    ("disgust", _Lexicon("disgust* revolting vile gross sickening nasty filthy")),
)

_QUESTION_START = _wordset(
    "what when where why how who which whom whose is are can could do does did will "
    "would should may has have"
)
_URGENT = _Lexicon("urgent* emergency immediate* asap critical* danger*")
_SOON = _Lexicon("today tonight tomorrow deadline* overdue now soon quickly expire* final")
_ESCALATE = _Lexicon("mp solicitor* legal ombudsman court lawyer* tribunal escalat*")

_LANGUAGES: tuple[tuple[str, frozenset[str]], ...] = tuple(
    (name, frozenset(words.split()))
    for name, words in (
        (
            "English",
            "the and is of to in that it for you with was my this have please thank are be on not",
        ),
        ("French", "le la les des et est une pour que je vous merci nous pas avec dans sur au du"),
        ("German", "der die das und ist nicht ich sie mit ein eine zu den danke bitte für auf wir"),
        ("Spanish", "el la los las y es que en por para gracias una con muy mi del se lo"),
        ("Italian", "il lo gli di è che per non grazie una sono con del della molto questo"),
        ("Portuguese", "o os as é que não obrigado obrigada uma um para com muito do da em"),
        ("Dutch", "de het een en is van niet ik dat met voor dank bedankt zijn op je"),
        ("Welsh", "y yr ac yn mae diolch chi ar ei gyda ddim hefyd eich ein fy bod"),
    )
)
_SCRIPTS: tuple[tuple[str, str], ...] = (
    ("Ѐ-ӿ", "Russian"),
    ("Ͱ-Ͽ", "Greek"),
    ("؀-ۿ", "Arabic"),
    ("֐-׿", "Hebrew"),
    ("぀-ヿ", "Japanese"),
    ("가-힯", "Korean"),
    ("一-鿿", "Chinese"),
    ("ऀ-ॿ", "Hindi"),
    ("฀-๿", "Thai"),
)

_TOPICS: tuple[tuple[str, _Lexicon], ...] = (
    ("payments", _Lexicon("pay* invoice* refund* fee* charg* billing money cost*")),
    ("claims", _Lexicon("claim*")),
    ("customer service", _Lexicon("service staff call* respon* phone rude helpful inspector*")),
    ("applications", _Lexicon("applic* apply form* deadline* scheme*")),
    ("inspections", _Lexicon("inspect* visit*")),
    ("account details", _Lexicon("address* details correspondence account* contact")),
    ("online services", _Lexicon("portal online website digital login password app")),
    ("land and mapping", _Lexicon("parcel* land field* hectare* mapping map* area ha")),
    ("complaints", _Lexicon("complain* unacceptable disgrace*")),
    ("appeals", _Lexicon("appeal* evidence review*")),
)

_STOPWORDS = _wordset(
    """a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each few
    for from further had has have having he her here hers herself him himself his how i if in
    into is it its itself just me more most my myself no nor not now of off on once only or
    other our ours out over own same she should so some such than that the their theirs them
    then there these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours yourself please thank thanks
    dear regards many much still since being get got let us via per i'm it's i've don't"""
)

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_WEEKDAYS = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
_DATE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?(?:\s+\d{{4}})?"
    rf"|(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}"
    rf"|\d{{1,2}}[/.]\d{{1,2}}[/.]\d{{2,4}}"
    rf"|(?:{_WEEKDAYS}))\b"
)
_PERSON = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Sir|Dame)\.?\s+[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+)?"
)
_ORG = re.compile(
    r"\b(?:[A-Z][\w&'-]*\s+){0,3}?(?:[A-Z][\w&'-]*\s+)"
    r"(?:Ltd|Limited|plc|PLC|Inc|LLC|LLP|Council|Agency|Bank|Department|Trust|Group|"
    r"University|Authority|Ministry|Association|Foundation|Company)\b"
)
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b(?![-/]?\d)")
_ACRONYM_SKIP = frozenset(
    {"MP", "OK", "NONE", "TEXT", "PS", "AM", "PM", "UK", "USA", "EU", "ID", "FAQ"}
)
_PLACE_AFTER = re.compile(
    r"\b(?:in|at|from|near|to|visit|visited)\s+((?:[A-Z][a-z'’-]+)(?:\s+[A-Z][a-z'’-]+){0,2})"
)
_KNOWN_PLACES = _wordset(
    """London Manchester Birmingham Leeds Liverpool Bristol Sheffield Newcastle Cardiff Swansea
    Edinburgh Glasgow Belfast York Oxford Cambridge Exeter Carlisle Kendal Penrith England
    Scotland Wales Ireland Britain Europe France Germany Spain Italy Paris Berlin Madrid Dublin"""
)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE = re.compile(r"(?<!\w)(?:\+44\s?\(?0?\)?\s?|\(?0)\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\w)")
_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b")
_ACCOUNT = re.compile(r"\b\d{6,}\b|\b[A-Z]{1,4}-?\d{4,}\b")
_MONEY = re.compile(
    r"[£$€]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|bn|million|billion))?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:GBP|USD|EUR|pounds|dollars|euros)\b",
    re.IGNORECASE,
)
_AREA = re.compile(r"\b\d+(?:\.\d+)?\s?(?:ha|hectares?|acres?|sq\s?m|m2)\b", re.IGNORECASE)
# Any token with a digit - used right after a cue word ("claim LIB-40517").
_REFERENCE = re.compile(r"\b(?=[A-Z0-9/-]*\d)[A-Z0-9]+(?:[-/][A-Z0-9]+)*\b")
# Without a cue word: letters and digits mixed, or a long digit run.
_REFERENCE_STRICT = re.compile(
    r"\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z0-9]+(?:-[A-Z0-9]+)*\b|\b\d{6,}\b"
)
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# ----------------------------------------------------------------------
# Small text helpers
# ----------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    return [t.lower().replace("’", "'") for t in _WORD.findall(text)]


def _first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def _stable_pick(options: Sequence[str], text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return options[int.from_bytes(digest[:4], "big") % len(options)]


def _polarity(tokens: Sequence[str]) -> tuple[int, int]:
    """(positive hits, negative hits); a negator up to 3 words back flips a hit."""
    pos = neg = 0
    for i, token in enumerate(tokens):
        is_pos, is_neg = token in _POSITIVE, token in _NEGATIVE
        if not (is_pos or is_neg):
            continue
        negated = any(t in _NEGATORS for t in tokens[max(0, i - 3) : i])
        if is_pos != negated:
            pos += 1
        else:
            neg += 1
    return pos, neg


def _sentiment_word(tokens: Sequence[str], allow_mixed: bool = True) -> str:
    pos, neg = _polarity(tokens)
    if pos and neg:
        if allow_mixed:
            return "mixed"
        return "positive" if pos > neg else "negative" if neg > pos else "neutral"
    if pos:
        return "positive"
    if neg:
        return "negative"
    return "neutral"


def _spans(text: str, patterns: Sequence[re.Pattern[str]]) -> list[tuple[int, int]]:
    """Non-overlapping match spans of all patterns, earliest (then longest) first."""
    found = sorted(
        {(m.start(), m.end()) for p in patterns for m in p.finditer(text) if m.end() > m.start()},
        key=lambda s: (s[0], -(s[1] - s[0])),
    )
    out: list[tuple[int, int]] = []
    for start, end in found:
        if not out or start >= out[-1][1]:
            out.append((start, end))
    return out


def _unique(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _content_words(text: str) -> list[str]:
    words = (t.removesuffix("'s") for t in _tokens(text))
    return [w for w in words if len(w) >= 4 and w not in _STOPWORDS and not w[0].isdigit()]


def _between(instruction: str, pattern: str) -> str | None:
    match = re.search(pattern, instruction, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _mentions(text_tokens: Sequence[str], phrase: str, require_all: bool = True) -> bool:
    """All (or any) content words of ``phrase``, crudely stemmed, occur in the text."""
    words = [w for w in _tokens(phrase) if w not in _STOPWORDS] or _tokens(phrase)
    if not words:
        return False
    found = (any(t.startswith(w[:5]) for t in text_tokens) for w in words)
    return all(found) if require_all else any(found)


# ----------------------------------------------------------------------
# Handlers: (text, instruction) -> answer
# ----------------------------------------------------------------------


def _sentiment(text: str, instruction: str) -> str:
    return _sentiment_word(_tokens(text))


def _emotion(text: str, instruction: str) -> str:
    tokens = _tokens(text)
    best, best_hits = "neutral", 0
    for name, lexicon in _EMOTIONS:
        hits = lexicon.count(tokens)
        if hits > best_hits:
            best, best_hits = name, hits
    return best


def _intent(text: str, instruction: str) -> str:
    tokens = _tokens(text)
    pos, neg = _polarity(tokens)
    low = text.lower()
    # "Thanks for the reply, but it's still wrong": what follows the "but" counts.
    tail_pos, tail_neg = _polarity(_tokens(re.split(r"\b(?:but|however)\b", low)[-1]))
    if any(t.startswith("complain") for t in tokens) or neg > pos or tail_neg > tail_pos:
        return "complaint"
    if pos >= 2 and pos > neg:
        return "praise"
    if re.search(r"\b(?:please|kindly)\s+(?:find|see)\s+attached\b", low):
        return "other"
    if re.search(r"\b(?:please|kindly|i would like|i'd like|i need|request)\b", low):
        return "request"
    if "?" in text or (tokens and tokens[0] in _QUESTION_START):
        return "query"
    if pos:
        return "praise"
    return "other"


def _urgency(text: str, instruction: str) -> str:
    tokens = _tokens(text)
    _, neg = _polarity(tokens)
    score = 1
    score += 2 * min(1, _URGENT.count(tokens))
    score += min(1, _SOON.count(tokens))
    score += min(1, _ESCALATE.count(tokens))
    score += (neg >= 1) + (neg >= 3) + ("!" in text)
    return str(max(1, min(5, score)))


def _yes_no(text: str, instruction: str) -> str:
    question = _between(instruction, r"QUESTION:\s*(.*?)\s*$") or instruction
    q_tokens, t_tokens = set(_tokens(question)), _tokens(text)
    pos, neg = _polarity(t_tokens)
    if q_tokens & {
        "complaint",
        "complain",
        "failure",
        "problem",
        "issue",
        "unhappy",
        "negative",
        "angry",
        "dissatisfied",
    }:
        return "Yes" if neg > pos else "No"
    if q_tokens & {"praise", "positive", "happy", "satisfied", "pleased", "compliment"}:
        return "Yes" if pos > neg else "No"
    if q_tokens & {"question", "query", "asking", "ask"}:
        return "Yes" if "?" in text else "No"
    if q_tokens & {"urgent", "urgency", "emergency"}:
        return "Yes" if int(_urgency(text, "")) >= 3 else "No"
    content = [
        w
        for w in _content_words(question)
        if w not in {"text", "does", "mention", "describe", "contain"}
    ]
    hit = any(any(t.startswith(w[:5]) for t in t_tokens) for w in content)
    return "Yes" if hit else "No"


def _categorise(text: str, instruction: str) -> str:
    raw = _between(instruction, r"categories:\s*(.*?)\.\s*Reply with") or _between(
        instruction, r"categories:\s*(.*?)\s*$"
    )
    categories = parse_label_list(raw)
    if not categories:
        return _first_words(text, 3)
    tokens = _tokens(text)
    for require_all in (True, False):
        for category in categories:
            if _mentions(tokens, category, require_all):
                return category
    others = [
        c for c in categories if c.lower() in {"other", "others", "misc", "miscellaneous", "none"}
    ]
    if others:
        return others[0]
    return _stable_pick(categories, text)


def _language(text: str, instruction: str) -> str:
    for block, name in _SCRIPTS:
        if len(re.findall(f"[{block}]", text)) >= 2:
            return name
    tokens = _tokens(text)
    best, best_hits = "English", 0
    for name, words in _LANGUAGES:
        hits = sum(1 for t in tokens if t in words)
        if hits > best_hits:
            best, best_hits = name, hits
    return best


def _entities(text: str, instruction: str) -> str:
    people = _unique([m.group(0) for m in _PERSON.finditer(text)])
    orgs = [re.sub(r"^The\s+", "", m.group(0)) for m in _ORG.finditer(text)]
    orgs += [a for a in _ACRONYM.findall(text) if a not in _ACRONYM_SKIP]
    organisations = _unique(orgs)
    dates = _unique([m.group(0) for m in _DATE.finditer(text)])
    taken = {x.lower() for x in (*people, *organisations, *dates)}
    places = [m.group(1) for m in _PLACE_AFTER.finditer(text)]
    places += [w for w in re.findall(r"\b[A-Z][a-z]+\b", text) if w in _KNOWN_PLACES]
    locations = _unique(
        [
            re.sub(r"^The\s+", "", p)
            for p in places
            if p.lower() not in taken and not re.fullmatch(f"(?:{_MONTHS}|{_WEEKDAYS})", p)
        ]
    )
    return json.dumps(
        {"people": people, "organisations": organisations, "locations": locations, "dates": dates},
        ensure_ascii=False,
    )


def _keywords(text: str, instruction: str) -> str:
    words = _content_words(text)
    if not words:
        return _first_words(text, 3)
    counts = Counter(words)
    order = {w: i for i, w in reversed(list(enumerate(words)))}
    ranked = sorted(counts, key=lambda w: (-counts[w], order[w]))
    return ", ".join(ranked[:5])


_PII_PATTERNS = (_EMAIL, _PHONE, _POSTCODE, _PERSON, _ACCOUNT)


def _pii(text: str, instruction: str) -> str:
    found = _unique([text[a:b] for a, b in _spans(text, _PII_PATTERNS)])
    return json.dumps({"pii": found}, ensure_ascii=False)


def _redact(text: str, instruction: str) -> str:
    out, last = [], 0
    for start, end in _spans(text, _PII_PATTERNS):
        out.append(text[last:start])
        out.append("[REDACTED]")
        last = end
    out.append(text[last:])
    return "".join(out).strip()


def _field(text: str, instruction: str) -> str:
    wanted = (
        _between(
            instruction, r"extract the following from the text below:\s*(.*?)\.\s*If it is not"
        )
        or instruction
    ).lower()
    # (cue words, pattern right after a cue word in the text, pattern anywhere)
    rules: tuple[tuple[tuple[str, ...], re.Pattern[str], re.Pattern[str]], ...] = (
        (("email",), _EMAIL, _EMAIL),
        (("phone", "telephone", "mobile"), _PHONE, _PHONE),
        (("postcode", "post code", "zip"), _POSTCODE, _POSTCODE),
        (
            ("amount", "money", "cost", "price", "fee", "payment", "sum", "value", "£", "$", "€"),
            _MONEY,
            _MONEY,
        ),
        (("area", "hectare", "acre", "size"), _AREA, _AREA),
        (("date", "deadline", "when", "day"), _DATE, _DATE),
        (("name", "person", "who", "inspector", "customer"), _PERSON, _PERSON),
        (
            ("reference", "ref", "number", "id", "claim", "account", "parcel", "code"),
            _REFERENCE,
            _REFERENCE_STRICT,
        ),
    )
    for cues, near_pattern, anywhere in rules:
        if not any(re.search(rf"(?<!\w){re.escape(cue)}s?(?!\w)", wanted) for cue in cues):
            continue
        # Prefer the value right after a word from the request
        # ("the claim number" finds "Claim LIB-40517").
        for word in _content_words(wanted):
            near = re.search(
                rf"\b{re.escape(word)}\w*\b\W{{0,12}}({near_pattern.pattern})",
                text,
                re.IGNORECASE,
            )
            if near:
                return near.group(1).strip()
        # A bare match must not be a piece of a phone number or an email.
        blocked = [] if anywhere in (_PHONE, _EMAIL) else _spans(text, (_PHONE, _EMAIL))
        for match in anywhere.finditer(text):
            if not any(a <= match.start() < b for a, b in blocked):
                return match.group(0).strip()
    return "NONE"


def _aspects(text: str, instruction: str) -> str:
    raw = _between(instruction, r"for each aspect in this list\s*-\s*(.*?)\s+-\s+give") or _between(
        instruction, r"aspects?:\s*(.*?)\s*$"
    )
    aspects = parse_label_list(raw)
    clauses = [
        c for c in re.split(r"[.!?;]+|\bbut\b|\bhowever\b", text, flags=re.IGNORECASE) if c.strip()
    ]
    result: dict[str, str] = {}
    for aspect in aspects:
        hits = [c for c in clauses if _mentions(_tokens(c), aspect)]
        if not hits:
            result[aspect] = "not_mentioned"
        else:
            result[aspect] = _sentiment_word(_tokens(" ".join(hits)), allow_mixed=False)
    return json.dumps(result, ensure_ascii=False)


def _topics(text: str, instruction: str) -> str:
    tokens = _tokens(text)
    scored = [(lexicon.count(tokens), i, name) for i, (name, lexicon) in enumerate(_TOPICS)]
    chosen = [name for hits, _, name in sorted(scored, key=lambda s: (-s[0], s[1])) if hits][:3]
    if not chosen:
        chosen = _keywords(text, instruction).split(", ")[:2]
    return ", ".join(chosen)


def _summary(text: str, instruction: str) -> str:
    sentences = [s for s in _SENTENCE_END.split(" ".join(text.split())) if s.strip()]
    if not sentences:
        return ""
    words = sentences[0].split()
    clipped = " ".join(words[:20]).rstrip(",;:-")
    if clipped and clipped[-1] not in ".!?":
        clipped += "."
    return clipped


_HANDLERS: dict[str, Callable[[str, str], str]] = {
    "sentiment": _sentiment,
    "emotion": _emotion,
    "intent": _intent,
    "urgency": _urgency,
    "yesno": _yes_no,
    "categorise": _categorise,
    "language": _language,
    "entities": _entities,
    "keywords": _keywords,
    "pii": _pii,
    "redact": _redact,
    "field": _field,
    "aspects": _aspects,
    "topics": _topics,
    "summary": _summary,
}
