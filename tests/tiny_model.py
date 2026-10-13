"""Build tiny random Llama models on disk for the transformers-backend tests.

The Hugging Face Hub is never contacted: the tokenizer is assembled with the
``tokenizers`` library from a fixed word list and the model is a 2-layer
Llama with random (seeded) weights. Everything is small enough to build and
load in well under a second on a CPU.

The tokenizer is word-level (one token per word or punctuation run, joined
with spaces on decode), so re-encoding a decoded output gives back exactly
as many tokens as were generated - which lets tests check
``max_new_tokens`` precisely.
"""

from __future__ import annotations

from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<s>{{ message['role'] }}: {{ message['content'] }}</s>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<s>assistant:{% endif %}"
)

WORDS = [
    "user",
    "assistant",
    "system",
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "with",
    "from",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "my",
    "your",
    "our",
    "their",
    "me",
    "us",
    "them",
    "classify",
    "sentiment",
    "emotion",
    "intent",
    "text",
    "below",
    "as",
    "positive",
    "negative",
    "neutral",
    "mixed",
    "reply",
    "with",
    "exactly",
    "one",
    "word",
    "only",
    "answer",
    "question",
    "yes",
    "no",
    "category",
    "categories",
    "summary",
    "summarise",
    "keyword",
    "keywords",
    "extract",
    "entities",
    "people",
    "place",
    "date",
    "please",
    "thank",
    "thanks",
    "service",
    "claim",
    "payment",
    "parcel",
    "form",
    "deadline",
    "application",
    "portal",
    "update",
    "address",
    "great",
    "terrible",
    "good",
    "bad",
    "happy",
    "sad",
    "angry",
    "quick",
    "slow",
    "late",
    "lost",
    "found",
    "never",
    "always",
    "again",
    "today",
    "tomorrow",
    "week",
    "month",
    "very",
    "really",
    "not",
    "so",
    "too",
    "more",
    "most",
    "less",
    "least",
    "new",
    "old",
    "big",
    "small",
    "first",
    "last",
    "next",
    "time",
    "hello",
    "world",
    "model",
    "batch",
    "prompt",
    "output",
    "token",
    "test",
    "short",
    "long",
    "longer",
    "longest",
    "medium",
]
PUNCTUATION = (":", ".", ",", "?", "!", "-", "'", '"', "(", ")")


def build_tokenizer(*, chat_template: bool | str = True, pad_token: bool = True):
    """A ``PreTrainedTokenizerFast`` with bos/eos/unk (and optionally pad).

    ``chat_template``: True for :data:`CHAT_TEMPLATE`, False for none, or
    the Jinja template to use.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    specials = [BOS, EOS, UNK] + ([PAD] if pad_token else [])
    vocab: dict[str, int] = {}
    for token in (*specials, *WORDS, *PUNCTUATION):
        vocab.setdefault(token, len(vocab))

    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    # Only used when add_special_tokens=True (the no-chat-template path).
    core.post_processor = processors.TemplateProcessing(
        single=f"{BOS} $A", pair=f"{BOS} $A {BOS} $B", special_tokens=[(BOS, vocab[BOS])]
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        bos_token=BOS,
        eos_token=EOS,
        unk_token=UNK,
        pad_token=PAD if pad_token else None,
    )
    if chat_template:
        tokenizer.chat_template = chat_template if isinstance(chat_template, str) else CHAT_TEMPLATE
    return tokenizer


def build_tiny_model(
    dest: Path,
    *,
    chat_template: bool | str = True,
    pad_token: bool = True,
    seed: int = 0,
) -> Path:
    """Write a tiny random Llama checkpoint + tokenizer to ``dest``.

    The output projection is scaled up so the gap between the best and the
    second-best logit dwarfs float rounding noise; greedy decoding is then
    identical whether a prompt is generated alone or left-padded in a batch.
    """
    import torch
    from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(chat_template=chat_template, pad_token=pad_token)

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=False,
    )
    torch.manual_seed(seed)
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        model.lm_head.weight.mul_(400.0)
    model.generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.save_pretrained(dest)
    tokenizer.save_pretrained(dest)
    return dest
