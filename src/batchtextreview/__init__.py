"""Batch Text Review: run prompts over a text corpus with local LLMs."""

from batchtextreview._env import bootstrap as _bootstrap

_bootstrap()

__version__ = "2.0.0"
APP_NAME = "Batch Text Review"

__all__ = ["APP_NAME", "__version__"]
