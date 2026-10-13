"""Page registry, so pages can link to each other without import cycles.

Usage::

    from batchtextreview.ui import nav

    st.switch_page(nav.page("results"))
    st.page_link(nav.page("history"), label="Open history")
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st
from streamlit.navigation.page import StreamlitPage

# name -> (title, icon, url_path)
_DEFS: dict[str, tuple[str, str, str]] = {
    "run": ("Run", ":material/play_circle:", "run"),
    "results": ("Results", ":material/table_view:", "results"),
    "history": ("History", ":material/leaderboard:", "history"),
    "settings": ("Models & settings", ":material/tune:", "settings"),
}

_PAGES: dict[str, StreamlitPage] = {}


def build(renderers: dict[str, Callable[[], None]]) -> list[StreamlitPage]:
    """Create the st.Page objects (in navigation order) and remember them."""
    pages = []
    for name, (title, icon, url_path) in _DEFS.items():
        page = st.Page(
            renderers[name], title=title, icon=icon, url_path=url_path, default=(name == "run")
        )
        _PAGES[name] = page
        pages.append(page)
    return pages


def page(name: str) -> StreamlitPage:
    """The page object for ``name`` ("run", "results", "history", "settings")."""
    return _PAGES[name]
