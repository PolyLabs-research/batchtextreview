"""App shell: page config, styles, navigation and job bookkeeping."""

from __future__ import annotations

import streamlit as st

from batchtextreview import APP_NAME
from batchtextreview.ui import components, nav, styles
from batchtextreview.ui.pages import history, results, run, settings


def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=":material/fact_check:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    styles.inject()
    pages = nav.build(
        {
            "run": run.render,
            "results": results.render,
            "history": history.render,
            "settings": settings.render,
        }
    )
    current = st.navigation(pages, position="top")
    components.note_page(current.title)
    # Before the page: a run that just finished is noticed (toast), and
    # queued toasts show on whichever page opens. Each page's header draws
    # the context strip (components.page_header).
    components.track_jobs()
    current.run()
