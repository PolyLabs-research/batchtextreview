"""The little CSS the theme config can't express. Keep this short."""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* Content starts just below the 60px top navigation bar (Streamlit's own
   default leaves a much larger gap). */
.block-container { padding-top: 4rem; padding-bottom: 3rem; }
/* The navigation's first link lines up with the content's left edge. */
[data-testid="stToolbar"] .rc-overflow { padding-left: calc(5rem - 8px); }
/* Section headers inside bordered cards: no extra top margin. */
[data-testid="stVerticalBlockBorderWrapper"] h3 { margin-top: 0; }
/* Prompt preview and code blocks wrap instead of scrolling sideways. */
[data-testid="stCode"] pre { white-space: pre-wrap; word-break: break-word; }
/* Captions: Streamlit fades them to 60% opacity, 4.1:1 on white. 75% keeps
   them muted at 6.5:1 (and above 6:1 in dark mode). */
[data-testid="stCaptionContainer"] { opacity: .75; }
[data-testid="stTable"] th { color: #4B5563; }
/* Inline code in captions and tables: Streamlit's 0.75em is 10.5px. */
[data-testid="stTable"] code, [data-testid="stCaptionContainer"] code { font-size: .8125rem; }
/* Numbers that tick (progress counts, KPIs) keep their width. */
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
/* components.empty_state: a large, muted, decorative icon. */
.btr-empty-icon {
  font-family: "Material Symbols Rounded"; font-size: 2.5rem; line-height: 1;
  text-align: center; opacity: .45; user-select: none;
}
/* components.color_key: one colour key shared by side-by-side charts. */
.btr-key { display: flex; flex-wrap: wrap; gap: .25rem 1.25rem; font-size: .875rem; }
.btr-key span { display: inline-flex; align-items: center; gap: .4rem; }
.btr-key i { width: .625rem; height: .625rem; border-radius: 50%; display: inline-block; }
/* Keyboard hint inside primary buttons: a small key cap, not part of the
   label; hidden while the button is disabled. */
button[kind^="primary"] kbd {
  opacity: .85; font-size: .75rem; margin-left: .5rem; padding: 0 .3rem;
  border: 1px solid currentColor; border-radius: .25rem;
}
button:disabled kbd { display: none; }
/* Run page: the plan card stays in view while the long left column scrolls
   (wide, tall windows only, so the card itself is never cut off) - but not
   while it holds the progress panel, which can be taller than the window. */
@media (min-width: 1100px) and (min-height: 820px) {
  [data-testid="stColumn"]:has(.st-key-btr_plan):not(:has(.st-key-btr_progress)) {
    position: sticky; top: 4.5rem; align-self: flex-start;
  }
}
/* Keyboard focus: a solid 2px ring (6.3:1), distinct from the "current
   page" pill. Streamlit's own ring is a 2.4:1 shadow; !important because its
   component styles are injected after this sheet. */
.stApp [data-testid="stTopNavLink"]:focus-visible, .stApp button:focus-visible,
.stApp [role="tab"]:focus-visible, .stApp a:focus-visible {
  outline: 2px solid #4F46E5 !important; outline-offset: 2px; box-shadow: none !important;
}
/* The dark theme's light indigo keeps primary-coloured text readable on the
   dark background, but white text on it is only 3:1: filled primary
   elements get near-black text (6.3:1). The toolbar is minimal (no theme
   menu), so the theme always follows the OS and this query matches it. */
@media (prefers-color-scheme: dark) {
  button[kind^="primary"]:not(:disabled),
  [data-testid="stMultiSelectTagsContainer"] [data-tag] { color: #0D1117; }
  [data-testid="stTable"] th { color: #AEB8C4; }
  .stApp [data-testid="stTopNavLink"]:focus-visible, .stApp button:focus-visible,
  .stApp [role="tab"]:focus-visible, .stApp a:focus-visible { outline-color: #A5B4FC !important; }
}
</style>
"""


def inject() -> None:
    """Add the app-wide CSS (once per rerun; Streamlit dedupes identical HTML)."""
    st.html(_CSS)
