"""Streamlit entry point bundled with the package (used by ``batchtextreview ui``).

Kept inside the package so the app also starts from a regular (non-editable)
install, where the repository's ``app.py`` doesn't exist.
"""

from batchtextreview.ui.main import main

main()
