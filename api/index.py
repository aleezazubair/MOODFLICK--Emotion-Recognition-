"""
Vercel entry point.

Vercel serves the WSGI object it finds in this file; the application itself
lives in app.py at the project root so that `python app.py` still works locally.
vercel.json rewrites every path here and lists the files this needs bundled.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402  (must follow the sys.path line above)

__all__ = ["app"]
