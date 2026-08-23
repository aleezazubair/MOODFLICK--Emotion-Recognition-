"""
Vercel entry point.

Vercel serves the WSGI object it finds in this file; the application itself
lives in app.py at the project root so that `python app.py` still works locally.

The rewrite in vercel.json points every request at this function, which means
the path the function receives is the function's own (/api/index) rather than
the one the visitor asked for -- so Flask matches no route and answers 404 to
everything. The rewrite therefore carries the real path in a __path query
parameter, which is passed through untouched, and the middleware below puts it
back before Flask routes on it.
"""

import os
import sys
from urllib.parse import parse_qs, urlencode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402  (must follow the sys.path line above)


class RestoreOriginalPath:
    """Undo the rewrite's effect on PATH_INFO."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        params = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)

        if "__path" in params:
            environ["PATH_INFO"] = params.pop("__path")[0] or "/"
            # Hand the request its own query string back, minus our parameter.
            environ["QUERY_STRING"] = urlencode(params, doseq=True)
        elif not environ.get("PATH_INFO"):
            # Belt and braces: an empty path is not a route Flask can match.
            environ["PATH_INFO"] = "/"

        return self.wsgi_app(environ, start_response)


app.wsgi_app = RestoreOriginalPath(app.wsgi_app)

__all__ = ["app"]
