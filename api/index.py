"""
Vercel entry point.

Vercel serves the WSGI object it finds in this file; the application itself
lives in app.py at the project root so that `python app.py` still works locally.

Routing note. vercel.json rewrites every request to this function, and the path
the function then receives is not the one the visitor asked for -- which made
Flask answer 404 to everything, /api/health included. The rewrite therefore
appends the real path to the function's own:

    /detector  ->  /api/index/detector

Whether the platform hands the function that whole path or only the part after
/api/index, stripping the prefix below leaves the path Flask needs. A __path
query parameter is honoured too, in case a future config passes it that way.
"""

import os
import sys
from urllib.parse import parse_qs, urlencode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402  (must follow the sys.path line above)
from flask import jsonify, request  # noqa: E402

FUNCTION_PREFIX = "/api/index"


class RestoreOriginalPath:
    """Undo the rewrite's effect on PATH_INFO, whatever shape it arrives in."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        # Kept for the 404 diagnostic below, which is the only way to see what
        # the platform actually sent.
        environ["moodflick.raw_path"] = environ.get("PATH_INFO", "")
        environ["moodflick.raw_query"] = environ.get("QUERY_STRING", "")

        params = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)
        path = environ.get("PATH_INFO", "")

        if "__path" in params:
            environ["PATH_INFO"] = params.pop("__path")[0] or "/"
            environ["QUERY_STRING"] = urlencode(params, doseq=True)
        elif path == FUNCTION_PREFIX or path.startswith(FUNCTION_PREFIX + "/"):
            environ["PATH_INFO"] = path[len(FUNCTION_PREFIX):] or "/"
        elif not path:
            environ["PATH_INFO"] = "/"

        return self.wsgi_app(environ, start_response)


app.wsgi_app = RestoreOriginalPath(app.wsgi_app)


@app.errorhandler(404)
def report_unmatched_path(error):
    """Say what the app was actually asked for.

    Registered here rather than in app.py, so this applies only on Vercel. A
    plain HTML 404 gives no way to tell a genuinely missing page from the path
    arriving mangled; this makes the difference visible. Only routing fields are
    exposed -- no environment, no secrets. Safe to delete once routing is
    settled.
    """
    return jsonify({
        "error": "No route matched.",
        "routed_on": request.path,
        "path_as_received": request.environ.get("moodflick.raw_path"),
        "query_as_received": request.environ.get("moodflick.raw_query"),
        "script_name": request.environ.get("SCRIPT_NAME"),
        "method": request.method,
        "known_routes": sorted(
            str(rule) for rule in app.url_map.iter_rules()
        ),
    }), 404


__all__ = ["app"]
