"""
TMDB catalogue for MoodFlick.

The app ships with a hand-written catalogue in the page itself, which is what
runs when no API key is configured. With a key, the same shelves are filled
from TMDB instead, which buys three things the static list cannot: real poster
art, official trailers, and per-country streaming availability.

Nothing here streams a film. TMDB serves metadata and, through JustWatch,
says which services carry a title in a given country -- so the app can link
straight to it rather than guessing with a search URL.

Configure with an environment variable (free key from
https://www.themoviedb.org/settings/api):

    TMDB_API_KEY=...        required to enable any of this
    TMDB_REGION=PK          country for streaming availability, default PK

Requests go through urllib rather than requests, to avoid adding a dependency
to a bundle that has a size limit to respect.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

API_KEY = os.environ.get("TMDB_API_KEY", "").strip()
REGION = os.environ.get("TMDB_REGION", "PK").strip().upper()

BASE_URL = "https://api.themoviedb.org/3"
POSTER_URL = "https://image.tmdb.org/t/p/w500"
BACKDROP_URL = "https://image.tmdb.org/t/p/w780"

TIMEOUT = 8
CACHE_TTL = 60 * 60 * 6  # TMDB data barely moves; a long TTL keeps us well
CACHE_LIMIT = 256        # inside the rate limit and off a small instance's CPU

# MoodFlick's shelves are moods, TMDB's vocabulary is genres. Each mood maps to
# the genres that actually deliver that feeling, plus the accent colour the page
# already uses for that shelf.
MOOD_GENRES = {
    "Happy":      {"genres": [35, 10751],   "color": "#FFD93D"},
    "Sad":        {"genres": [18],          "color": "#6BB5FF"},
    "Excited":    {"genres": [28, 12],      "color": "#FF6B35"},
    "Scared":     {"genres": [27, 53],      "color": "#B06FFF"},
    "Romantic":   {"genres": [10749],       "color": "#FF6B9D"},
    "Inspired":   {"genres": [18, 36, 99],  "color": "#FFB347"},
    "Chill":      {"genres": [16, 10402],   "color": "#87CEEB"},
    "Mind-blown": {"genres": [878, 9648],   "color": "#52FFB8"},
}

MOOD_EMOJI = {
    "Happy": "\U0001F604", "Sad": "\U0001F622", "Excited": "\U0001F525",
    "Scared": "\U0001F47B", "Romantic": "\U0001F495", "Inspired": "\u2728",
    "Chill": "\U0001F60C", "Mind-blown": "\U0001F92F",
}

_cache = {}
_cache_lock = threading.Lock()


def is_configured():
    """Whether an API key is present. Without one the page keeps its own list."""
    return bool(API_KEY)


def _cached_get(path, params):
    """GET a TMDB endpoint, memoised. Returns None on any failure."""
    key = path + "?" + urllib.parse.urlencode(sorted(params.items()))
    now = time.time()

    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1]

    query = dict(params, api_key=API_KEY)
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(query)}"

    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        print(f"TMDB request failed for {path}: {exc}")
        return None

    with _cache_lock:
        if len(_cache) >= CACHE_LIMIT:
            # Cheapest useful eviction: drop the oldest entry.
            oldest = min(_cache, key=lambda k: _cache[k][0])
            del _cache[oldest]
        _cache[key] = (now, payload)

    return payload


def _genre_names():
    """TMDB genre ids -> names, so cards can show 'Horror' not '27'."""
    payload = _cached_get("/genre/movie/list", {"language": "en-US"})
    if not payload:
        return {}
    return {g["id"]: g["name"] for g in payload.get("genres", [])}


def _shape(raw, mood, names):
    """Convert a TMDB result into the shape the page's cards already expect."""
    genre_ids = raw.get("genre_ids") or [g["id"] for g in raw.get("genres", [])]
    release = raw.get("release_date") or ""

    return {
        "id": raw["id"],
        "title": raw.get("title") or raw.get("name") or "Untitled",
        "year": int(release[:4]) if release[:4].isdigit() else None,
        "rating": round(float(raw.get("vote_average") or 0), 1),
        "genres": [names.get(gid) for gid in genre_ids if names.get(gid)],
        "mood": [mood] if mood else [],
        "actors": [],  # only present on the details endpoint
        "overview": raw.get("overview") or "No synopsis available.",
        "poster": MOOD_EMOJI.get(mood, "\U0001F3AC"),  # fallback if art is missing
        "posterUrl": POSTER_URL + raw["poster_path"] if raw.get("poster_path") else None,
        "backdropUrl": BACKDROP_URL + raw["backdrop_path"] if raw.get("backdrop_path") else None,
        "color": MOOD_GENRES.get(mood, {}).get("color", "#22e0c8"),
    }


def movies_for_mood(mood, limit=18):
    """Popular, well-rated titles matching a mood's genres."""
    config = MOOD_GENRES.get(mood)
    if not config or not is_configured():
        return []

    payload = _cached_get("/discover/movie", {
        "with_genres": ",".join(str(g) for g in config["genres"]),
        "sort_by": "popularity.desc",
        # A vote floor keeps obscure titles with a lone 10/10 off the shelves.
        "vote_count.gte": 300,
        "language": "en-US",
        "include_adult": "false",
        "page": 1,
    })
    if not payload:
        return []

    names = _genre_names()
    return [_shape(raw, mood, names) for raw in payload.get("results", [])[:limit]]


def movie_details(movie_id):
    """Full record for one title: cast, trailer, and where it can be watched."""
    if not is_configured():
        return None

    payload = _cached_get(f"/movie/{movie_id}", {
        "language": "en-US",
        "append_to_response": "videos,credits,watch/providers",
    })
    if not payload:
        return None

    details = _shape(payload, None, _genre_names())
    details["runtime"] = payload.get("runtime")
    details["tagline"] = payload.get("tagline") or ""

    cast = payload.get("credits", {}).get("cast", [])
    details["actors"] = [person["name"] for person in cast[:6]]

    # Prefer an official YouTube trailer; fall back to any YouTube clip.
    videos = [
        v for v in payload.get("videos", {}).get("results", [])
        if v.get("site") == "YouTube"
    ]
    trailer = next(
        (v for v in videos if v.get("type") == "Trailer" and v.get("official")),
        next((v for v in videos if v.get("type") == "Trailer"),
             videos[0] if videos else None),
    )
    details["trailerKey"] = trailer["key"] if trailer else None

    # JustWatch data, via TMDB. "flatrate" is subscription streaming; buy and
    # rent are listed separately so the page can be honest about which it is.
    region = payload.get("watch/providers", {}).get("results", {}).get(REGION, {})
    details["watch"] = {
        "region": REGION,
        "link": region.get("link"),
        "stream": [p["provider_name"] for p in region.get("flatrate", [])],
        "rent": [p["provider_name"] for p in region.get("rent", [])],
        "buy": [p["provider_name"] for p in region.get("buy", [])],
    }

    return details


def catalogue(per_mood=18):
    """Every shelf in one list, deduplicated.

    The page filters one global array, and a title can legitimately sit under
    more than one mood, so entries are merged rather than repeated: the same
    film keeps a single card carrying every mood it belongs to.
    """
    if not is_configured():
        return []

    merged = {}
    for mood in MOOD_GENRES:
        for movie in movies_for_mood(mood, per_mood):
            existing = merged.get(movie["id"])
            if existing:
                if mood not in existing["mood"]:
                    existing["mood"].append(mood)
            else:
                merged[movie["id"]] = movie

    return sorted(merged.values(), key=lambda m: m["rating"], reverse=True)
