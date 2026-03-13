"""
Flask HTTP server exposing the Bluesky custom feed endpoints.

Endpoints:
  GET /.well-known/did.json          — DID document (required by AT Protocol)
  GET /xrpc/app.bsky.feed.describeFeedGenerator   — list feeds this service offers
  GET /xrpc/app.bsky.feed.getFeedSkeleton         — return ranked post skeletons
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, jsonify, request, abort
from database.models import db, Post
from feed_generator.feed_logic import get_feed_posts
from config.settings import (
    FEED_DOMAIN,
    FEED_URI_POLITICS,
    FEED_URI_SCIENCE,
)

app = Flask(__name__)

# Map feed URI → domain label used in the database
FEED_MAP = {
    FEED_URI_POLITICS: "politics",
    FEED_URI_SCIENCE:  "science",
}


# ---------------------------------------------------------------------------
# Middleware: open / close DB connection per request
# ---------------------------------------------------------------------------

@app.before_request
def _open_db():
    if db.is_closed():
        db.connect()


@app.teardown_request
def _close_db(exc):
    if not db.is_closed():
        db.close()


# ---------------------------------------------------------------------------
# DID document — required so Bluesky can resolve the feed generator service
# ---------------------------------------------------------------------------

@app.route("/.well-known/did.json")
def did_document():
    return jsonify({
        "@context": ["https://www.w3.org/ns/did/v1"],
        "id": f"did:web:{FEED_DOMAIN}",
        "service": [
            {
                "id": "#bsky_fg",
                "type": "BskyFeedGenerator",
                "serviceEndpoint": f"https://{FEED_DOMAIN}",
            }
        ],
    })


# ---------------------------------------------------------------------------
# describeFeedGenerator — tells Bluesky which feeds this service provides
# ---------------------------------------------------------------------------

@app.route("/xrpc/app.bsky.feed.describeFeedGenerator")
def describe_feed_generator():
    return jsonify({
        "did": f"did:web:{FEED_DOMAIN}",
        "feeds": [
            {"uri": FEED_URI_POLITICS},
            {"uri": FEED_URI_SCIENCE},
        ],
    })


# ---------------------------------------------------------------------------
# getFeedSkeleton — the main feed endpoint called by Bluesky clients
# ---------------------------------------------------------------------------

@app.route("/xrpc/app.bsky.feed.getFeedSkeleton")
def get_feed_skeleton():
    feed_uri = request.args.get("feed", "")
    cursor   = request.args.get("cursor", None)
    limit    = int(request.args.get("limit", 30))

    domain = FEED_MAP.get(feed_uri)
    if domain is None:
        abort(400, description=f"Unknown feed URI: {feed_uri}")

    posts, next_cursor = get_feed_posts(domain=domain, cursor=cursor, limit=limit)

    skeleton = [{"post": p.uri} for p in posts]

    response = {"feed": skeleton}
    if next_cursor:
        response["cursor"] = next_cursor

    return jsonify(response)


# ---------------------------------------------------------------------------
# Health check — useful for deployment monitoring
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    total = Post.select().count()
    return jsonify({"status": "ok", "total_posts": total})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"Starting feed server on port {port} (debug={debug})")
    print(f"  Politics feed : {FEED_URI_POLITICS}")
    print(f"  Science feed  : {FEED_URI_SCIENCE}")
    app.run(host="0.0.0.0", port=port, debug=debug)
