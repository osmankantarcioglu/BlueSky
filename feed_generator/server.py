"""
Flask HTTP server — AT Protocol feed endpoints + admin panel.

Endpoints:
  GET  /.well-known/did.json
  GET  /xrpc/app.bsky.feed.describeFeedGenerator
  GET  /xrpc/app.bsky.feed.getFeedSkeleton
  GET  /health

  /admin/*  — feed management UI (requires ADMIN_SECRET)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, jsonify, request, abort
from config.settings import (
    FEED_DOMAIN,
    DATABASE_PATH,
)

os.makedirs(os.path.dirname(os.path.abspath(DATABASE_PATH)), exist_ok=True)

from database.models import db, Post, Feed, create_tables

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "bluesky-feed-secret-2026")

# Bootstrap tables once at startup
with app.app_context():
    if db.is_closed():
        db.connect()
    create_tables()
    db.close()

# Register admin blueprint
from admin import admin_bp
app.register_blueprint(admin_bp)


# ---------------------------------------------------------------------------
# DB connection middleware
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
# DID document
# ---------------------------------------------------------------------------

@app.route("/.well-known/did.json")
def did_document():
    return jsonify({
        "@context": ["https://www.w3.org/ns/did/v1"],
        "id": f"did:web:{FEED_DOMAIN}",
        "service": [{
            "id": "#bsky_fg",
            "type": "BskyFeedGenerator",
            "serviceEndpoint": f"https://{FEED_DOMAIN}",
        }],
    })


# ---------------------------------------------------------------------------
# describeFeedGenerator — enumerate active feeds from DB
# ---------------------------------------------------------------------------

@app.route("/xrpc/app.bsky.feed.describeFeedGenerator")
def describe_feed_generator():
    active_feeds = (
        Feed.select(Feed.at_uri)
        .where(Feed.is_active == True, Feed.at_uri.is_null(False))
    )
    return jsonify({
        "did": f"did:web:{FEED_DOMAIN}",
        "feeds": [{"uri": f.at_uri} for f in active_feeds],
    })


# ---------------------------------------------------------------------------
# getFeedSkeleton — route by Feed.at_uri, fall back to domain_label for legacy
# ---------------------------------------------------------------------------

@app.route("/xrpc/app.bsky.feed.getFeedSkeleton")
def get_feed_skeleton():
    from feed_generator.feed_logic import get_posts_for_feed, get_feed_posts

    feed_uri = request.args.get("feed", "")
    cursor   = request.args.get("cursor", None)
    limit    = int(request.args.get("limit", 30))

    # Try dynamic Feed lookup first
    feed = Feed.get_or_none(Feed.at_uri == feed_uri, Feed.is_active == True)

    if feed:
        posts, next_cursor = get_posts_for_feed(feed.id, cursor=cursor, limit=limit)
    else:
        # Legacy fallback using hardcoded env-var URIs
        from config.settings import FEED_URI_POLITICS, FEED_URI_SCIENCE
        _legacy_map = {
            FEED_URI_POLITICS: "politics",
            FEED_URI_SCIENCE:  "science",
        }
        domain = _legacy_map.get(feed_uri)
        if domain is None:
            abort(400, description=f"Unknown feed URI: {feed_uri}")
        posts, next_cursor = get_feed_posts(domain=domain, cursor=cursor, limit=limit)

    response = {"feed": [{"post": p.uri} for p in posts]}
    if next_cursor:
        response["cursor"] = next_cursor
    return jsonify(response)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    total = Post.select().count()
    feed_count = Feed.select().where(Feed.is_active == True).count()
    return jsonify({"status": "ok", "total_posts": total, "active_feeds": feed_count})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"Starting feed server on :{port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
