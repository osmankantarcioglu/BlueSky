"""
Admin Blueprint — feed management UI (no authentication required).

Routes:
  GET  /admin                      — dashboard (feed list + stats)
  GET  /admin/feeds/new            — new feed form
  POST /admin/feeds/generate       — AJAX: LLM seed preview (no DB save)
  POST /admin/feeds                — create feed (LLM + centroid + Bluesky)
  GET  /admin/feeds/<id>           — feed detail (keywords, sentences, posts)
  POST /admin/feeds/<id>/toggle    — activate / deactivate
  POST /admin/feeds/<id>/rebuild   — rebuild centroid from existing sentences
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import (
    Blueprint, render_template, request, jsonify,
    redirect, url_for, flash, abort,
)
from database.models import Feed, Post

admin_bp = Blueprint(
    "admin",
    __name__,
    url_prefix="/admin",
    template_folder="templates",
)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@admin_bp.route("/")
def dashboard():
    feeds = list(Feed.select().order_by(Feed.created_at.desc()))
    stats = {}
    for feed in feeds:
        stats[feed.id] = {
            "post_count": Post.select().where(Post.feed_id == feed.id).count(),
        }
    return render_template("dashboard.html", feeds=feeds, stats=stats)


# ---------------------------------------------------------------------------
# New feed — form + creation
# ---------------------------------------------------------------------------

@admin_bp.route("/feeds/new")
def feed_new():
    return render_template("feed_new.html")


@admin_bp.route("/feeds/generate", methods=["POST"])
def feed_generate():
    """AJAX endpoint: LLM preview (no DB writes)."""
    data = request.get_json(silent=True) or {}
    topic    = data.get("topic", "").strip()
    language = data.get("language", "tr")

    if not topic:
        return jsonify({"error": "topic is required"}), 400

    try:
        from admin.services import generate_seeds_preview
        result = generate_seeds_preview(topic=topic, language=language)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@admin_bp.route("/feeds", methods=["POST"])
def feed_create():
    """Create feed from form submission (LLM + centroid + optional Bluesky)."""
    form = request.form

    feed_id        = form.get("feed_id", "").strip()
    display_name   = form.get("display_name", "").strip()
    description    = form.get("description", "").strip()
    language       = form.get("language", "tr")
    topic          = form.get("topic", "").strip()
    emb_model      = form.get("embedding_model", "berturk")
    threshold      = float(form.get("similarity_threshold", "0.30"))
    publish_bsky   = form.get("publish_bluesky") == "on"

    raw_keywords   = form.get("keywords_json", "").strip()
    raw_sentences  = form.get("sentences_json", "").strip()

    keywords  = json.loads(raw_keywords)  if raw_keywords  else None
    sentences = json.loads(raw_sentences) if raw_sentences else None

    if not feed_id or not display_name or not topic:
        flash("feed_id, display_name ve topic zorunludur.", "danger")
        return redirect(url_for("admin.feed_new"))

    try:
        from admin.services import create_feed
        feed = create_feed(
            feed_id=feed_id,
            display_name=display_name,
            description=description,
            language=language,
            topic=topic,
            embedding_model=emb_model,
            similarity_threshold=threshold,
            keywords=keywords,
            seed_sentences=sentences,
            publish_to_bluesky=publish_bsky,
        )
        flash(f"Feed '{feed.display_name}' başarıyla oluşturuldu!", "success")
        return redirect(url_for("admin.feed_detail", feed_id=feed.id))
    except Exception as exc:
        flash(f"Hata: {exc}", "danger")
        return redirect(url_for("admin.feed_new"))


# ---------------------------------------------------------------------------
# Feed detail
# ---------------------------------------------------------------------------

@admin_bp.route("/feeds/<int:feed_id>")
def feed_detail(feed_id: int):
    feed = Feed.get_or_none(Feed.id == feed_id)
    if not feed:
        abort(404)
    recent_posts = (
        Post.select()
        .where(Post.feed_id == feed_id)
        .order_by(Post.created_at.desc())
        .limit(20)
    )
    return render_template("feed_detail.html", feed=feed, posts=recent_posts)


# ---------------------------------------------------------------------------
# Toggle active
# ---------------------------------------------------------------------------

@admin_bp.route("/feeds/<int:feed_id>/toggle", methods=["POST"])
def feed_toggle(feed_id: int):
    feed = Feed.get_or_none(Feed.id == feed_id)
    if not feed:
        abort(404)
    feed.is_active = not feed.is_active
    feed.touch()
    status = "aktif" if feed.is_active else "pasif"
    flash(f"Feed '{feed.display_name}' artık {status}.", "info")
    return redirect(url_for("admin.feed_detail", feed_id=feed_id))


# ---------------------------------------------------------------------------
# Rebuild centroid
# ---------------------------------------------------------------------------

@admin_bp.route("/feeds/<int:feed_id>/rebuild", methods=["POST"])
def feed_rebuild(feed_id: int):
    feed = Feed.get_or_none(Feed.id == feed_id)
    if not feed:
        abort(404)
    try:
        from nlp.pipeline import MultiPipeline
        MultiPipeline.build_centroid_for_feed(feed)
        flash("Centroid yeniden oluşturuldu.", "success")
    except Exception as exc:
        flash(f"Centroid oluşturma hatası: {exc}", "danger")
    return redirect(url_for("admin.feed_detail", feed_id=feed_id))
