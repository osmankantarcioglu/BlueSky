"""
Publish (or update) the feed generators on Bluesky.

This script creates two app.bsky.feed.generator records on your account:
  - turkiye-siyaset  (Turkish Politics feed)
  - turkiye-bilim    (Turkish Science feed)

Run once to register the feeds, then again whenever you update
the display name or description.

Usage (from project root):
    python scripts/publish_feed.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atproto import Client, models
from config.settings import BSKY_HANDLE, BSKY_APP_PASSWORD, FEED_DOMAIN

# Feed definitions: (rkey, display_name, description)
FEEDS = [
    (
        "turkiye-siyaset",
        "Türkiye Siyaset",
        "Türkiye siyasetine dair Bluesky paylaşımları. "
        "BERTurk tabanlı NLP ile otomatik filtrelenir.",
    ),
    (
        "turkiye-bilim",
        "Türkiye Bilim",
        "Türkiye'den bilim, akademi ve araştırma paylaşımları. "
        "BERTurk tabanlı NLP ile otomatik filtrelenir.",
    ),
]


def publish_feed(client: Client, rkey: str, display_name: str, description: str) -> str:
    """Create or update a single feed generator record. Returns the feed URI."""
    did = client.me.did

    feed_record = models.AppBskyFeedGenerator.Record(
        did=f"did:web:{FEED_DOMAIN}",
        display_name=display_name,
        description=description,
        created_at=client.get_current_time_iso(),
    )

    response = client.com.atproto.repo.put_record(
        models.ComAtprotoRepoPutRecord.Data(
            repo=did,
            collection="app.bsky.feed.generator",
            rkey=rkey,
            record=feed_record,
        )
    )

    feed_uri = f"at://{did}/app.bsky.feed.generator/{rkey}"
    return feed_uri


def main():
    if not BSKY_HANDLE or not BSKY_APP_PASSWORD:
        print("ERROR: Set BSKY_HANDLE and BSKY_APP_PASSWORD in your .env file.")
        sys.exit(1)

    client = Client()
    client.login(BSKY_HANDLE, BSKY_APP_PASSWORD)
    print(f"Logged in as {BSKY_HANDLE} (DID: {client.me.did})")
    print(f"Feed service domain: did:web:{FEED_DOMAIN}\n")

    for rkey, display_name, description in FEEDS:
        print(f"Publishing: {display_name} ({rkey})...")
        uri = publish_feed(client, rkey, display_name, description)
        print(f"  Published: {uri}")

    print("\nDone. Update FEED_URI_POLITICS and FEED_URI_SCIENCE in .env with:")
    print(f"  FEED_URI_POLITICS=at://{client.me.did}/app.bsky.feed.generator/turkiye-siyaset")
    print(f"  FEED_URI_SCIENCE=at://{client.me.did}/app.bsky.feed.generator/turkiye-bilim")
    print("\nIMPORTANT: Your feed server must be publicly accessible at:")
    print(f"  https://{FEED_DOMAIN}")
    print("  (Use ngrok or deploy to a server for production)")


if __name__ == "__main__":
    main()
