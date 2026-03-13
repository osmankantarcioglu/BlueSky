"""
One-time script to build and save domain centroid vectors.

Centroids are computed from keyword-based seed sentences using BERTurk.
Run this once before starting the firehose listener, then periodically
re-run after collecting real posts to improve accuracy.

Usage (from project root):
    python scripts/build_domain_centroids.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nlp.embedder import TurkishEmbedder
from nlp.domain_classifier import DomainClassifier

CENTROIDS_PATH = "data/centroids.npy"


def main():
    os.makedirs("data", exist_ok=True)

    print("Loading BERTurk embedding model...")
    embedder = TurkishEmbedder()

    print("Building centroids from keyword seed sentences...")
    classifier = DomainClassifier(embedder)
    classifier.build_centroids_from_keywords()

    classifier.save_centroids(CENTROIDS_PATH)
    print(f"\nCentroids saved to: {CENTROIDS_PATH}")

    # Quick sanity check — acceptable labels listed per test case
    print("\nSanity check:")
    tests = [
        ("TBMM'de yeni anayasa tartismalari basliyor.", {"politics", "both"}),
        ("Nature dergisinde makalemiz yayinlandi.", {"science", "both"}),
        ("Bugun hava cok guzel.", {"other"}),
    ]
    for text, acceptable in tests:
        label, score = classifier.classify(text)
        status = "OK" if label in acceptable else f"FAIL (expected one of {acceptable})"
        print(f"  [{status}] '{text[:50]}' -> {label} ({score:.3f})")


if __name__ == "__main__":
    main()
