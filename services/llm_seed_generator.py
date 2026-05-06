"""
LLM-based seed data generator for new feed topics.

Given a topic string and language, generates:
  - 100+ keyword phrases  (used for fast firehose pre-filtering)
  - 50+  seed sentences   (social-media style, used to build centroid)

Providers:
  - anthropic (default) — claude-sonnet-4-6
  - openai              — gpt-4o
"""
import json
import re
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    LLM_PROVIDER,
)

# Minimum counts we require from the LLM
MIN_KEYWORDS  = 50
MIN_SENTENCES = 30


def _build_prompt(topic: str, language: str) -> str:
    lang_map = {"tr": "Türkçe", "en": "English", "multi": "multilingual (mix of Turkish and English)"}
    lang_label = lang_map.get(language, language)

    if language == "tr":
        instructions = f"""Sen bir sosyal medya içerik uzmanısın.
"{topic}" konusu için Bluesky feed'i oluşturuyoruz.

Lütfen aşağıdaki JSON formatında veri üret:
{{
  "keywords": ["kelime1", "kelime2", ...],
  "seed_sentences": ["cümle1", "cümle2", ...]
}}

Gereksinimler:
- "keywords": En az 100 adet anahtar kelime / kısa ifade ({lang_label}, küçük harf).
  Bunlar firehose ön filtrelemesi için kullanılır. Konuyla yakından ilgili,
  özgün ve çeşitli olmalı (isimler, takım adları, teknik terimler, hashtag'ler vb.).
- "seed_sentences": En az 50 adet gerçek Bluesky/Twitter paylaşımı tarzında cümle ({lang_label}).
  Kısa, doğal, argo/slang içerebilir, hashtag ve mention kullanabilir.
  Bu cümleler centroid vektörü oluşturmak için kullanılır, bu yüzden
  gerçek sosyal medya postlarına çok benzemelidir.

SADECE geçerli JSON döndür, başka açıklama ekleme."""
    else:
        instructions = f"""You are a social media content expert.
We are building a Bluesky feed for the topic: "{topic}".

Return ONLY valid JSON in the following format:
{{
  "keywords": ["word1", "word2", ...],
  "seed_sentences": ["sentence1", "sentence2", ...]
}}

Requirements:
- "keywords": At least 100 keyword phrases ({lang_label}, lowercase).
  Used for fast firehose pre-filtering. Include player names, team names,
  hashtags, slang, technical terms related to the topic.
- "seed_sentences": At least 50 sentences in real Bluesky/Twitter post style ({lang_label}).
  Short, natural, can include hashtags and mentions.
  These build the centroid vector — they MUST resemble real social posts.

Return ONLY valid JSON, no explanation."""

    return instructions


def _parse_response(text: str) -> dict:
    """Extract JSON from LLM response (handles markdown code fences)."""
    # Strip markdown code fence if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # Find first { ... } block
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    data = json.loads(match.group(0))
    if "keywords" not in data or "seed_sentences" not in data:
        raise ValueError("Response missing 'keywords' or 'seed_sentences' keys")
    return data


def generate_with_anthropic(topic: str, language: str) -> dict:
    import anthropic

    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = _build_prompt(topic, language)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_response(message.content[0].text)


def generate_with_openai(topic: str, language: str) -> dict:
    from openai import OpenAI

    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = _build_prompt(topic, language)

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return _parse_response(response.choices[0].message.content)


def generate_seeds(topic: str, language: str = "tr", provider: str | None = None) -> dict:
    """
    Generate keywords and seed sentences for a topic.

    Args:
        topic:    Human-readable topic description (e.g. "FIFA World Cup 2026")
        language: 'tr' | 'en' | 'multi'
        provider: 'anthropic' | 'openai' (defaults to LLM_PROVIDER setting)

    Returns:
        {
          "keywords":       list[str],   # 100+
          "seed_sentences": list[str],   # 50+
        }
    """
    provider = provider or LLM_PROVIDER or "anthropic"

    if provider == "anthropic":
        data = generate_with_anthropic(topic, language)
    elif provider == "openai":
        data = generate_with_openai(topic, language)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'anthropic' or 'openai'.")

    # Deduplicate and clean
    keywords = list(dict.fromkeys(
        kw.strip().lower() for kw in data.get("keywords", []) if kw.strip()
    ))
    sentences = list(dict.fromkeys(
        s.strip() for s in data.get("seed_sentences", []) if len(s.strip()) > 5
    ))

    if len(keywords) < MIN_KEYWORDS:
        print(f"Warning: only {len(keywords)} keywords generated (expected {MIN_KEYWORDS}+)")
    if len(sentences) < MIN_SENTENCES:
        print(f"Warning: only {len(sentences)} seed sentences generated (expected {MIN_SENTENCES}+)")

    return {"keywords": keywords, "seed_sentences": sentences}
