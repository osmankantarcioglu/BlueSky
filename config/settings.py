import os
from dotenv import load_dotenv
load_dotenv()

# Bluesky hesap bilgileri
BSKY_HANDLE = os.getenv("BSKY_HANDLE")
BSKY_APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")

# Feed domain
FEED_DOMAIN = os.getenv("FEED_DOMAIN", "yourdomain.com")

# Feed URI'leri (publish sonrası gerçek değerlerle güncelle)
FEED_URI_POLITICS = f"at://did:web:{FEED_DOMAIN}/app.bsky.feed.generator/turkiye-siyaset"
FEED_URI_SCIENCE  = f"at://did:web:{FEED_DOMAIN}/app.bsky.feed.generator/turkiye-bilim"

# Model
BERTURK_MODEL = "dbmdz/bert-base-turkish-cased"
EMBEDDING_MODEL = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
EMBEDDING_DIM = 768

# Database
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/feeds.db")

# NLP eşik değerleri
DOMAIN_SIMILARITY_THRESHOLD = 0.35
STANCE_CONFIDENCE_THRESHOLD = 0.60

# Feed ayarları
MAX_FEED_POSTS = 200
FEED_CACHE_TTL = 300
MIN_TURKISH_PROB = 0.80

# Anahtar kelime listeleri
POLITICS_KEYWORDS = [
    "meclis", "tbmm", "milletvekili", "cumhurbaşkanı", "seçim",
    "parti", "hükümet", "muhalefet", "chp", "akp", "mhp", "hdp",
    "dem parti", "iyiparti", "erdoğan", "kılıçdaroğlu", "özel",
    "bütçe", "kanun", "yasa", "cumhur ittifakı", "millet ittifakı",
    "anayasa", "bakanlar kurulu", "siyaset", "politika"
]

SCIENCE_KEYWORDS = [
    "araştırma", "makale", "yayın", "bilim", "akademi", "üniversite",
    "arxiv", "doi", "nature", "science", "dergi", "preprint",
    "deney", "hipotez", "veri analizi", "istatistik", "metodoloji",
    "çalışma", "bulgu", "sonuç", "tez", "doktora", "lisans üstü"
]

ALLIANCE_KEYWORDS = [
    "akp", "mhp", "cumhur ittifakı", "erdoğan", "devlet bahçeli",
    "iktidar", "hükümet başarısı", "güçlü türkiye", "milli irade"
]

OPPOSITION_KEYWORDS = [
    "chp", "iyiparti", "dem", "hdp", "millet ittifakı", "muhalefet",
    "kılıçdaroğlu", "ekrem imamoğlu", "mansur yavas", "özgürlük"
]
