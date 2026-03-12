# Bluesky Türkiye Siyaset & Bilim Custom Feed — Teknik Geliştirme Planı

> Bu döküman Claude Code'a verilmek üzere hazırlanmıştır. Her adım, tam olarak hangi kütüphane, model, dosya ve komutun kullanılacağını belirtir. Adımları sırasıyla takip edin.

---

## 0. Genel Mimari Özeti

```
┌─────────────────────────────────────────────────────────────────┐
│                        BLUESKY CLIENT                           │
│  (user pins feed → Bluesky calls /xrpc/app.bsky.feed.getFeedSkeleton) │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP GET (user DID + cursor + limit)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               FEED GENERATOR SERVER  (Flask/Waitress)           │
│   - /xrpc/app.bsky.feed.getFeedSkeleton                        │
│   - Reads from SQLite/PostgreSQL (pre-scored posts)            │
│   - Returns ordered post URI list in <50ms                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ reads
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATABASE (SQLite → PostgreSQL)            │
│   posts table: uri, did, text, created_at, domain_label,       │
│                stance_label, embedding_vector, score            │
│   users table: did, handle, party, domain                      │
└──────────┬──────────────────────────────────────────────────────┘
           │ writes                          │ reads
           ▼                                ▼
┌──────────────────────┐        ┌───────────────────────────────┐
│  FIREHOSE LISTENER   │        │    NLP PIPELINE (offline)     │
│  (atproto SDK)       │──────► │    BERTurk Embeddings         │
│  Filters by:         │        │    Cosine Similarity          │
│  - Seed user DIDs    │        │    Stance Detection           │
│  - Turkish keywords  │        └───────────────────────────────┘
└──────────────────────┘
```

**Temel bileşenler:**
1. **Firehose Listener** — AT Protocol firehose'undan gerçek zamanlı post akışı
2. **NLP Pipeline** — BERTurk ile embedding, domain (Siyaset/Bilim) ve stance (İttifak/Muhalefet) sınıflandırması
3. **Feed Generator Server** — Bluesky'ın çağırdığı HTTP endpoint
4. **Database** — İşlenmiş postları tutan SQLite (geliştirme) / PostgreSQL (production)

---

## 1. Kurulum ve Ortam Hazırlığı

### 1.1 Proje Dizin Yapısı

```
bluesky-turkish-feed/
├── config/
│   ├── settings.py          # Tüm konfigürasyon sabitleri
│   └── seed_users.csv       # 273 milletvekili/siyasetçi listesi
├── data_collection/
│   ├── firehose_listener.py # AT Protocol firehose subscriber
│   ├── seed_discovery.py    # Bluesky API ile kullanıcı keşfi
│   └── starter_pack_fetcher.py  # Starter pack'lerden kullanıcı toplama
├── nlp/
│   ├── embedder.py          # BERTurk sentence embedding
│   ├── domain_classifier.py # Siyaset vs Bilim sınıflandırması
│   ├── stance_detector.py   # İttifak vs Muhalefet sınıflandırması
│   └── pipeline.py          # End-to-end NLP pipeline orchestrator
├── database/
│   ├── models.py            # SQLite/PostgreSQL ORM modelleri (Peewee)
│   └── migrations.py        # Tablo oluşturma scriptleri
├── feed_generator/
│   ├── server.py            # Flask feed generator server
│   ├── feed_logic.py        # Feed sıralama ve filtreleme mantığı
│   └── did_resolver.py      # did:web identity yayınlama
├── scripts/
│   ├── build_domain_centroids.py  # Domain merkez vektörlerini oluştur
│   ├── fine_tune_stance.py        # Stance detection fine-tuning
│   └── discover_turkish_users.py  # Yeni Türk kullanıcı keşfi
├── requirements.txt
└── README.md
```

### 1.2 Python Ortamı

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install atproto==0.0.55          # AT Protocol SDK
pip install flask waitress           # Feed generator web server
pip install peewee                   # ORM (SQLite & PostgreSQL)
pip install transformers==4.40.0     # HuggingFace (BERTurk)
pip install sentence-transformers    # Sentence embedding wrapper
pip install torch                    # PyTorch (CPU veya CUDA)
pip install scikit-learn             # Cosine similarity, clustering
pip install numpy pandas             # Veri işleme
pip install requests httpx           # HTTP client
pip install python-dotenv            # .env yönetimi
pip install langdetect               # Dil tespiti (Türkçe filtresi)
pip install psycopg2-binary          # PostgreSQL bağlantısı (production)
pip install apscheduler              # Arka plan zamanlanmış görevler
```

### 1.3 Konfigürasyon Dosyası

**`config/settings.py`:**

```python
import os
from dotenv import load_dotenv
load_dotenv()

# Bluesky hesap bilgileri
BSKY_HANDLE = os.getenv("BSKY_HANDLE")       # örn: yourhandle.bsky.social
BSKY_APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")

# Feed kimliği
FEED_URI_POLITICS = "at://did:web:yourdomain.com/app.bsky.feed.generator/turkiye-siyaset"
FEED_URI_SCIENCE  = "at://did:web:yourdomain.com/app.bsky.feed.generator/turkiye-bilim"

# Model
BERTURK_MODEL = "dbmdz/bert-base-turkish-cased"  # HuggingFace model ID
EMBEDDING_DIM  = 768

# Database
DATABASE_PATH = "data/feeds.db"  # SQLite (dev)
# DATABASE_URL = os.getenv("DATABASE_URL")  # PostgreSQL (prod)

# NLP eşik değerleri
DOMAIN_SIMILARITY_THRESHOLD = 0.35   # Bu değerin üstündeki postlar domain'e eklenir
STANCE_CONFIDENCE_THRESHOLD = 0.60   # Stance sınıflandırması için minimum güven

# Feed ayarları
MAX_FEED_POSTS = 200       # Feed'de tutulacak maksimum post sayısı
FEED_CACHE_TTL = 300       # Saniye cinsinden cache süresi
MIN_TURKISH_PROB = 0.80    # langdetect Türkçe olasılığı eşiği

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
```

**.env dosyası (asla git'e commit etme):**
```
BSKY_HANDLE=your_handle.bsky.social
BSKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
DATABASE_URL=postgresql://user:pass@localhost/feeddb
```

---

## 2. Adım 1: Türk Kullanıcı Keşfi (Seed List + API Discovery)

Bu adımın amacı: Bluesky'da Türk siyaset ve bilim alanında aktif kullanıcıların DID ve handle listesini oluşturmak.

### 2.1 Starter Pack'lerden Kullanıcı Toplama

**`data_collection/starter_pack_fetcher.py`:**

```python
"""
Bluesky Starter Pack API'sini kullanarak Türk kullanıcı starter pack'lerini tara.
Endpoint: app.bsky.graph.getStarterPack ve app.bsky.graph.searchStarterPacks
"""
from atproto import Client
import pandas as pd
import time

def fetch_starter_pack_users(client: Client, search_terms: list[str]) -> pd.DataFrame:
    """
    Arama terimlerine göre starter pack'leri bul ve içindeki kullanıcıları topla.
    
    KULLANIM:
        client = Client()
        client.login(BSKY_HANDLE, BSKY_APP_PASSWORD)
        df = fetch_starter_pack_users(client, ["türkiye siyaset", "turkey science"])
    """
    users = []
    
    for term in search_terms:
        try:
            # Starter pack arama - atproto SDK metodu
            response = client.app.bsky.graph.search_starter_packs({"q": term, "limit": 25})
            
            for pack in response.starter_packs:
                print(f"Pack bulundu: {pack.record.name}")
                
                # Pack detaylarını çek
                pack_detail = client.app.bsky.graph.get_starter_pack(
                    {"starterPack": pack.uri}
                )
                
                # Pack içindeki kullanıcıları al
                if pack_detail.starter_pack.list:
                    list_uri = pack_detail.starter_pack.list.uri
                    members = client.app.bsky.graph.get_list(
                        {"list": list_uri, "limit": 100}
                    )
                    
                    for item in members.items:
                        users.append({
                            "did": item.subject.did,
                            "handle": item.subject.handle,
                            "display_name": item.subject.display_name,
                            "source": f"starter_pack:{pack.record.name}",
                            "search_term": term
                        })
                
                time.sleep(0.5)  # Rate limit koruması
                
        except Exception as e:
            print(f"Hata - {term}: {e}")
            continue
    
    df = pd.DataFrame(users).drop_duplicates(subset="did")
    return df


def search_feeds_by_keyword(client: Client, keywords: list[str]) -> list[dict]:
    """
    Türk siyaset/bilim feed'lerini bul.
    Endpoint: app.bsky.feed.searchFeeds (veya app.bsky.unspecced.getSuggestions)
    """
    feeds = []
    for kw in keywords:
        try:
            resp = client.app.bsky.feed.search_feeds({"q": kw, "limit": 20})
            for feed in resp.feeds:
                feeds.append({
                    "uri": feed.uri,
                    "name": feed.display_name,
                    "description": feed.description,
                    "creator_did": feed.creator.did,
                    "like_count": feed.like_count
                })
        except Exception as e:
            print(f"Feed arama hatası: {e}")
    return feeds
```

### 2.2 API Search ile Türk Kullanıcı Keşfi

**`data_collection/seed_discovery.py`:**

```python
"""
Bluesky'ın search API'sini kullanarak Türk kullanıcıları keşfet.
Bluesky'da Türk siyasetçiler için tipik handle formatları:
  - ad-soyad.bsky.social
  - @tbmm üyeleri profil bio'larında genellikle parti adını belirtir
"""
from atproto import Client
import pandas as pd
from langdetect import detect
import time

def search_turkish_users(client: Client, query: str, limit: int = 100) -> list[dict]:
    """
    app.bsky.actor.searchActors endpoint'i ile kullanıcı ara.
    Bio veya display name'de Türkçe içerik araması yapar.
    """
    users = []
    cursor = None
    
    while len(users) < limit:
        try:
            resp = client.app.bsky.actor.search_actors({
                "q": query,
                "limit": 25,
                "cursor": cursor
            })
            
            for actor in resp.actors:
                # Bio'da Türkçe içerik kontrolü
                bio = actor.description or ""
                if bio and len(bio) > 10:
                    try:
                        lang = detect(bio)
                        if lang == "tr":
                            users.append({
                                "did": actor.did,
                                "handle": actor.handle,
                                "display_name": actor.display_name,
                                "bio": bio,
                                "followers": actor.followers_count,
                                "source": f"search:{query}"
                            })
                    except:
                        pass
            
            cursor = resp.cursor
            if not cursor:
                break
            time.sleep(0.3)
            
        except Exception as e:
            print(f"Arama hatası: {e}")
            break
    
    return users


def get_followers_of_seed(client: Client, seed_did: str, max_depth: int = 1) -> list[dict]:
    """
    Seed kullanıcının takipçilerini al.
    max_depth=1: sadece seed'in takipçileri
    max_depth=2: takipçilerin takipçileri (dikkatli kullan, büyür)
    
    NOT: Bu yöntem "network expansion" için kullanılır.
    Milletvekillerini takip eden kişiler genellikle siyasi ilgi gösterir.
    """
    users = []
    cursor = None
    
    for _ in range(10):  # Max 10 sayfa (250 kullanıcı)
        try:
            resp = client.app.bsky.graph.get_followers({
                "actor": seed_did,
                "limit": 25,
                "cursor": cursor
            })
            
            for follower in resp.followers:
                users.append({
                    "did": follower.did,
                    "handle": follower.handle,
                    "display_name": follower.display_name,
                    "source": f"follower_of:{seed_did}"
                })
            
            cursor = resp.cursor
            if not cursor:
                break
            time.sleep(0.2)
        except Exception as e:
            break
    
    return users


def load_csv_seeds(csv_path: str) -> pd.DataFrame:
    """
    CSV'den milletvekili/siyasetçi listesini yükle.
    
    Beklenen CSV sütunları:
    - handle: Bluesky handle (örn: ahmetarslan.bsky.social)
    - name: Tam ad
    - party: Parti adı (AKP, CHP, MHP, HDP, IYI, vb.)
    - domain: politics veya science
    
    Eğer handle yoksa, isim bazlı search_actors ile bul.
    """
    df = pd.read_csv(csv_path)
    
    # Parti etiketleri → İttifak/Muhalefet
    alliance_parties = {"AKP", "MHP", "BBP", "YRP"}
    opposition_parties = {"CHP", "IYI", "DEM", "HDP", "DEVA", "GP", "SP"}
    
    df["stance"] = df["party"].apply(
        lambda p: "alliance" if str(p).upper() in alliance_parties
                  else "opposition" if str(p).upper() in opposition_parties
                  else "unknown"
    )
    
    return df
```

**Çalıştırma scripti `scripts/discover_turkish_users.py`:**

```python
from atproto import Client
from data_collection.starter_pack_fetcher import fetch_starter_pack_users, search_feeds_by_keyword
from data_collection.seed_discovery import search_turkish_users, load_csv_seeds
from config.settings import BSKY_HANDLE, BSKY_APP_PASSWORD
import pandas as pd

client = Client()
client.login(BSKY_HANDLE, BSKY_APP_PASSWORD)

# 1. CSV seed listesini yükle
seeds = load_csv_seeds("config/seed_users.csv")
print(f"CSV'den {len(seeds)} seed yüklendi")

# 2. Starter pack'lerden kullanıcı topla
pack_users = fetch_starter_pack_users(client, [
    "türkiye siyaset", "turkish politics", "tbmm",
    "turkish science", "türk bilim", "akademi türkiye"
])

# 3. API search ile ek kullanıcılar
search_queries = ["milletvekili", "tbmm", "siyasetçi", "türk akademisyen", "türkiye bilim"]
searched_users = []
for q in search_queries:
    searched_users.extend(search_turkish_users(client, q, limit=200))

# 4. Feed creator'ları bul
feeds = search_feeds_by_keyword(client, ["türkiye", "turkish", "siyaset", "bilim"])
print(f"Bulunan feedler: {len(feeds)}")

# 5. Tümünü birleştir ve kaydet
all_users = pd.concat([
    seeds[["did", "handle", "party", "stance", "domain"]],
    pd.DataFrame(pack_users),
    pd.DataFrame(searched_users)
]).drop_duplicates(subset="did")

all_users.to_csv("data/discovered_users.csv", index=False)
print(f"Toplam {len(all_users)} kullanıcı kaydedildi")
```

---

## 3. Adım 2: Veritabanı Modelleri

**`database/models.py`:**

```python
"""
Peewee ORM ile SQLite (geliştirme) / PostgreSQL (production) modelleri.
"""
from peewee import *
from config.settings import DATABASE_PATH
import datetime

# SQLite için:
db = SqliteDatabase(DATABASE_PATH)

# PostgreSQL için (production'da bu satırı kullan):
# from playhouse.pool import PooledPostgresqlExtDatabase
# db = PooledPostgresqlExtDatabase(DATABASE_URL, max_connections=32, stale_timeout=300)


class BaseModel(Model):
    class Meta:
        database = db


class TrackedUser(BaseModel):
    """Takip edilen Türk kullanıcılar (milletvekilleri + keşfedilenler)."""
    did = CharField(primary_key=True)
    handle = CharField(index=True)
    display_name = CharField(null=True)
    party = CharField(null=True)           # AKP, CHP, MHP, vb.
    stance = CharField(null=True)          # 'alliance', 'opposition', 'unknown'
    domain = CharField(null=True)          # 'politics', 'science', 'both'
    source = CharField(null=True)          # CSV, starter_pack, search
    created_at = DateTimeField(default=datetime.datetime.utcnow)
    is_active = BooleanField(default=True)

    class Meta:
        table_name = "tracked_users"


class Post(BaseModel):
    """
    Firehose'dan toplanan ve işlenmiş postlar.
    embedding sütunu: BERTurk vektörü (768 boyutlu, JSON string olarak saklanır)
    """
    uri = CharField(primary_key=True)      # at://did:xxx/app.bsky.feed.post/xxx
    cid = CharField()
    author_did = CharField(index=True)
    author_handle = CharField(null=True)
    text = TextField()
    
    # NLP sonuçları
    domain_label = CharField(null=True)    # 'politics', 'science', 'both', 'other'
    stance_label = CharField(null=True)    # 'alliance', 'opposition', 'neutral'
    domain_score = FloatField(null=True)   # Cosine similarity skoru
    stance_score = FloatField(null=True)   # Stance model confidence
    embedding = TextField(null=True)       # JSON string: [0.12, -0.34, ...]
    
    # Metadata
    created_at = DateTimeField(index=True)
    indexed_at = DateTimeField(default=datetime.datetime.utcnow)
    language = CharField(null=True)        # 'tr', 'en', vb.
    
    # Engagement (firehose'dan gelir)
    like_count = IntegerField(default=0)
    repost_count = IntegerField(default=0)
    reply_count = IntegerField(default=0)
    
    # Feed sıralama skoru (combined score)
    feed_score = FloatField(default=0.0, index=True)
    
    class Meta:
        table_name = "posts"
        indexes = (
            (("domain_label", "created_at"), False),
            (("stance_label", "domain_label", "created_at"), False),
        )


class LikeEvent(BaseModel):
    """Firehose'dan gelen like olayları."""
    uri = CharField()              # Liked post URI
    liker_did = CharField()
    created_at = DateTimeField(default=datetime.datetime.utcnow)

    class Meta:
        table_name = "like_events"


def create_tables():
    with db:
        db.create_tables([TrackedUser, Post, LikeEvent], safe=True)


if __name__ == "__main__":
    create_tables()
    print("Tablolar oluşturuldu.")
```

---

## 4. Adım 3: NLP Pipeline — BERTurk Embedding ve Sınıflandırma

Bu, projenin en kritik bileşenidir. İki ana görev:
1. **Domain Classification**: Post, Siyaset mi yoksa Bilim mi?
2. **Stance Detection**: Siyaset postları, İttifak mı Muhalefet mi?

### 4.1 BERTurk Embedder

**`nlp/embedder.py`:**

```python
"""
BERTurk tabanlı sentence embedding modülü.
Model: dbmdz/bert-base-turkish-cased (Türkçe için fine-tune edilmiş BERT)

ALTERNATİF MODELLER (daha iyi Türkçe performansı için test et):
- "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  (çok dilli, hızlı)
- "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"  (Türkçe STS, önerilen)
- "dbmdz/bert-base-turkish-128k-cased"  (daha büyük vocabulary)
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from typing import Union
import json

class TurkishEmbedder:
    """
    Sentence Transformer wrapper, Türkçe metinler için optimize edilmiş.
    
    KULLANIM:
        embedder = TurkishEmbedder()
        vector = embedder.embed("Meclis'te yeni kanun teklifi kabul edildi.")
        # vector.shape = (768,)
    """
    
    def __init__(self, model_name: str = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"):
        """
        Model yükleme.
        
        ÖNEMLİ: İlk çalıştırmada model ~400MB indirilir.
        Sonraki çalıştırmalarda cache'den yüklenir.
        
        GPU varsa otomatik kullanılır, yoksa CPU'da çalışır.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding modeli yükleniyor: {model_name} ({device})")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        print("Model yüklendi.")
    
    def embed(self, text: str) -> np.ndarray:
        """
        Tek bir metni vektöre dönüştür.
        Returns: numpy array, shape (768,), dtype float32
        """
        text = self._preprocess(text)
        vector = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vector
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Toplu vektörleştirme (firehose işleme için kullan).
        Returns: numpy array, shape (len(texts), 768)
        
        batch_size=32: GPU belleğine göre artırılabilir (64, 128)
        """
        texts = [self._preprocess(t) for t in texts]
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        return vectors
    
    def _preprocess(self, text: str) -> str:
        """
        Post metnini temizle.
        - URL'leri kaldır (anlam taşımaz, sadece gürültü)
        - Handle mention'ları kaldır (@kullanici)
        - Maksimum 512 token (BERT limiti) → yaklaşık 400 karakter
        """
        import re
        text = re.sub(r'http\S+', '', text)   # URL kaldır
        text = re.sub(r'@\w+', '', text)       # Mention kaldır
        text = re.sub(r'#(\w+)', r'\1', text)  # Hashtag sembolünü kaldır, kelimeyi bırak
        text = ' '.join(text.split())           # Fazla boşlukları temizle
        return text[:512]  # BERT token limiti için güvenli kesim
    
    def vector_to_json(self, vector: np.ndarray) -> str:
        """Veritabanında saklamak için JSON string'e dönüştür."""
        return json.dumps(vector.tolist())
    
    def json_to_vector(self, json_str: str) -> np.ndarray:
        """Veritabanından yüklenen JSON'u numpy array'e dönüştür."""
        return np.array(json.loads(json_str), dtype=np.float32)
```

### 4.2 Domain Sınıflandırıcı

**`nlp/domain_classifier.py`:**

```python
"""
Cosine Similarity tabanlı domain sınıflandırması.
Her post, "Siyaset" ve "Bilim" merkez vektörlerine (centroid) olan
uzaklığına göre sınıflandırılır.

CENTROID OLUŞTURMA:
1. Çok sayıda örnek siyaset ve bilim postu topla
2. Her grubun embedding ortalamasını al → centroid
3. Centroid'leri kaydet (numpy .npy dosyası)
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nlp.embedder import TurkishEmbedder
from config.settings import DOMAIN_SIMILARITY_THRESHOLD, POLITICS_KEYWORDS, SCIENCE_KEYWORDS


class DomainClassifier:
    """
    Sıfır-shot domain sınıflandırıcı.
    
    İKİ YÖNTEM desteklenir:
    1. Centroid-based: Seed metinlerden hesaplanan centroid vektörlerine cosine similarity
    2. Keyword-boosted: Kelime eşleşmesi ile sinyali güçlendir (düşük kaliteli postlar için fallback)
    
    KULLANIM:
        classifier = DomainClassifier(embedder)
        classifier.load_centroids("data/centroids.npy")
        label, score = classifier.classify("Erdoğan TBMM'de konuşma yaptı")
        # label = 'politics', score = 0.78
    """
    
    def __init__(self, embedder: TurkishEmbedder):
        self.embedder = embedder
        self.centroids = {}  # {'politics': np.array([...]), 'science': np.array([...])}
    
    def build_centroids_from_texts(self, texts_by_domain: dict[str, list[str]]) -> None:
        """
        Verilen metinlerden centroid hesapla ve kaydet.
        
        PARAMETRE:
            texts_by_domain = {
                'politics': ["Meclis oturumu...", "Parti toplantısı...", ...],
                'science': ["Yeni araştırma...", "Nature dergisi...", ...]
            }
        
        EN AZ 50 örnek per domain önerilir.
        """
        for domain, texts in texts_by_domain.items():
            print(f"{domain} için {len(texts)} metin embedding yapılıyor...")
            embeddings = self.embedder.embed_batch(texts)
            centroid = embeddings.mean(axis=0)
            centroid /= np.linalg.norm(centroid)  # Normalize
            self.centroids[domain] = centroid
    
    def build_centroids_from_keywords(self) -> None:
        """
        Keyword listelerini kullanarak HIZLI centroid oluşturma.
        
        Bu yöntem, başlangıçta yeterli örnek metin yoksa kullanılabilir.
        Ancak uzun vadede gerçek post örnekleri ile daha iyi sonuç alınır.
        """
        # Keyword'leri temsili cümlelere dönüştür
        politics_texts = [
            f"Bugün {kw} hakkında önemli gelişme oldu." for kw in POLITICS_KEYWORDS
        ] + [
            "Meclis'te yeni yasa teklifi kabul edildi.",
            "Cumhurbaşkanı açıklamasında muhalefeti eleştirdi.",
            "Seçim kampanyası sürecinde partiler tartışıyor.",
            "TBMM genel kurulunda bütçe görüşmeleri yapıldı."
        ]
        
        science_texts = [
            f"Bu {kw} konusundaki yeni çalışmam yayınlandı." for kw in SCIENCE_KEYWORDS
        ] + [
            "arXiv'de yeni makalemiz çıktı, bağlantı profilde.",
            "Nature dergisinde Türk araştırmacıların çalışması.",
            "Bu araştırmanın bulguları dikkat çekici sonuçlar gösteriyor.",
            "Doktora tezim kabul edildi, teşekkürler."
        ]
        
        self.build_centroids_from_texts({
            'politics': politics_texts,
            'science': science_texts
        })
    
    def save_centroids(self, path: str = "data/centroids.npy") -> None:
        np.save(path, self.centroids)
        print(f"Centroid'ler kaydedildi: {path}")
    
    def load_centroids(self, path: str = "data/centroids.npy") -> None:
        self.centroids = np.load(path, allow_pickle=True).item()
        print(f"Centroid'ler yüklendi: {list(self.centroids.keys())}")
    
    def classify(self, text: str, embedding: np.ndarray = None) -> tuple[str, float]:
        """
        Bir metni 'politics', 'science', 'both', veya 'other' olarak sınıflandır.
        
        PARAMETRE:
            text: Post metni
            embedding: Zaten hesaplanmış embedding (None ise hesaplanır)
        
        DÖNÜŞ:
            (label, score): label = domain adı, score = en yüksek cosine similarity
        
        MANTIK:
            1. Embedding vektörünü hesapla
            2. Her centroid'e cosine similarity hesapla
            3. Her iki eşiği de geçiyorsa 'both'
            4. Sadece biri geçiyorsa o domain
            5. Hiçbiri geçmiyorsa keyword fallback
            6. Keyword de yoksa 'other'
        """
        if embedding is None:
            embedding = self.embedder.embed(text)
        
        scores = {}
        for domain, centroid in self.centroids.items():
            sim = cosine_similarity(
                embedding.reshape(1, -1),
                centroid.reshape(1, -1)
            )[0][0]
            scores[domain] = float(sim)
        
        above_threshold = {
            d: s for d, s in scores.items()
            if s >= DOMAIN_SIMILARITY_THRESHOLD
        }
        
        if len(above_threshold) == 2:
            return 'both', max(scores.values())
        elif len(above_threshold) == 1:
            label = list(above_threshold.keys())[0]
            return label, above_threshold[label]
        else:
            # Fallback: keyword kontrolü
            text_lower = text.lower()
            has_politics = any(kw in text_lower for kw in POLITICS_KEYWORDS)
            has_science = any(kw in text_lower for kw in SCIENCE_KEYWORDS)
            
            if has_politics and has_science:
                return 'both', max(scores.values())
            elif has_politics:
                return 'politics', scores.get('politics', 0.0)
            elif has_science:
                return 'science', scores.get('science', 0.0)
            else:
                return 'other', 0.0
```

### 4.3 Stance Detector

**`nlp/stance_detector.py`:**

```python
"""
Siyasi tutum (stance) tespiti.
YÖNTEMLERİ:

YÖNTEM 1 (Önerilen - Başlangıç için): Keyword + Cosine Similarity
  - Hızlı, veri gerektirmez
  - Yeterli doğruluk sağlar

YÖNTEM 2 (Gelişmiş - Fine-tuning): 
  - CSV'deki parti etiketli kullanıcıların postlarıyla BERTurk fine-tune
  - Daha yüksek doğruluk ama veri toplama gerektirir

Bu dosyada her iki yöntem de implemente edilmiştir.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nlp.embedder import TurkishEmbedder
from config.settings import (
    STANCE_CONFIDENCE_THRESHOLD,
    ALLIANCE_KEYWORDS,
    OPPOSITION_KEYWORDS
)


class StanceDetectorV1:
    """
    Yöntem 1: Keyword + Centroid tabanlı stance detection.
    
    Fine-tuning verisine ihtiyaç duymaz.
    CSV'deki 273 milletvekilinin partilerine göre hazırlanan
    örnek metinlerden centroid hesaplar.
    """
    
    def __init__(self, embedder: TurkishEmbedder):
        self.embedder = embedder
        self.stance_centroids = {}
    
    def build_stance_centroids_from_users(
        self,
        client,  # atproto Client
        stance_users: dict[str, list[str]]  # {'alliance': [did1, did2...], 'opposition': [...]}
    ) -> None:
        """
        Her stance grubunun kullanıcılarının son postlarını çekip centroid hesapla.
        
        Bu fonksiyon bir kez çalıştırılır ve centroid kayıt edilir.
        Yaklaşık 1-2 saat sürebilir (rate limit nedeniyle).
        
        PARAMETRE:
            stance_users: Her gruptan en az 20 kullanıcı önerilir
        """
        import time
        
        for stance, dids in stance_users.items():
            all_texts = []
            
            for did in dids[:50]:  # Her gruptan max 50 kullanıcı
                try:
                    feed = client.app.bsky.feed.get_author_feed({
                        "actor": did,
                        "limit": 20,
                        "filter": "posts_no_replies"
                    })
                    
                    for item in feed.feed:
                        text = item.post.record.text
                        if len(text) > 20:  # Çok kısa postları atla
                            all_texts.append(text)
                    
                    time.sleep(0.3)
                except Exception as e:
                    continue
            
            if all_texts:
                print(f"{stance}: {len(all_texts)} post embedding yapılıyor...")
                embeddings = self.embedder.embed_batch(all_texts)
                centroid = embeddings.mean(axis=0)
                centroid /= np.linalg.norm(centroid)
                self.stance_centroids[stance] = centroid
                print(f"{stance} centroid hazır.")
    
    def detect_stance(self, text: str, embedding: np.ndarray = None) -> tuple[str, float]:
        """
        Post metninin siyasi tutumunu belirle.
        
        DÖNÜŞ:
            ('alliance' | 'opposition' | 'neutral', confidence_score)
        
        NOT: Bu fonksiyon SADECE domain_label='politics' olan postlar için çağrılmalıdır.
        """
        if embedding is None:
            embedding = self.embedder.embed(text)
        
        # Centroid yoksa keyword fallback
        if not self.stance_centroids:
            return self._keyword_fallback(text)
        
        scores = {}
        for stance, centroid in self.stance_centroids.items():
            sim = cosine_similarity(
                embedding.reshape(1, -1),
                centroid.reshape(1, -1)
            )[0][0]
            scores[stance] = float(sim)
        
        best_stance = max(scores, key=scores.get)
        best_score = scores[best_stance]
        
        if best_score < STANCE_CONFIDENCE_THRESHOLD:
            # Düşük güven → keyword fallback
            kw_stance, kw_score = self._keyword_fallback(text)
            if kw_stance != 'neutral':
                return kw_stance, kw_score
            return 'neutral', best_score
        
        return best_stance, best_score
    
    def _keyword_fallback(self, text: str) -> tuple[str, float]:
        """Keyword sayım bazlı basit stance tespiti."""
        text_lower = text.lower()
        alliance_count = sum(1 for kw in ALLIANCE_KEYWORDS if kw in text_lower)
        opposition_count = sum(1 for kw in OPPOSITION_KEYWORDS if kw in text_lower)
        
        if alliance_count > opposition_count:
            score = min(0.5 + alliance_count * 0.1, 0.9)
            return 'alliance', score
        elif opposition_count > alliance_count:
            score = min(0.5 + opposition_count * 0.1, 0.9)
            return 'opposition', score
        elif alliance_count == opposition_count > 0:
            return 'neutral', 0.5
        else:
            return 'neutral', 0.3
    
    def save(self, path: str = "data/stance_centroids.npy") -> None:
        np.save(path, self.stance_centroids)
    
    def load(self, path: str = "data/stance_centroids.npy") -> None:
        self.stance_centroids = np.load(path, allow_pickle=True).item()


class StanceDetectorV2:
    """
    Yöntem 2: Fine-tuned BERTurk ile stance detection.
    
    REQUIREMENTS:
    - En az 500 etiketli post per sınıf (alliance/opposition/neutral)
    - Fine-tuning: ~2-4 saat (CPU) veya ~30 dakika (GPU)
    
    FINE-TUNING ADIMLARI:
    1. CSV'deki milletvekillerinin postlarını çek (Adım 2'deki seed users)
    2. Her postu kullanıcının partisine göre etiketle (AKP→alliance, CHP→opposition)
    3. BERTurk'ü sequence classification görevi için fine-tune et
    4. Model kaydet ve bu sınıfta yükle
    
    Bu, scripts/fine_tune_stance.py ile yapılır (aşağıda detaylandırılmıştır).
    """
    
    def __init__(self, model_path: str = "models/stance_berturk"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.label_map = {0: 'alliance', 1: 'opposition', 2: 'neutral'}
    
    def detect_stance(self, text: str) -> tuple[str, float]:
        import torch
        
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=128, padding=True
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        label_idx = probs.argmax()
        return self.label_map[label_idx], float(probs[label_idx])
```

### 4.4 Fine-tuning Scripti

**`scripts/fine_tune_stance.py`:**

```python
"""
BERTurk modelini stance detection için fine-tune eder.
CSV'deki milletvekillerinin postlarını kullanarak 3-sınıflı
(alliance / opposition / neutral) sınıflandırma modeli eğitir.

ÇALIŞTIRMA:
    python scripts/fine_tune_stance.py

GEREKSİNİMLER:
    - En az 300 etiketli post (150 alliance + 150 opposition)
    - GPU önerilir (~30 dakika) ama CPU'da da çalışır (~4 saat)
"""
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset
import pandas as pd
import numpy as np
import torch

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
OUTPUT_DIR = "models/stance_berturk"
LABEL2ID = {'alliance': 0, 'opposition': 1, 'neutral': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_training_data(csv_path: str = "data/stance_training_data.csv") -> pd.DataFrame:
    """
    Beklenen CSV formatı:
    - text: Post metni
    - stance: 'alliance', 'opposition', veya 'neutral'
    
    Bu CSV şu şekilde oluşturulur:
    1. Seed kullanıcıların postlarını çek
    2. Kullanıcının parti etiketini postuna ata
    3. Tarafsız postlar için manuel etiketleme veya keyword heuristic
    """
    df = pd.read_csv(csv_path)
    df['label'] = df['stance'].map(LABEL2ID)
    return df.dropna(subset=['text', 'label'])


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=128,
        padding='max_length'
    )


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro')
    }


def train():
    print("Eğitim verisi yükleniyor...")
    df = load_training_data()
    print(f"Toplam örnek: {len(df)}")
    print(df['stance'].value_counts())
    
    # Train/val split
    train_df = df.sample(frac=0.85, random_state=42)
    val_df = df.drop(train_df.index)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,            # 5 epoch genellikle yeterli
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,            # BERT fine-tuning için standart LR
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="best",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        fp16=torch.cuda.is_available(),  # GPU varsa mixed precision
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    print("Fine-tuning başlıyor...")
    trainer.train()
    
    # Modeli kaydet
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
```

### 4.5 Birleşik NLP Pipeline

**`nlp/pipeline.py`:**

```python
"""
End-to-end NLP pipeline.
Firehose'dan gelen ham post → domain_label + stance_label + embedding
"""
from nlp.embedder import TurkishEmbedder
from nlp.domain_classifier import DomainClassifier
from nlp.stance_detector import StanceDetectorV1
from database.models import Post
from langdetect import detect, LangDetectException
from config.settings import MIN_TURKISH_PROB
import datetime


class NLPPipeline:
    """
    Tüm NLP bileşenlerini yöneten orchestrator.
    
    KULLANIM:
        pipeline = NLPPipeline()
        pipeline.load_models()
        
        result = pipeline.process_post(
            uri="at://did:xxx/app.bsky.feed.post/xxx",
            cid="bafyxxx",
            author_did="did:plc:xxx",
            text="Meclis'te önemli karar alındı",
            created_at=datetime.datetime.utcnow()
        )
        # result = Post nesnesi (veritabanına kaydedilmiş)
    """
    
    def __init__(self):
        self.embedder = None
        self.domain_classifier = None
        self.stance_detector = None
        self._loaded = False
    
    def load_models(
        self,
        centroid_path: str = "data/centroids.npy",
        stance_path: str = "data/stance_centroids.npy"
    ) -> None:
        """Tüm modelleri yükle. Uygulama başlangıcında bir kez çağrılmalı."""
        print("NLP modelleri yükleniyor...")
        
        self.embedder = TurkishEmbedder()
        
        self.domain_classifier = DomainClassifier(self.embedder)
        try:
            self.domain_classifier.load_centroids(centroid_path)
        except FileNotFoundError:
            print("Centroid dosyası bulunamadı, keyword-based oluşturuluyor...")
            self.domain_classifier.build_centroids_from_keywords()
            self.domain_classifier.save_centroids(centroid_path)
        
        self.stance_detector = StanceDetectorV1(self.embedder)
        try:
            self.stance_detector.load(stance_path)
        except FileNotFoundError:
            print("Stance centroid bulunamadı. Kullanıcı postları toplanana kadar keyword modu kullanılır.")
        
        self._loaded = True
        print("NLP pipeline hazır.")
    
    def is_turkish(self, text: str) -> bool:
        """Metnin Türkçe olup olmadığını kontrol et."""
        try:
            lang = detect(text)
            return lang == "tr"
        except LangDetectException:
            return False
    
    def process_post(
        self,
        uri: str,
        cid: str,
        author_did: str,
        author_handle: str,
        text: str,
        created_at: datetime.datetime
    ) -> Post | None:
        """
        Ham postu işle ve veritabanına kaydet.
        
        DÖNÜŞ: Post nesnesi veya None (Türkçe değilse veya domain 'other'ise)
        """
        assert self._loaded, "load_models() çağrılmamış!"
        
        # 1. Türkçe filtresi
        if not self.is_turkish(text):
            return None
        
        # 2. Embedding hesapla
        embedding = self.embedder.embed(text)
        embedding_json = self.embedder.vector_to_json(embedding)
        
        # 3. Domain sınıflandırması
        domain_label, domain_score = self.domain_classifier.classify(text, embedding)
        
        if domain_label == 'other':
            return None  # İlgisiz post
        
        # 4. Stance detection (sadece siyaset postları için)
        stance_label = 'neutral'
        stance_score = 0.0
        
        if domain_label in ('politics', 'both'):
            stance_label, stance_score = self.stance_detector.detect_stance(text, embedding)
        
        # 5. Feed skoru hesapla (basit formül, iyileştirilebilir)
        feed_score = domain_score
        
        # 6. Veritabanına kaydet
        post, created = Post.get_or_create(
            uri=uri,
            defaults={
                'cid': cid,
                'author_did': author_did,
                'author_handle': author_handle,
                'text': text,
                'domain_label': domain_label,
                'stance_label': stance_label,
                'domain_score': domain_score,
                'stance_score': stance_score,
                'embedding': embedding_json,
                'created_at': created_at,
                'language': 'tr',
                'feed_score': feed_score
            }
        )
        
        return post
```

---

## 5. Adım 4: Firehose Listener

**`data_collection/firehose_listener.py`:**

```python
"""
AT Protocol Firehose'una abone ol ve ilgili postları yakala.

Firehose, Bluesky'daki TÜM public eventleri gerçek zamanlı olarak yayınlar.
Saniyede ~200-500 event gelir; bunların büyük çoğunluğu ilgisiz olacak.

Filtreleme stratejisi (hızlı → yavaş sırasıyla):
1. Sadece 'app.bsky.feed.post' type event'leri al (post creation)
2. Author DID'i seed users setinde mi? → evet ise işle
3. Türkçe dil kontrolü (langdetect, hızlı)
4. Keyword kontrolü (regex, çok hızlı)
5. BERTurk NLP pipeline (yavaş, sadece gerekirse)
"""
from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message
from atproto.exceptions import FirehoseError
import json
import time
import threading
from collections import deque
from database.models import db, Post, TrackedUser
from nlp.pipeline import NLPPipeline
from config.settings import POLITICS_KEYWORDS, SCIENCE_KEYWORDS
import datetime
import re

# Türkçe anahtar kelime seti (hızlı kontrol için set olarak tut)
ALL_KEYWORDS = set(POLITICS_KEYWORDS + SCIENCE_KEYWORDS)


class FirehoseProcessor:
    """
    Firehose event'lerini işleyen ana sınıf.
    
    PERFORMANS NOTLARI:
    - NLP pipeline yüklenmesi 30-60 saniye sürer
    - Firehose ~300 event/sn işler
    - BERTurk embedding ~50ms/post (CPU), ~5ms/post (GPU)
    - Batch processing kullanılır (32 post biriktir, hepsini birlikte işle)
    """
    
    def __init__(self):
        self.nlp_pipeline = NLPPipeline()
        self.seed_dids: set[str] = set()
        self.post_queue = deque(maxlen=1000)  # İşlenecek post kuyruğu
        self.running = False
    
    def setup(self):
        """Başlangıç hazırlıkları."""
        db.connect(reuse_if_open=True)
        
        # Seed kullanıcıları belleğe yükle (hızlı lookup için set)
        self.seed_dids = set(
            u.did for u in TrackedUser.select(TrackedUser.did).where(TrackedUser.is_active == True)
        )
        print(f"Seed DID'ler yüklendi: {len(self.seed_dids)}")
        
        # NLP pipeline yükle
        self.nlp_pipeline.load_models()
    
    def _has_relevant_keyword(self, text: str) -> bool:
        """Hızlı keyword kontrolü."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in ALL_KEYWORDS)
    
    def _extract_post_data(self, commit, op):
        """
        Firehose commit'inden post verisini çıkar.
        
        DÖNÜŞ: (uri, cid, author_did, text, created_at) tuple veya None
        """
        try:
            record = op.get('record', {})
            if record.get('$type') != 'app.bsky.feed.post':
                return None
            
            text = record.get('text', '')
            if not text or len(text) < 10:
                return None
            
            author_did = commit.repo
            rkey = op.get('rkey', '')
            uri = f"at://{author_did}/app.bsky.feed.post/{rkey}"
            cid = op.get('cid', '')
            
            # created_at parse
            created_at_str = record.get('createdAt', '')
            try:
                created_at = datetime.datetime.fromisoformat(
                    created_at_str.replace('Z', '+00:00')
                )
            except:
                created_at = datetime.datetime.utcnow()
            
            return uri, cid, author_did, text, created_at
        except Exception:
            return None
    
    def on_message_handler(self, message) -> None:
        """
        Her firehose mesajı için çağrılır.
        Hızlı filtreleme yapıp kuyruğa ekler.
        """
        try:
            commit = parse_subscribe_repos_message(message)
            
            if not hasattr(commit, 'ops'):
                return
            
            for op in commit.ops:
                if op.get('action') != 'create':
                    continue
                
                post_data = self._extract_post_data(commit, op)
                if not post_data:
                    continue
                
                uri, cid, author_did, text, created_at = post_data
                
                # Hızlı filtreler
                is_seed_user = author_did in self.seed_dids
                has_keyword = self._has_relevant_keyword(text)
                
                if is_seed_user or has_keyword:
                    self.post_queue.append({
                        'uri': uri, 'cid': cid,
                        'author_did': author_did,
                        'author_handle': '',  # Handle firehose'da gelmez, API'den çekilebilir
                        'text': text, 'created_at': created_at
                    })
        except Exception as e:
            pass
    
    def _process_queue_worker(self):
        """
        Arka plan thread'i: kuyruktaki postları NLP pipeline ile işler.
        Her 5 saniyede bir çalışır veya kuyruk 32 post dolunca.
        """
        batch = []
        
        while self.running:
            while self.post_queue and len(batch) < 32:
                batch.append(self.post_queue.popleft())
            
            if batch:
                for post_data in batch:
                    try:
                        self.nlp_pipeline.process_post(**post_data)
                    except Exception as e:
                        print(f"Post işleme hatası: {e}")
                
                print(f"Batch işlendi: {len(batch)} post")
                batch = []
            else:
                time.sleep(5)
    
    def start(self):
        """Firehose dinlemeyi başlat."""
        self.running = True
        self.setup()
        
        # Queue worker thread'ini başlat
        worker = threading.Thread(target=self._process_queue_worker, daemon=True)
        worker.start()
        
        # Firehose client
        client = FirehoseSubscribeReposClient()
        
        print("Firehose bağlanıyor...")
        
        while self.running:
            try:
                client.start(self.on_message_handler)
            except FirehoseError as e:
                print(f"Firehose hatası, 10 saniye sonra yeniden bağlanılıyor: {e}")
                time.sleep(10)
            except Exception as e:
                print(f"Beklenmeyen hata: {e}")
                time.sleep(5)


if __name__ == "__main__":
    processor = FirehoseProcessor()
    processor.start()
```

---

## 6. Adım 5: Feed Generator Server

**`feed_generator/server.py`:**

```python
"""
Bluesky'ın çağırdığı HTTP feed generator server.
Flask + Waitress (production WSGI server)

ENDPOINTS:
- GET /xrpc/app.bsky.feed.getFeedSkeleton  → Post listesi döner
- GET /xrpc/app.bsky.feed.describeFeedGenerator  → Feed metadata
- GET /.well-known/did.json  → DID document

PERFORMANS HEDEFİ: <50ms response time
Bunun için postlar zaten skorlanmış ve veritabanında, sadece query yapılır.
"""
from flask import Flask, request, jsonify, abort
from feed_generator.feed_logic import FeedLogic
from config.settings import FEED_URI_POLITICS, FEED_URI_SCIENCE
import os

app = Flask(__name__)
feed_logic = FeedLogic()

# Feed URI → (domain, stance) mapping
FEED_ROUTES = {
    FEED_URI_POLITICS:        ("politics", None),       # Tüm siyaset
    FEED_URI_SCIENCE:         ("science", None),        # Tüm bilim
    # İsteğe bağlı stance feed'leri:
    # FEED_URI_ALLIANCE:      ("politics", "alliance"),
    # FEED_URI_OPPOSITION:    ("politics", "opposition"),
}


@app.route('/xrpc/app.bsky.feed.getFeedSkeleton')
def get_feed_skeleton():
    """
    Ana feed endpoint'i.
    
    PARAMETRE:
        feed: Feed URI (hangi feed?)
        limit: Kaç post döndürülsün? (default: 30, max: 100)
        cursor: Sayfalama için offset
    
    DÖNÜŞ:
        {
            "feed": [{"post": "at://..."}],
            "cursor": "50"
        }
    
    AUTHENTICATION:
        Bluesky Authorization header ile user DID gönderir.
        Kişiselleştirme için kullanılabilir.
    """
    feed_uri = request.args.get('feed')
    limit = min(int(request.args.get('limit', 30)), 100)
    cursor = request.args.get('cursor', None)
    
    if feed_uri not in FEED_ROUTES:
        abort(400, f"Bilinmeyen feed: {feed_uri}")
    
    domain, stance = FEED_ROUTES[feed_uri]
    
    # İsteğe bağlı: user DID ile kişiselleştirme
    # auth_header = request.headers.get('Authorization', '')
    # user_did = resolve_auth_token(auth_header)
    
    try:
        posts, next_cursor = feed_logic.get_posts(
            domain=domain,
            stance=stance,
            limit=limit,
            cursor=cursor
        )
    except Exception as e:
        print(f"Feed hatası: {e}")
        abort(500, "Feed alınamadı")
    
    return jsonify({
        "feed": [{"post": uri} for uri in posts],
        "cursor": next_cursor
    })


@app.route('/xrpc/app.bsky.feed.describeFeedGenerator')
def describe_feed_generator():
    """Feed generator metadata."""
    return jsonify({
        "did": f"did:web:{os.getenv('FEED_DOMAIN', 'yourdomain.com')}",
        "feeds": [
            {
                "uri": FEED_URI_POLITICS,
                "displayName": "Türkiye Siyaset",
                "description": "Türk siyasetçilerin ve politika tartışmalarının BERTurk ile analiz edilmiş akışı"
            },
            {
                "uri": FEED_URI_SCIENCE,
                "displayName": "Türkiye Bilim",
                "description": "Türk akademisyenler ve araştırmacıların paylaşımları"
            }
        ]
    })


@app.route('/.well-known/did.json')
def did_document():
    """
    did:web identity document.
    Bu endpoint, feed'in 'did:web:yourdomain.com' kimliğini doğrular.
    
    DEPLOY NOTLARI:
    - Bu endpoint HTTPS üzerinde erişilebilir olmalı
    - yourdomain.com'u gerçek domain'inize göre değiştirin
    """
    domain = os.getenv('FEED_DOMAIN', 'yourdomain.com')
    return jsonify({
        "@context": ["https://www.w3.org/ns/did/v1"],
        "id": f"did:web:{domain}",
        "service": [
            {
                "id": "#bsky_fg",
                "type": "BskyFeedGenerator",
                "serviceEndpoint": f"https://{domain}"
            }
        ]
    })


if __name__ == '__main__':
    # Development: Flask dev server
    app.run(host='0.0.0.0', port=3000, debug=True)
    
    # Production (waitress kullan):
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=3000, threads=4)
```

**`feed_generator/feed_logic.py`:**

```python
"""
Feed sıralama ve filtreleme mantığı.
"""
from database.models import Post
from peewee import fn
import datetime


class FeedLogic:
    """
    Feed sorgu mantığı.
    
    SIRALAMA KRİTERLERİ (öncelik sırasıyla):
    1. feed_score (domain similarity + engagement bonus)
    2. created_at (yeni postlar üstte)
    
    BONUS HESAPLAMA:
    feed_score = domain_score * 0.7 + engagement_bonus * 0.3
    engagement_bonus = log(1 + like_count + repost_count * 2) / 10
    
    NOT: Paper Skygest makalesi, feed sırasının engagement'ı 4x etkilediğini gösterdi.
    Bu nedenle sıralama algoritması kritik öneme sahiptir.
    """
    
    def get_posts(
        self,
        domain: str,            # 'politics', 'science', 'both'
        stance: str | None,     # 'alliance', 'opposition', None (hepsi)
        limit: int = 30,
        cursor: str | None = None,
        max_age_hours: int = 48  # Son 48 saatin postları
    ) -> tuple[list[str], str | None]:
        """
        Veritabanından sıralı post URI listesi döner.
        
        DÖNÜŞ:
            (post_uri_listesi, next_cursor)
        """
        offset = int(cursor) if cursor else 0
        
        # Zaman filtresi
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=max_age_hours)
        
        # Base query
        query = Post.select(Post.uri).where(
            Post.created_at >= cutoff,
            Post.language == 'tr'
        )
        
        # Domain filtresi
        if domain == 'both':
            query = query.where(Post.domain_label.in_(['politics', 'science', 'both']))
        else:
            query = query.where(Post.domain_label.in_([domain, 'both']))
        
        # Stance filtresi (opsiyonel)
        if stance:
            query = query.where(Post.stance_label == stance)
        
        # Sıralama: feed_score + zaman
        query = query.order_by(
            Post.feed_score.desc(),
            Post.created_at.desc()
        )
        
        # Sayfalama
        total = query.count()
        posts = list(query.offset(offset).limit(limit))
        post_uris = [p.uri for p in posts]
        
        # Next cursor
        next_offset = offset + limit
        next_cursor = str(next_offset) if next_offset < total else None
        
        return post_uris, next_cursor
    
    def update_feed_scores(self) -> None:
        """
        Tüm postların feed skorlarını güncelle.
        Bu fonksiyon periyodik olarak (örn: her 30 dakikada) çalıştırılmalı.
        """
        import math
        
        for post in Post.select().where(Post.domain_label != 'other'):
            # Engagement bonus
            engagement = post.like_count + post.repost_count * 2 + post.reply_count
            engagement_bonus = math.log(1 + engagement) / 10
            
            # Zaman cezası (eski postlar daha düşük skor)
            age_hours = (
                datetime.datetime.utcnow() - post.created_at
            ).total_seconds() / 3600
            time_penalty = max(0, 1 - age_hours / 48)  # 48 saatte 0'a düşer
            
            new_score = (
                post.domain_score * 0.5 +
                engagement_bonus * 0.3 +
                time_penalty * 0.2
            )
            
            Post.update(feed_score=new_score).where(Post.uri == post.uri).execute()
```

---

## 7. Adım 6: Centroid Oluşturma Scripti

**`scripts/build_domain_centroids.py`:**

```python
"""
Domain ve stance centroid'lerini oluştur.

ÇALIŞTIRMA SIRASI:
1. python scripts/build_domain_centroids.py --mode keyword
   (Hızlı başlangıç, keyword-based)

2. Birkaç gün firehose çalıştıktan sonra:
   python scripts/build_domain_centroids.py --mode posts
   (Gerçek post verisiyle daha iyi centroid)
"""
import argparse
from nlp.embedder import TurkishEmbedder
from nlp.domain_classifier import DomainClassifier
from nlp.stance_detector import StanceDetectorV1
from database.models import Post, TrackedUser, db
from atproto import Client
from config.settings import BSKY_HANDLE, BSKY_APP_PASSWORD
import os

def build_from_keywords():
    print("Keyword-based centroid oluşturuluyor...")
    embedder = TurkishEmbedder()
    classifier = DomainClassifier(embedder)
    classifier.build_centroids_from_keywords()
    
    os.makedirs("data", exist_ok=True)
    classifier.save_centroids("data/centroids.npy")
    print("✓ Domain centroid'ler hazır: data/centroids.npy")

def build_from_posts():
    print("Veritabanındaki postlardan centroid oluşturuluyor...")
    db.connect(reuse_if_open=True)
    
    embedder = TurkishEmbedder()
    classifier = DomainClassifier(embedder)
    
    # Mevcut seed kullanıcıların postlarından örnek al
    politics_posts = list(
        Post.select(Post.text)
        .where(Post.domain_label == 'politics')
        .limit(500)
    )
    science_posts = list(
        Post.select(Post.text)
        .where(Post.domain_label == 'science')
        .limit(500)
    )
    
    if len(politics_posts) < 50 or len(science_posts) < 50:
        print("⚠ Yeterli post yok, keyword-based kullanılıyor...")
        build_from_keywords()
        return
    
    classifier.build_centroids_from_texts({
        'politics': [p.text for p in politics_posts],
        'science': [p.text for p in science_posts]
    })
    classifier.save_centroids("data/centroids.npy")
    print("✓ Gerçek veriden centroid'ler hazır.")

def build_stance_centroids():
    print("Stance centroid'leri oluşturuluyor...")
    db.connect(reuse_if_open=True)
    
    client = Client()
    client.login(BSKY_HANDLE, BSKY_APP_PASSWORD)
    
    embedder = TurkishEmbedder()
    detector = StanceDetectorV1(embedder)
    
    # Seed kullanıcıları stance'a göre grupla
    alliance_users = list(
        TrackedUser.select(TrackedUser.did)
        .where(TrackedUser.stance == 'alliance')
        .limit(50)
    )
    opposition_users = list(
        TrackedUser.select(TrackedUser.did)
        .where(TrackedUser.stance == 'opposition')
        .limit(50)
    )
    
    print(f"İttifak kullanıcı: {len(alliance_users)}, Muhalefet: {len(opposition_users)}")
    
    detector.build_stance_centroids_from_users(
        client,
        {
            'alliance': [u.did for u in alliance_users],
            'opposition': [u.did for u in opposition_users]
        }
    )
    detector.save("data/stance_centroids.npy")
    print("✓ Stance centroid'ler hazır: data/stance_centroids.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['keyword', 'posts', 'stance', 'all'], default='all')
    args = parser.parse_args()
    
    if args.mode in ('keyword', 'all'):
        build_from_keywords()
    if args.mode in ('posts',):
        build_from_posts()
    if args.mode in ('stance', 'all'):
        build_stance_centroids()
```

---

## 8. Adım 7: Feed'i Bluesky'a Yayınlama

### 8.1 Feed Record Oluşturma

```python
# scripts/publish_feed.py
"""
Feed'i Bluesky hesabına kaydet.
Bu işlem bir kez yapılır.
"""
from atproto import Client
from config.settings import BSKY_HANDLE, BSKY_APP_PASSWORD
import os

def publish_feed(client: Client, feed_id: str, display_name: str, description: str):
    """
    PARAMETRE:
        feed_id: URL-safe kısa isim, örn: "turkiye-siyaset"
        display_name: Kullanıcıya görünen isim
        description: Feed açıklaması
    """
    domain = os.getenv('FEED_DOMAIN')
    generator_did = f"did:web:{domain}"
    
    response = client.app.bsky.feed.generator.create(
        repo=client.me.did,
        record={
            '$type': 'app.bsky.feed.generator',
            'did': generator_did,
            'displayName': display_name,
            'description': description,
            'createdAt': client.get_current_time_iso()
        }
    )
    
    print(f"Feed yayınlandı: {response.uri}")
    return response.uri


if __name__ == "__main__":
    client = Client()
    client.login(BSKY_HANDLE, BSKY_APP_PASSWORD)
    
    publish_feed(
        client,
        feed_id="turkiye-siyaset",
        display_name="🇹🇷 Türkiye Siyaset",
        description="Türk siyasetçi ve yorumcuların paylaşımları. BERTurk analizi ile filtrelenmiştir."
    )
    
    publish_feed(
        client,
        feed_id="turkiye-bilim",
        display_name="🔬 Türkiye Bilim",
        description="Türk akademisyen ve araştırmacıların bilimsel içerikleri."
    )
```

### 8.2 Deployment Seçenekleri

```
SEÇENEK A: Local / VPS Deployment (Önerilen başlangıç)
─────────────────────────────────────────────────────
1. VPS: DigitalOcean Droplet ($6/ay, 1GB RAM)
   veya Hetzner CX11 (€4/ay)
   
2. Domain: Cloudflare'dan ücretsiz subdomain veya
   kendi domain'iniz (feed.yourdomain.com)

3. HTTPS: Let's Encrypt (certbot) ile ücretsiz

4. Process Manager: PM2 veya systemd
   pm2 start "python feed_generator/server.py" --name feed-server
   pm2 start "python data_collection/firehose_listener.py" --name firehose

SEÇENEK B: AWS (Paper Skygest mimarisi)
─────────────────────────────────────────
- Feed Generator: AWS Lambda + API Gateway (auto-scale)
- Firehose: AWS EC2 t3.small
- Database: AWS DynamoDB (yüksek throughput)
- Cache: AWS DynamoDB'de pre-computed recommendations

SEÇENEK C: Railway / Render (Kolay deploy)
───────────────────────────────────────────
- railway.app veya render.com
- GitHub repo bağla, otomatik deploy
- PostgreSQL add-on ekle
- Ücretsiz tier başlangıç için yeterli
```

---

## 9. Tam Çalıştırma Sırası

Projeyi sıfırdan çalıştırmak için adımları sırayla takip edin:

```bash
# 1. Ortam kurulumu
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Veritabanı oluştur
python database/models.py

# 3. Türk kullanıcıları keşfet ve CSV'den yükle
python scripts/discover_turkish_users.py
# → data/discovered_users.csv oluşur

# 4. Kullanıcıları veritabanına aktar
python -c "
from database.models import TrackedUser
import pandas as pd
df = pd.read_csv('data/discovered_users.csv')
for _, row in df.iterrows():
    TrackedUser.get_or_create(did=row['did'], defaults=row.to_dict())
print('Kullanıcılar eklendi.')
"

# 5. Domain ve stance centroid'lerini oluştur
python scripts/build_domain_centroids.py --mode all
# → data/centroids.npy ve data/stance_centroids.npy oluşur

# 6. Feed generator'ı başlat (terminal 1)
python feed_generator/server.py

# 7. Firehose listener'ı başlat (terminal 2)
python data_collection/firehose_listener.py

# 8. Feed'i Bluesky'a yayınla (bir kez)
python scripts/publish_feed.py

# 9. (Opsiyonel) Stance fine-tuning için veri yeterince toplandıktan sonra:
# python scripts/fine_tune_stance.py
```

---

## 10. Test ve Doğrulama

```python
# scripts/test_pipeline.py
"""
Pipeline'ı test et.
"""
from nlp.pipeline import NLPPipeline
import datetime

pipeline = NLPPipeline()
pipeline.load_models()

test_posts = [
    # Beklenen: politics, alliance
    ("Cumhurbaşkanı Erdoğan bugün TBMM'de önemli açıklamalarda bulundu.", "politics", "alliance"),
    # Beklenen: politics, opposition  
    ("CHP Genel Başkanı muhalefet partilerinin birlikte hareket etmesi gerektiğini söyledi.", "politics", "opposition"),
    # Beklenen: science
    ("Yeni makaleimiz Nature'da yayınlandı! arXiv preprint linki profilde.", "science", "neutral"),
    # Beklenen: other (Türkçe değil)
    ("Had a great day at the conference today!", None, None),
]

print("Pipeline Test Sonuçları:")
print("=" * 60)
for text, expected_domain, expected_stance in test_posts:
    post = pipeline.process_post(
        uri=f"at://test/app.bsky.feed.post/{hash(text)}",
        cid="test_cid",
        author_did="did:plc:test",
        author_handle="test.bsky.social",
        text=text,
        created_at=datetime.datetime.utcnow()
    )
    
    if post:
        status = "✓" if post.domain_label == expected_domain else "✗"
        print(f"{status} Domain: {post.domain_label} (beklenen: {expected_domain})")
        print(f"  Stance: {post.stance_label} | Score: {post.domain_score:.3f}")
        print(f"  Metin: {text[:60]}...")
    else:
        status = "✓" if expected_domain is None else "✗"
        print(f"{status} Post filtrelendi (beklenen: {expected_domain})")
    print()
```

---

## 11. Önemli Notlar ve Tuzaklar

### Rate Limiting
- Bluesky API: 3000 istek/5 dakika
- Firehose'da rate limit yoktur (WebSocket bağlantısı)
- `time.sleep(0.3)` her API çağrısı arasına ekle

### Model Seçimi Önerileri
| Model | Hız | Türkçe Kalitesi | Önerilen Kullanım |
|-------|-----|-----------------|-------------------|
| `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` | Orta | ⭐⭐⭐⭐⭐ | Domain classifier (önerilen) |
| `dbmdz/bert-base-turkish-cased` | Orta | ⭐⭐⭐⭐ | Fine-tuning base model |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Hızlı | ⭐⭐⭐ | Hız öncelikliyse fallback |

### Paper Skygest'ten Öğrenilenler (Kaynak: arXiv:2601.04253v1)
1. **Feed sırası kritiktir**: İlk pozisyondaki post 4x daha fazla like alır. Sıralama algoritmasına zaman ayırın.
2. **Başlangıç gecikmesi**: İlk prototip 2 günde yapılır ama production'a hazır olması aylar alır.
3. **Cold start**: Yeni kullanıcılar için default feed göster (tüm kullanıcıların göreceği genel feed).
4. **Bot filtreleme**: Firehose'da çok sayıda bot hesabı var. Handle'da "bot" geçenleri ve çok sık post atanları filtrele.
5. **50ms latency hedefi**: Önerileri önceden hesapla ve cache'le, asla gerçek zamanlı hesaplama yapma.
6. **Pagination**: `cursor` parametresini mutlaka implemente et.

### Türkçe NLP Özel Durumlar
- Türkçe büyük/küçük harf: `text.lower()` genellikle yeterli
- Özel karakterler: `ğ, ş, ı, ö, ü, ç` Unicode'da sorunsuz
- Kısaltmalar: "tbmm", "chp", "akp" keyword listesine küçük harfle ekle
- Dialect: Sosyal medyada yazım hataları ve argo yaygın → BERTurk bu konuda robust

---

## 12. Proje Dosya Listesi (Özet)

```
Oluşturulacak dosyalar (öncelik sırasıyla):
─────────────────────────────────────────────
ÖNCE:
  config/settings.py              ← Tüm konfigürasyon
  database/models.py              ← Veritabanı şeması
  requirements.txt                ← Bağımlılıklar

SONRA (NLP):
  nlp/embedder.py                 ← BERTurk embedding
  nlp/domain_classifier.py        ← Politics/Science sınıflandırması
  nlp/stance_detector.py          ← Alliance/Opposition tespiti
  nlp/pipeline.py                 ← Birleşik pipeline

SONRA (Data):
  data_collection/firehose_listener.py   ← Gerçek zamanlı veri
  data_collection/seed_discovery.py      ← Kullanıcı keşfi
  scripts/build_domain_centroids.py      ← Model hazırlığı

SON OLARAK (Server):
  feed_generator/server.py        ← HTTP server
  feed_generator/feed_logic.py    ← Sıralama mantığı
  scripts/publish_feed.py         ← Bluesky'a yayınlama
```

---

*Bu döküman, Paper Skygest (arXiv:2601.04253v1) ve Bluesky AT Protocol dökümentasyonu referans alınarak hazırlanmıştır.*
*GitHub: https://github.com/Skygest/PaperSkygest (referans mimari)*