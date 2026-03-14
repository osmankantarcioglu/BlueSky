"""
Political stance detection for Turkish posts.

METHOD 1 (StanceDetectorV1) - Recommended for initial use:
    Keyword + centroid-based, no training data required.
    Stance centroids are built from posts of known-party politicians.

METHOD 2 (StanceDetectorV2) - Advanced:
    Fine-tuned BERTurk classifier.
    Requires ~500 labeled posts per class and GPU for practical training time.
    See scripts/fine_tune_stance.py for training instructions.
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
    Keyword + centroid-based stance detection. No fine-tuning required.

    Stance centroids are computed from posts of seed users whose party
    affiliation is known (from the CSV seed list).

    Usage:
        detector = StanceDetectorV1(embedder)
        detector.load("data/stance_centroids.npy")
        stance, score = detector.detect_stance("CHP lideri muhalefeti elestirdi.")
        # stance = 'opposition', score = 0.72
    """

    def __init__(self, embedder: TurkishEmbedder):
        self.embedder = embedder
        self.stance_centroids = {}

    def build_stance_centroids_from_users(self, client, stance_users: dict) -> None:
        """
        Fetch recent posts from known-party users and compute stance centroids.

        This runs once and is slow (~1-2 hours due to rate limits).
        Results are saved and reloaded on subsequent runs.

        Args:
            client:       atproto Client (already logged in)
            stance_users: {'alliance': [did1, did2, ...], 'opposition': [...]}
                          At least 20 users per group recommended.
        """
        import time

        for stance, dids in stance_users.items():
            all_texts = []

            for did in dids[:50]:  # Max 50 users per group
                try:
                    feed = client.app.bsky.feed.get_author_feed({
                        "actor": did,
                        "limit": 20,
                        "filter": "posts_no_replies"
                    })
                    for item in feed.feed:
                        text = item.post.record.text
                        if len(text) > 20:
                            all_texts.append(text)
                    time.sleep(0.3)
                except Exception:
                    continue

            if all_texts:
                print(f"Building centroid for '{stance}' from {len(all_texts)} posts...")
                embeddings = self.embedder.embed_batch(all_texts)
                centroid = embeddings.mean(axis=0)
                centroid /= np.linalg.norm(centroid)
                self.stance_centroids[stance] = centroid
                print(f"Centroid ready for '{stance}'.")

    def build_stance_centroids_from_keywords(self) -> None:
        """
        Build stance centroids quickly from keyword-based example sentences.
        Use this before real post data is available.
        """
        alliance_texts = [
            f"Bugun {kw} hakkinda guzel gelismeler yasandi." for kw in ALLIANCE_KEYWORDS
        ] + [
            "Cumhur ittifaki ulkenin istikrari icin onemli adimlar atiyor.",
            "Erdogan liderliginde Turkiye guclu bir sekilde yoluna devam ediyor.",
            "Hukumetin yeni yatirim paketleri ekonomiye guc katiyor.",
            "Cumhur ittifaki milli iradeyi temsil etmeye devam ediyor.",
            "Devletin guclenmesi icin atilan adimlar halktan destek goruyor.",
            "Iktidar partisi savunma sanayisinde buyuk basarilar elde etti.",
            "Yeni projeler Turkiye'nin gelecegi icin umut veriyor.",
            "Cumhurbaskani Erdogan ulke icin tarihi kararlar almaya devam ediyor.",
            "Hukumetin ekonomi programi istikrar saglamayi hedefliyor.",
            "Cumhur ittifaki birlik ve beraberlik mesajlari veriyor.",
            "Turkiye savunma sanayisinde yeni bir basari hikayesi yaziyor.",
            "Devlet projeleri sayesinde ulke kalkinmaya devam ediyor.",
            "Iktidar partisi ulusal guvenligi onceleyen politikalar izliyor.",
            "Cumhur ittifaki secimlerde guclu bir destek aldi.",
            "Erdogan'in liderligi ulke icin onemli bir avantaj olarak goruluyor.",
            "Yeni otoyol ve altyapi projeleri hizla ilerliyor.",
            "Turkiye enerji alaninda stratejik adimlar atiyor.",
            "Iktidar ulkenin ekonomik buyumesine odaklaniyor.",
            "Cumhur ittifaki yerli ve milli projeleri destekliyor.",
            "Turkiye'nin savunma projeleri dunya genelinde dikkat cekiyor.",
            "Cumhurbaskani yeni yatirimlari duyurdu.",
            "Ulkenin kalkinmasi icin yeni stratejiler belirleniyor.",
            "Cumhur ittifaki secmen tabanini genisletmeye devam ediyor.",
            "Hukumet yeni sosyal destek paketini acikladi.",
            "Erdogan uluslararasi zirvede Turkiye'yi temsil etti.",
            "Yerli teknoloji projeleri hiz kesmeden devam ediyor.",
            "Cumhur ittifaki ulke icin uzun vadeli planlar yapiyor.",
            "Iktidar partisi ulke ekonomisini guclendirmeyi hedefliyor.",
            "Turkiye bolgesel guc olma yolunda ilerliyor.",
            "Yeni savunma sistemleri ulkenin guvenligini artiriyor.",
            "Erdogan yeni reform paketini duyurdu.",
            "Hukumetin politikasi istikrar ve buyume uzerine kurulu.",
            "Cumhur ittifaki yeni secim stratejisini acikladi.",
            "Turkiye'nin diplomatik girisimleri dikkat cekiyor.",
            "Iktidar partisi ulusal projeleri hizlandirdi.",
            "Cumhurbaskani ulkeye yeni yatirimlar kazandirdi.",
            "Turkiye uluslararasi alanda daha etkili rol oynuyor.",
            "Hukumet yeni ekonomik destek programini baslatti.",
            "Cumhur ittifaki birlik mesajlari veriyor.",
            "Erdogan yeni kalkinma planini tanitti.",
            "Yerli savunma teknolojileri buyuk ilerleme kaydetti.",
            "Turkiye'nin yeni enerji stratejisi aciklandi.",
            "Iktidar partisi yeni yatirim projeleri duyurdu.",
            "Cumhur ittifaki secmen destegini koruyor.",
            "Turkiye bolgesinde guclu bir aktor olmaya devam ediyor.",
            "Hukumet yeni sosyal yardim paketini duyurdu.",
            "Erdogan yeni diplomatik temaslarda bulundu.",
            "Cumhur ittifaki yeni politikalar gelistiriyor.",
            "Turkiye'nin ekonomik hedefleri yuksek tutuluyor.",
            "Iktidar ulkenin kalkinmasi icin projeler uretiyor."
        ]

        opposition_texts = [
            f"Bugun {kw} hakkinda onemli aciklamalar yapildi." for kw in OPPOSITION_KEYWORDS
        ] + [
            "Muhalefet hukumetin ekonomi politikalarini elestirmeye devam ediyor.",
            "CHP yeni secim stratejisini kamuoyuna tanitti.",
            "Ekrem Imamoglu Istanbul icin yeni projeler acikladi.",
            "Muhalefet liderleri demokrasi vurgusu yapti.",
            "Ekonomik kriz vatandaslari zor durumda birakiyor.",
            "Muhalefet partileri erken secim cagrisini yineledi.",
            "CHP lideri hukumetin politikalarini sert sekilde elestirdi.",
            "Muhalefet toplumsal adalet vurgusu yapti.",
            "Demokrasi ve hukuk devleti talepleri gundemde.",
            "Muhalefet liderleri ortak basin toplantisi duzenledi.",
            "Ekrem Imamoglu sehir yonetimi icin yeni planlar anlatti.",
            "Muhalefet secmen destegini artirmayi hedefliyor.",
            "Ekonomik sorunlar mecliste gundeme getirildi.",
            "CHP milletvekilleri yeni yasa teklifini sundu.",
            "Muhalefet partileri ortak strateji belirliyor.",
            "Ekrem Imamoglu yeni sosyal projeleri duyurdu.",
            "Muhalefet liderleri secim icin sahaya indi.",
            "CHP hukumetin ekonomi yonetimini elestirdi.",
            "Muhalefet adalet ve seffaflik talep ediyor.",
            "Ekrem Imamoglu yeni yatirim projeleri acikladi.",
            "Muhalefet halkin ekonomik sikintilarina dikkat cekiyor.",
            "CHP yeni reform onerilerini paylasti.",
            "Muhalefet liderleri secim kampanyasina hazirlaniyor.",
            "Ekonomik kriz siyasetin ana gundem maddesi oldu.",
            "Muhalefet partileri ortak bildiri yayinladi.",
            "Ekrem Imamoglu sosyal yardim projelerini anlatti.",
            "CHP lideri yeni secim vizyonunu acikladi.",
            "Muhalefet ekonomik reform taleplerini dile getirdi.",
            "Ekrem Imamoglu yerel yonetim projelerini tanitti.",
            "Muhalefet hukumeti hesap vermeye davet etti.",
            "CHP yeni politika belgesini yayinladi.",
            "Muhalefet demokrasi ve ozgurluk vurgusu yapti.",
            "Ekrem Imamoglu yeni ulasim projelerini duyurdu.",
            "Muhalefet partileri ortak miting duzenledi.",
            "CHP lideri halk bulusmalarina devam ediyor.",
            "Muhalefet ekonomik esitlik talebini yineledi.",
            "Ekrem Imamoglu Istanbul icin yeni planlarini anlatti.",
            "Muhalefet secim icin yeni strateji belirliyor.",
            "CHP yeni anayasa tartismalarini gundeme getirdi.",
            "Muhalefet hukumetin politikalarini elestirdi.",
            "Ekrem Imamoglu yeni sosyal destek projelerini duyurdu.",
            "Muhalefet liderleri ulke sorunlarini tartisti.",
            "CHP yeni ekonomi programini tanitti.",
            "Muhalefet partileri secim ittifaki uzerinde calisiyor.",
            "Ekrem Imamoglu sehir planlama projelerini paylasti.",
            "Muhalefet hukumetin kararlarini sorguluyor.",
            "CHP lideri yeni politikalarini anlatti.",
            "Muhalefet halkin taleplerini dile getirdi.",
            "Ekrem Imamoglu yeni kentsel donusum projelerini acikladi.",
            "Muhalefet demokrasi icin birlikte hareket ediyor."
        ]

        self.build_centroids_from_texts({
            'alliance': alliance_texts,
            'opposition': opposition_texts
        })

    def build_centroids_from_texts(self, texts_by_stance: dict) -> None:
        """Compute and store stance centroids from labeled text lists."""
        for stance, texts in texts_by_stance.items():
            print(f"Embedding {len(texts)} texts for stance '{stance}'...")
            embeddings = self.embedder.embed_batch(texts)
            centroid = embeddings.mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            self.stance_centroids[stance] = centroid
        print("Stance centroids built.")

    def detect_stance(self, text: str, embedding: np.ndarray = None) -> tuple:
        """
        Detect political stance of a post.

        Args:
            text:      Post text
            embedding: Pre-computed embedding (computed here if None)

        Returns:
            ('alliance' | 'opposition' | 'neutral', confidence_score)

        Note: Call this only for posts where domain_label == 'politics'.
        """
        if embedding is None:
            embedding = self.embedder.embed(text)

        # Fall back to keywords if no centroids loaded
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
            # Low confidence → try keyword fallback
            kw_stance, kw_score = self._keyword_fallback(text)
            if kw_stance != 'neutral':
                return kw_stance, kw_score
            return 'neutral', best_score

        return best_stance, best_score

    def _keyword_fallback(self, text: str) -> tuple:
        """Simple keyword-count-based stance detection."""
        text_lower = text.lower()
        alliance_count = sum(1 for kw in ALLIANCE_KEYWORDS if kw in text_lower)
        opposition_count = sum(1 for kw in OPPOSITION_KEYWORDS if kw in text_lower)

        if alliance_count > opposition_count:
            return 'alliance', min(0.5 + alliance_count * 0.1, 0.9)
        elif opposition_count > alliance_count:
            return 'opposition', min(0.5 + opposition_count * 0.1, 0.9)
        elif alliance_count == opposition_count > 0:
            return 'neutral', 0.5
        else:
            return 'neutral', 0.3

    def save(self, path: str = "data/stance_centroids.npy") -> None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, self.stance_centroids)
        print(f"Stance centroids saved: {path}")

    def load(self, path: str = "data/stance_centroids.npy") -> None:
        self.stance_centroids = np.load(path, allow_pickle=True).item()
        print(f"Stance centroids loaded: {list(self.stance_centroids.keys())}")


class StanceDetectorV2:
    """
    Fine-tuned BERTurk stance classifier.

    Requirements:
    - At least 500 labeled posts per class (alliance/opposition/neutral)
    - ~30 min GPU or ~4 hours CPU for fine-tuning

    Training steps:
    1. Collect posts from seed politicians
    2. Label each post by the user's party (AKP → alliance, CHP → opposition)
    3. Run scripts/fine_tune_stance.py
    4. Load the saved model here
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

    def detect_stance(self, text: str) -> tuple:
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
