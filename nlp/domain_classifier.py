"""
Cosine similarity-based domain classification.
Each post is classified by its distance to pre-computed domain centroids
(average embedding vectors for 'politics' and 'science').

CENTROID BUILDING:
1. Collect many example politics and science posts
2. Compute the mean embedding of each group → centroid
3. Save centroids to a .npy file for reuse
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nlp.embedder import TurkishEmbedder
from config.settings import (
    DOMAIN_SIMILARITY_THRESHOLD,
    POLITICS_KEYWORDS,
    SCIENCE_KEYWORDS
)


class DomainClassifier:
    """
    Zero-shot domain classifier using centroid vectors.

    Two methods are supported:
    1. Centroid-based: cosine similarity to seed-text centroids
    2. Keyword fallback: simple keyword matching when similarity is too low

    Usage:
        classifier = DomainClassifier(embedder)
        classifier.load_centroids("data/centroids.npy")
        label, score = classifier.classify("Erdogan TBMM'de konusma yapti")
        # label = 'politics', score = 0.78
    """

    def __init__(self, embedder: TurkishEmbedder):
        self.embedder = embedder
        self.centroids = {}  # {'politics': np.array([...]), 'science': np.array([...])}

    def build_centroids_from_texts(self, texts_by_domain: dict) -> None:
        """
        Compute and store centroids from labeled example texts.

        Args:
            texts_by_domain: {'politics': ["...", ...], 'science': ["...", ...]}

        At least 50 examples per domain are recommended for reliable results.
        """
        for domain, texts in texts_by_domain.items():
            print(f"Embedding {len(texts)} texts for domain '{domain}'...")
            embeddings = self.embedder.embed_batch(texts)
            centroid = embeddings.mean(axis=0)
            centroid /= np.linalg.norm(centroid)  # Normalize to unit vector
            self.centroids[domain] = centroid
        print("Centroids built.")

    def build_centroids_from_keywords(self) -> None:
        """
        Build centroids quickly from keyword lists without real post data.

        Use this at startup before enough posts have been collected.
        Replace with real post data once available for better accuracy.
        """
        politics_texts = [
            f"Bugun {kw} hakkinda onemli gelisme oldu." for kw in POLITICS_KEYWORDS
        ] + [
            "Meclis te yeni yasa teklifi kabul edildi.",
            "Cumhurbaskani aciklamasinda muhalefeti elestirdi.",
            "Secim kampanyasi surecinde partiler tartisiyor.",
            "TBMM genel kurulunda butce gorusmeleri yapildi.",
            "Hukumet yeni ekonomi paketini acikladi.",
            "Muhalefet partileri ortak bildiri yayinladi.",
            "Cumhurbaskani bugun TBMM'de yeni anayasa tartismalarina degindi.",
            "Meclis genel kurulunda butce gorusmeleri devam ediyor.",
            "Yeni yasa tasarisi parlamentoda oylamaya sunuldu.",
            "Secim kampanyasi surecinde partiler arasi tartismalar artti.",
            "Hukumet ekonomi politikalarini yeniden duzenlemeyi planliyor.",
            "Muhalefet lideri bugunku toplantida sert aciklamalar yapti.",
            "Yerel secimler icin aday belirleme sureci basladi.",
            "Parlamentoda yeni reform paketi gorusuluyor.",
            "Dis politika konusunda yeni diplomatik gorusmeler yapildi.",
            "Partiler arasi koalisyon gorusmeleri devam ediyor.",
            "Secim sonuclari siyasi dengeleri degistirdi.",
            "Hukumetin yeni vergi politikalari tartisma yaratti.",
            "TBMM komisyonunda yeni kanun teklifleri inceleniyor.",
            "Siyasi partiler secim stratejilerini belirliyor.",
            "Cumhurbaskani ulusa seslenis konusmasi yapti.",
            "Yeni anayasa degisikligi teklif edildi.",
            "Parti liderleri televizyon tartisma programina katildi.",
            "Sehir belediye baskanligi icin adaylar aciklandi.",
            "Siyasi analistler secim sonuclarini degerlendiriyor.",
            "Parlamenter sistem tartismalari yeniden gundeme geldi.",
            "Yeni dis politika stratejisi aciklandi.",
            "Milletvekilleri yeni yasa teklifini tartisti.",
            "Siyasi partiler ortak bildiri yayinladi.",
            "Hukumet yeni sosyal politika paketini tanitti.",
            "Secim mitinglerinde binlerce kisi toplandi.",
            "Mecliste uzun sureli tartismalar yasandi.",
            "Yeni siyasi ittifak kurulacagi iddia edildi.",
            "Anayasa mahkemesi onemli bir karar verdi.",
            "Siyasi liderler ekonomi politikalarini tartisti.",
            "Parti kongresinde yeni yonetim secildi.",
            "Yeni secim kanunu teklif edildi.",
            "Cumhurbaskani yabanci liderlerle gorustu.",
            "Dis iliskiler konusunda yeni acilimlar planlaniyor.",
            "Siyasi partiler secim kampanyasini baslatti.",
            "Hukumet yeni reform paketini duyurdu.",
            "Yerel yonetimler icin yeni duzenleme yapildi.",
            "Parlamentoda uzun sureli oylamalar yapildi.",
            "Siyasi tartismalar sosyal medyada buyuk yankı buldu.",
            "Milletvekilleri yeni yasa teklifini destekledi.",
            "Ekonomi politikasi konusunda farkli gorusler dile getirildi.",
            "Parti liderleri secim stratejisini anlatti.",
            "Meclis komisyonu yeni tasariyi kabul etti.",
            "Yeni anayasa calismalari baslatildi.",
            "Siyasi liderler ulusal guvenlik konusunu tartisti.",
            "Dis politika konusunda kritik bir zirve yapildi.",
            "Hukumet yeni kalkinma planini tanitti.",
            "Parti icinde liderlik yarisi basladi.",
            "Secim sonuclari siyasi dengeleri etkiledi.",
            "Siyasi yorumcular yeni donemi analiz ediyor.",
            "Parlamentoda yeni yasa tasarisi kabul edildi."
        ]

        science_texts = [
            f"Bu {kw} konusundaki yeni calismam yayinlandi." for kw in SCIENCE_KEYWORDS
        ] + [
            "arXiv de yeni makalemiz cikti, baglanti profilde.",
            "Nature dergisinde Turk arastirmacilarin calismasi.",
            "Bu arastirmanin bulgulari dikkat cekici sonuclar gosteriyor.",
            "Doktora tezim kabul edildi, tesekkurler.",
            "Konferansta sunumumuz buyuk ilgi gordu.",
            "Yeni veri seti ve metodoloji makalesi yayinlandi.",
            "Bu arastirmanin sonuclari yeni bilimsel bulgular ortaya koyuyor.",
            "Yeni makalemiz arXiv'de yayinlandi.",
            "Nature dergisinde yayinlanan calisma buyuk ilgi gordu.",
            "Bu deney sonucunda yeni bir metodoloji gelistirildi.",
            "Bilim insanlari yeni bir veri seti olusturdu.",
            "Arastirmacilar yeni bir algoritma gelistirdi.",
            "Laboratuvarda yapilan deneyler dikkat cekici sonuclar verdi.",
            "Bu calisma makine ogrenmesi alaninda yeni bir yaklasim sunuyor.",
            "Konferansta sundugumuz arastirma buyuk ilgi gordu.",
            "Yeni veri analizi yontemi gelistirildi.",
            "Bu makale derin ogrenme modellerini inceliyor.",
            "Arastirma ekibi yeni bir hipotez ortaya koydu.",
            "Bilimsel calisma yeni bulgular ortaya cikardi.",
            "Bu tez yapay zeka uygulamalarini inceliyor.",
            "Yeni bir veri seti kamuya acik olarak paylasildi.",
            "Arastirmacilar deneysel sonuclari analiz etti.",
            "Bu calisma biyoteknoloji alaninda ilerleme sagliyor.",
            "Bilim insanlari yeni bir model gelistirdi.",
            "Makale hakemli dergide yayinlandi.",
            "Arastirma sonucunda yeni bir teori gelistirildi.",
            "Yeni bir deney tasarimi onerildi.",
            "Bu calisma istatistiksel analiz yontemlerini kullaniyor.",
            "Bilimsel konferansta yeni bulgular sunuldu.",
            "Yeni bir simülasyon modeli gelistirildi.",
            "Arastirma makalesi kabul edildi.",
            "Yeni bir veri madenciligi yontemi tanitildi.",
            "Bilim insanlari yeni bir deney gerceklestirdi.",
            "Arastirmacilar yeni bir veri analizi araci gelistirdi.",
            "Makale bilimsel toplulukta tartisma yaratti.",
            "Bu calisma fizik alaninda yeni bir model oneriyor.",
            "Yeni bir matematiksel model gelistirildi.",
            "Arastirma sonuclari bilimsel dergide yayinlandi.",
            "Bilimsel makale hakem surecinden gecti.",
            "Yeni bir hesaplamali model gelistirildi.",
            "Arastirma yeni veri analizi tekniklerini kullaniyor.",
            "Bilim insanlari yeni bir deney duzenegi kurdu.",
            "Yeni bir veri toplama metodu gelistirildi.",
            "Arastirmacilar buyuk veri setlerini analiz etti.",
            "Bu calisma yapay zeka alaninda yeni yaklasimlar sunuyor.",
            "Yeni bir deneysel yontem tanitildi.",
            "Arastirma ekibi yeni bulgular paylasti.",
            "Bilimsel makale konferansta sunuldu.",
            "Yeni bir makine ogrenmesi modeli egitildi.",
            "Arastirmacilar yeni bir veri seti yayinladi.",
            "Bu calisma istatistiksel modelleme kullaniyor.",
            "Yeni bir bilimsel hipotez test edildi.",
            "Arastirma yeni metodolojik yaklasimlar sunuyor.",
            "Bilimsel topluluk yeni bulgulari tartisiyor.",
            "Yeni bir deneysel analiz yapildi.",
            "Arastirma makalesi bilimsel dergide yayinlandi."
        ]

        self.build_centroids_from_texts({
            'politics': politics_texts,
            'science': science_texts
        })

    def save_centroids(self, path: str = "data/centroids.npy") -> None:
        np.save(path, self.centroids)
        print(f"Centroids saved: {path}")

    def load_centroids(self, path: str = "data/centroids.npy") -> None:
        self.centroids = np.load(path, allow_pickle=True).item()
        print(f"Centroids loaded: {list(self.centroids.keys())}")

    def classify(self, text: str, embedding: np.ndarray = None) -> tuple:
        """
        Classify text as 'politics', 'science', 'both', or 'other'.

        Args:
            text:      Raw post text
            embedding: Pre-computed embedding (computed here if None)

        Returns:
            (label, score) where score is the highest cosine similarity

        Logic:
            1. Compute embedding
            2. Compare to each domain centroid via cosine similarity
            3. If both exceed threshold → 'both'
            4. If only one exceeds threshold → that domain
            5. If neither → keyword fallback
            6. If no keywords match → 'other'
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

        above = {d: s for d, s in scores.items() if s >= DOMAIN_SIMILARITY_THRESHOLD}

        if len(above) == 2:
            # Only return 'both' when scores are very close (within 0.04)
            # Otherwise prefer the clearly dominant domain
            sorted_above = sorted(above.items(), key=lambda x: x[1], reverse=True)
            if sorted_above[0][1] - sorted_above[1][1] < 0.04:
                return 'both', sorted_above[0][1]
            else:
                return sorted_above[0][0], sorted_above[0][1]
        elif len(above) == 1:
            label = list(above.keys())[0]
            return label, above[label]
        else:
            # Fallback: keyword matching
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
