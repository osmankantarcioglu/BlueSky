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
        Build centroids from tweet-style Turkish sentences.

        Sentences are written to resemble real Bluesky/Twitter posts:
        short, colloquial, with Turkish diacritics, hashtags, and first-person voice.
        """
        politics_texts = [
            # Short reaction-style tweets
            "Erdoğan bugün TBMM'de konuştu, muhalefet sert tepki verdi",
            "CHP'nin açıklaması gündem oldu",
            "AKP-MHP ittifakı çatladı mı?",
            "Yeni anayasa tartışması yeniden başladı",
            "Seçim ne zaman? Herkes merak ediyor",
            "Belediye başkanı görevden alındı, kayyum atandı",
            "Meclis bugün oyladı, kanun geçti",
            "Bakan istifa etti",
            "Muhalefet mitinge çıktı, katılım büyük",
            "Sandıkta ne olacak? Anketler çelişiyor",
            # First-person political commentary
            "Bu seçim sonuçları çok şaşırttı beni",
            "İmamoğlu davasını yakından takip ediyorum",
            "Siyasi gerilim had safhada şu an",
            "Hükümetin bu kararına gerçekten şaşırdım",
            "Muhalefet bir türlü birleşemiyor, üzücü",
            "AKP'nin oy oranı düşüyor mu yoksa?",
            "Kılıçdaroğlu ne dedi bugün?",
            "Bahçeli'nin açıklaması dikkat çekiciydi",
            "Özgür Özel TBMM'de sert konuştu",
            "Demirtaş cezaevinden açıklama yaptı",
            # News-style short posts
            "TBMM'de bütçe görüşmeleri sürüyor",
            "Yeni vergi paketi meclise sunuldu",
            "Anayasa mahkemesi kararı bekleniyor",
            "Dışişleri bakanı NATO zirvesine katıldı",
            "Yerel seçimler için adaylar açıklanıyor",
            "Koalisyon müzakereleri tıkandı",
            "Erken seçim senaryoları konuşuluyor",
            "Siyasi kriz derinleşiyor",
            "Belediye ihalelerinde yolsuzluk iddiası",
            "Referandum tartışması yeniden gündemde",
            # Hashtag/mention style
            "#seçim2024 sonuçları açıklandı",
            "#TBMM gündeminde neler var bugün",
            "Türk siyasetinde yeni bir dönem mi başlıyor",
            "Hükümet ekonomi paketini açıkladı, piyasalar karışık",
            "Muhalefet partileri ortak bildiri yayınladı",
            "Cumhurbaşkanı: 'Seçim zamanında yapılacak'",
            "Milletvekili dokunulmazlıkları tekrar gündemde",
            "Siyasi gerilim tırmanıyor, çözüm nerede?",
            "Parti içi muhalefet güçleniyor",
            "Türkiye-AB ilişkileri yeniden nasıl şekillenecek",
            # More casual/opinionated
            "bu siyasetçiler hiç değişmiyor ya",
            "yine aynı söylemler, farklı yüzler",
            "seçmenin sesini kim dinleyecek",
            "iktidar muhalefet kavgası bıktırdı",
            "sandık her şeyi çözer diyenler haklı mı",
        ]

        science_texts = [
            # First-person researcher tweets
            "Yeni makalemiz arXiv'e yüklendi, link biyoda",
            "Nature'da yayınlanan çalışmamız çok ilgi gördü",
            "Doktora savunmam geçti! Çok mutluyum 🎉",
            "Konferansta posterimizi sunduk, güzel geri dönüşler aldık",
            "Veri setimizi açık kaynak olarak paylaşıyoruz",
            "Makine öğrenmesi modelimiz yüzde 94 doğruluk verdi",
            "Yeni paper yayınlandı, abstract thread'de",
            "Hakemler makalemi kabul etti nihayet!",
            "Laboratuvarda ilginç sonuçlar çıktı bugün",
            "Tezimin son bölümünü yazıyorum, bitiyorum artık",
            # Short science news commentary
            "Bu çalışma derin öğrenme alanında çığır açıyor",
            "CERN'den yeni bulgular geldi, fiziği değiştirebilir",
            "Türk araştırmacılar Science'da yayın yaptı, gurur verici",
            "İklim değişikliği verilerini analiz ettik, tablo kötü",
            "COVID varyantları üzerine yeni analiz yayınlandı",
            "Yapay zeka artık protein yapısını tahmin edebiliyor",
            "Kuantum hesaplama bir adım daha ilerledi",
            "Bu algoritma klasik yöntemden 10x daha hızlı",
            "Biyoteknoloji harika şeyler yapıyor son dönemde",
            "Uzay teleskobundan inanılmaz görüntüler geldi",
            # Academic community style
            "NLP modelimizi Türkçe üzerine fine-tune ettik",
            "Peer review süreci çok uzadı, 8 ay oldu",
            "Konferans bildiri kabul oranı %18, çok zor",
            "Açık erişim bilimin geleceği, keşke herkes benimsese",
            "Veri analizinde yeni bir yöntem kullandık, çok işe yaradı",
            "Araştırma grubumuz 3 yeni üye aldı bu dönem",
            "Meta-analiz sonuçları ilginç çıktı, beklenmedik",
            "Simülasyon modeli gerçek verilerle örtüştü",
            "Hipotezimiz doğrulandı, şimdi makale yazıyoruz",
            "İstatistiksel analizde p<0.001 çıktı, güçlü bulgu",
            # Hashtag/thread style
            "#YapayZeka son gelişmeler hakkında bir thread",
            "#bilim insanları bu yıl çok önemli keşifler yaptı",
            "Makale özeti: yeni bir derin öğrenme yaklaşımı",
            "Araştırma sorusu: dil modelleri gerçekten anlıyor mu",
            "Doktora öğrencileri için veri analizi ipuçları",
            "Hesaplamalı biyoloji alanında yeni araçlar çıktı",
            "Fizik deneyi sonuçları standart modeli zorluyor",
            "Kimya literatüründe yeni bir sentez yöntemi",
            "Tıp araştırması: ilaç yan etkileri beklentiden az",
            "Sosyal bilimlerde büyük veri kullanımı artıyor",
            # Casual researcher voice
            "bugün lab'da güzel veriler topladık",
            "makale revizyon geldi, yeniden başlıyorum",
            "konferans kabul maili geldi, çok sevindim",
            "veri temizleme işi hiç bitmez ya",
            "model overfitting yapıyor, başım belada",
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

        if len(above) >= 1:
            # Always pick the single highest-scoring domain — never return 'both'
            best_domain, best_score = max(above.items(), key=lambda x: x[1])
            return best_domain, best_score

        # No domain exceeded the similarity threshold — discard post
        return 'other', 0.0
