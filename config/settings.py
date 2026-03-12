import os
from dotenv import load_dotenv
load_dotenv()

# Bluesky account credentials
BSKY_HANDLE = os.getenv("BSKY_HANDLE")
BSKY_APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")

# Feed domain
FEED_DOMAIN = os.getenv("FEED_DOMAIN", "yourdomain.com")

# Feed URIs (update with real values after publishing)
FEED_URI_POLITICS = f"at://did:web:{FEED_DOMAIN}/app.bsky.feed.generator/turkiye-siyaset"
FEED_URI_SCIENCE  = f"at://did:web:{FEED_DOMAIN}/app.bsky.feed.generator/turkiye-bilim"

# Model identifiers
BERTURK_MODEL = "dbmdz/bert-base-turkish-cased"
EMBEDDING_MODEL = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
EMBEDDING_DIM = 768

# Database
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/feeds.db")

# NLP thresholds
DOMAIN_SIMILARITY_THRESHOLD = 0.35   # Posts above this score are added to the domain feed
STANCE_CONFIDENCE_THRESHOLD = 0.60   # Minimum confidence for stance classification

# Feed settings
MAX_FEED_POSTS = 200       # Maximum number of posts kept in the feed
FEED_CACHE_TTL = 300       # Cache duration in seconds
MIN_TURKISH_PROB = 0.80    # langdetect Turkish probability threshold

# Keyword lists
POLITICS_KEYWORDS = [
"meclis","tbmm","milletvekili","vekil","parlamento","genel kurul",
"cumhurbaşkanı","cb","reis","seçim","erken seçim","yerel seçim",
"oy","sandık","seçmen","oylama","referandum",

"parti","siyasi parti","iktidar","hükümet","muhalefet",
"koalisyon","ittifak","siyasi kriz",

"akp","ak parti","mhp","chp","hdp","dem","dem parti","iyiparti",
"iyi parti","zafer partisi","gelecek partisi","deva","tip",

"erdoğan","recep tayyip erdoğan","rte",
"kılıçdaroğlu","kemal kılıçdaroğlu",
"özgür özel","devlet bahçeli","ekrem imamoğlu",
"mansur yavaş","selahattin demirtaş",

"bakan","bakanlık","kabine","bakanlar kurulu",
"içişleri bakanı","dışişleri bakanı","adalet bakanı",

"bütçe","kanun","yasa","yasa teklifi","kanun teklifi",
"anayasa","anayasa değişikliği","anayasa mahkemesi",
"yargı","hukuk devleti",

"cumhur ittifakı","millet ittifakı",
"ittifak siyaseti","koalisyon hükümeti",

"siyaset","politika","siyasi gündem","siyasi kriz",
"kampanya","seçim kampanyası","miting","propaganda",

"dış politika","diplomasi","uluslararası ilişkiler",
"nato","ab","avrupa birliği","birleşmiş milletler",

"yerel yönetim","belediye","belediye başkanı",
"valilik","kayyum","şehir yönetimi"
]

SCIENCE_KEYWORDS = [
"araştırma","bilim","bilimsel","akademi","akademik",
"makale","yayın","paper","journal","dergi",
"hakemli dergi","peer review",

"üniversite","fakülte","laboratuvar","lab",
"araştırma merkezi","enstitü",

"arxiv","doi","nature","science","springer","elsevier",
"scopus","wos","web of science","preprint",

"deney","deneysel","hipotez","teori","model",
"metodoloji","yöntem","yöntem geliştirme",

"veri","veri analizi","veri seti","dataset",
"istatistik","istatistiksel analiz","regresyon",
"olasılık","p value","confidence interval",

"makine öğrenmesi","machine learning","deep learning",
"yapay zeka","ai","neural network","cnn","transformer",

"algoritma","hesaplama","simülasyon","modelleme",

"sonuç","bulgu","analiz","çıktı","değerlendirme",

"tez","yüksek lisans","doktora","phd","master",
"lisans üstü","dissertation",

"konferans","sempozyum","workshop","poster",
"sunum","presentation",

"literatür","literatür taraması","review paper",
"systematic review","meta analysis"
]

ALLIANCE_KEYWORDS = [
"akp","ak parti","mhp","cumhur ittifakı",
"erdoğan","recep tayyip erdoğan","rte",
"devlet bahçeli",

"iktidar","iktidar partisi","hükümet",
"hükümet politikası","hükümet başarısı",

"güçlü türkiye","yerli ve milli","milli irade",
"istikrar","istikrarlı yönetim",

"cumhur ittifakı politikası",
"cumhur ittifakı seçmeni",

"reis","liderimiz","başkanımız",

"yerli savunma sanayi","yerli teknoloji",
"milli teknoloji hamlesi",

"savunma sanayi","siha","iha","bayraktar",

"togg","yerli otomobil",

"kalkınma","ekonomik büyüme",
"mega projeler","altyapı yatırımı",

"milli güvenlik","terörle mücadele",

"2023 hedefleri","2053 vizyonu","2071 vizyonu"
]

OPPOSITION_KEYWORDS = [
"chp","iyiparti","iyi parti","hdp","dem","dem parti",
"millet ittifakı","muhalefet",

"kılıçdaroğlu","kemal kılıçdaroğlu",
"ekrem imamoğlu","mansur yavaş","özgür özel",

"demokrasi","özgürlük","hukuk devleti",
"adalet","eşitlik",

"iktidar eleştirisi","hükümet eleştirisi",

"erken seçim","seçim çağrısı",

"ekonomik kriz","enflasyon","hayat pahalılığı",

"yolsuzluk","şeffaflık","hesap verebilirlik",

"basın özgürlüğü","ifade özgürlüğü",
"insan hakları",

"demokratik reform","parlamenter sistem",

"güçlendirilmiş parlamenter sistem",

"yerel yönetim başarısı",

"belediye projeleri","sosyal yardım projeleri",

"seçim güvenliği","sandık güvenliği"
]
