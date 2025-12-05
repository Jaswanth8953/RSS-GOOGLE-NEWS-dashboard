import os
import json
import sqlite3
import time
import datetime as dt
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import feedparser
import requests
import torch
from collections import Counter
import re
import plotly.express as px
import plotly.graph_objects as go

from gnews import GNews
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# ========================== CONFIG ==========================

DB_PATH = "news_articles.db"

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "https://www.marketwatch.com/marketwatch/rss/topstories",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/energyNews",
]

OPENAI_EMBED_MODEL = "text-embedding-3-small"
FINBERT_MODEL = "ProsusAI/finbert"

MIN_SIMILARITY = 0.30          # similarity threshold
MAX_ARTICLES_DEFAULT = 200
GDELT_MAX_RECORDS = 250
GNEWS_MAX_RESULTS = 200

DetectorFactory.seed = 0

# ========================== DB SETUP (LOCK-FREE) ==========================

def get_connection():
    """
    Single shared SQLite connection.
    WAL mode + timeout to avoid 'database is locked'.
    Also adds extra columns for language and precomputed sentiment.
    """
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
        timeout=30  # wait up to 30s if locked
    )

    # WAL mode allows concurrent reads/writes
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size = -20000;")  # use memory cache

    # Base table (will not modify existing)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            summary TEXT,
            published TEXT,
            link TEXT UNIQUE,
            source TEXT,
            content TEXT,
            embedding TEXT,
            language TEXT DEFAULT 'unknown',
            precomputed_sentiment_label TEXT,
            precomputed_sentiment_score REAL,
            embedding_generated INTEGER DEFAULT 0,
            sentiment_computed INTEGER DEFAULT 0
        );
        """
    )

    # Backward compatibility: add missing columns if old DB existed
    for col_def in [
        "ALTER TABLE articles ADD COLUMN language TEXT DEFAULT 'unknown'",
        "ALTER TABLE articles ADD COLUMN precomputed_sentiment_label TEXT",
        "ALTER TABLE articles ADD COLUMN precomputed_sentiment_score REAL",
        "ALTER TABLE articles ADD COLUMN embedding_generated INTEGER DEFAULT 0",
        "ALTER TABLE articles ADD COLUMN sentiment_computed INTEGER DEFAULT 0",
    ]:
        try:
            conn.execute(col_def)
        except Exception:
            pass

    conn.execute("CREATE INDEX IF NOT EXISTS idx_published ON articles(published);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_language ON articles(language);")
    conn.commit()
    return conn


conn = get_connection()


def safe_execute(query: str, params: tuple = ()):
    """
    Safer wrapper for INSERT/UPDATE with retry if DB is locked.
    """
    max_retries = 10
    for _ in range(max_retries):
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            conn.commit()
            return cur
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg:
                time.sleep(0.5)
                continue
            raise
    raise Exception("Database write failed after multiple retries")


# ========================== OPENAI CLIENT ==========================

@st.cache_resource
def get_openai_client():
    """
    Uses either Streamlit secrets OR environment variable.
    Works on Streamlit Cloud and AWS.
    """
    api_key = None
    # Streamlit Cloud: secrets.toml with [openai] api_key="..."
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        api_key = st.secrets["openai"]["api_key"]

    # Fallback: standard env var (AWS, local)
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "❌ OpenAI API key not found.\n\n"
            "Set it in `.streamlit/secrets.toml` under [openai].api_key\n"
            "or as environment variable OPENAI_API_KEY."
        )
        st.stop()

    return OpenAI(api_key=api_key)


# ========================== LANGUAGE DETECTION ==========================

def detect_language(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"
    except Exception:
        return "unknown"


def ensure_language_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows with unknown/empty language, detect language using title+summary+content,
    update DB, and return updated df.
    """
    if "language" not in df.columns:
        df["language"] = "unknown"

    mask = df["language"].isna() | df["language"].eq("") | df["language"].eq("unknown")
    if not mask.any():
        return df

    subset = df[mask].copy()
    for idx, row in subset.iterrows():
        text = f"{row.get('title','')} {row.get('summary','')} {row.get('content','')}"
        lang = detect_language(text)
        df.at[idx, "language"] = lang
        try:
            safe_execute("UPDATE articles SET language = ? WHERE id = ?", (lang, int(row["id"])))
        except Exception:
            pass

    return df


# ========================== FINBERT SENTIMENT ==========================

@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def finbert_sentiment(texts: List[str]) -> List[dict]:
    """
    Run FinBERT and apply calibration to avoid unrealistic 100% predictions.
    - If max probability < 0.60 → treat as neutral, score = 0.50
    - Cap max confidence at 0.95
    - If positive but negative prob also high → neutral
    """
    if not texts:
        return []
    tokenizer, model, device = load_finbert()
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    results = []
    for p in probs:
        idx = int(np.argmax(p))
        max_score = float(p[idx])
        base_label = id2label[idx]

        if max_score < 0.60:
            label = "neutral"
            score = 0.50
        else:
            label = base_label
            score = min(max_score, 0.95)

        # If positive but negative also high → neutral
        if label == "positive" and p[0] > 0.25:
            label = "neutral"
            score = 0.50

        results.append({"label": label, "score": score})
    return results


# ========================== EMBEDDINGS ==========================

def get_embedding(text: str) -> List[float]:
    client = get_openai_client()
    # short, cleaned text for cheaper, faster embeddings
    text = (text or "").replace("\n", " ")
    if len(text) > 500:
        text = text[:500]
    emb = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text
    )
    return emb.data[0].embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ========================== RSS INGESTION ==========================

def parse_rss(url: str) -> List[dict]:
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        link = entry.get("link", "")

        if hasattr(entry, "published_parsed") and entry.published_parsed:
            pub = dt.datetime(*entry.published_parsed[:6])
        else:
            pub = dt.datetime.utcnow()
        published = pub.isoformat()

        if "content" in entry and entry.content:
            content = entry.content[0].value
        else:
            content = summary

        articles.append(
            {
                "title": title,
                "summary": summary,
                "published": published,
                "link": link,
                "source": url,
                "content": content,
            }
        )
    return articles


def fetch_rss_articles() -> int:
    new_count = 0
    for url in RSS_FEEDS:
        try:
            arts = parse_rss(url)
        except Exception:
            continue
        for art in arts:
            try:
                safe_execute(
                    """
                    INSERT OR IGNORE INTO articles
                    (title, summary, published, link, source, content, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        art["title"],
                        art["summary"],
                        art["published"],
                        art["link"],
                        art["source"],
                        art["content"],
                    ),
                )
                new_count += 1
            except Exception:
                pass
    return new_count


# ========================== GDELT INGESTION ==========================

def fetch_gdelt_articles(query: str, start: dt.date, end: dt.date) -> int:
    """
    Pulls articles from GDELT 2.0 DOC API for a given query and date range.
    """
    new_count = 0

    start_str = start.strftime("%Y%m%d000000")
    end_str = end.strftime("%Y%m%d235959")

    base_url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={requests.utils.quote(query)}"
        f"&mode=artlist&maxrecords={GDELT_MAX_RECORDS}&format=json"
        f"&startdatetime={start_str}&enddatetime={end_str}"
    )

    try:
        r = requests.get(base_url, timeout=20)
        if r.status_code != 200:
            return 0
        data = r.json()
        articles = data.get("articles", [])
    except Exception:
        return 0

    for a in articles:
        title = a.get("title", "")
        summary = a.get("seendate", "")
        link = a.get("url", "")
        src = a.get("domain", "gdelt")
        seen = a.get("seendate", "")
        try:
            published = dt.datetime.strptime(seen, "%Y%m%d%H%M%S").isoformat()
        except Exception:
            published = dt.datetime.utcnow().isoformat()
        content = a.get("snippet", "") or summary

        try:
            safe_execute(
                """
                INSERT OR IGNORE INTO articles
                (title, summary, published, link, source, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (title, summary, published, link, src, content),
            )
            new_count += 1
        except Exception:
            pass

    return new_count


# ========================== GOOGLE NEWS (GNews) ==========================

def fetch_gnews_articles(query: str, start: dt.date, end: dt.date) -> int:
    new_count = 0

    try:
        gn = GNews(language="en", max_results=GNEWS_MAX_RESULTS)
        gn.start_date = (start.year, start.month, start.day)
        gn.end_date = (end.year, end.month, end.day)
        results = gn.get_news(query)
    except Exception:
        return 0

    for r in results:
        title = r.get("title", "")
        link = r.get("url") or r.get("link", "")
        pub = r.get("published date") or r.get("published_date")
        desc = r.get("description", "")
        source = r.get("publisher", {}).get("title", "google-news")

        try:
            published = str(pub) if pub else dt.datetime.utcnow().isoformat()
        except Exception:
            published = dt.datetime.utcnow().isoformat()

        content = desc

        try:
            safe_execute(
                """
                INSERT OR IGNORE INTO articles
                (title, summary, published, link, source, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (title, desc, published, link, source, content),
            )
            new_count += 1
        except Exception:
            pass

    return new_count


# ========================== EMBEDDING MANAGEMENT ==========================

def load_articles_for_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary, published, link, source, content, embedding, language,
               precomputed_sentiment_label, precomputed_sentiment_score
        FROM articles
        WHERE date(published) BETWEEN ? AND ?
        """,
        (start.isoformat(), end.isoformat()),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "id",
            "title",
            "summary",
            "published",
            "link",
            "source",
            "content",
            "embedding",
            "language",
            "precomputed_sentiment_label",
            "precomputed_sentiment_score",
        ],
    )
    return df


def ensure_embeddings(df_ids_title_summary: pd.DataFrame):
    """
    Add embeddings for rows that don't have them yet.
    Uses safe_execute to avoid DB locking.
    """
    for _, row in df_ids_title_summary.iterrows():
        emb_str = row.get("embedding")
        if emb_str in (None, "", "NULL", "null"):
            text = (row["title"] or "") + " " + (row["summary"] or "")
            if not text.strip():
                continue
            emb = get_embedding(text)
            try:
                safe_execute(
                    "UPDATE articles SET embedding = ?, embedding_generated = 1 WHERE id = ?",
                    (json.dumps(emb), int(row["id"])),
                )
            except Exception:
                pass


def hybrid_search(
    query: str,
    start: dt.date,
    end: dt.date,
    top_k: int,
    min_sim: float = MIN_SIMILARITY,
) -> pd.DataFrame:
    """
    Hybrid keyword + semantic search within date range.
    Returns a DataFrame with 'similarity' and 'match_type'.
    """
    df = load_articles_for_range(start, end)
    if df.empty:
        return df

    df["embedding"] = df["embedding"].apply(
        lambda x: None if x in (None, "", "NULL", "null") else x
    )

    # only compute embeddings for rows that are missing
    ensure_embeddings(df[["id", "title", "summary", "embedding"]])

    df = load_articles_for_range(start, end)
    if df.empty:
        return df

    # semantic similarity
    q_emb = np.array(get_embedding(query))
    sims = []
    for _, row in df.iterrows():
        try:
            emb_vec = np.array(json.loads(row["embedding"]))
            sims.append(cosine_similarity(q_emb, emb_vec))
        except Exception:
            sims.append(0.0)
    df["similarity"] = sims

    # keyword mask
    q_lower = query.lower().strip()
    kw_mask = (
        df["title"].str.lower().str.contains(q_lower, na=False)
        | df["summary"].str.lower().str.contains(q_lower, na=False)
        | df["content"].str.lower().str.contains(q_lower, na=False)
    )

    sem_mask = df["similarity"] >= min_sim

    if kw_mask.any() and sem_mask.any():
        df["match_type"] = np.where(
            kw_mask & sem_mask,
            "keyword+semantic",
            np.where(kw_mask, "keyword", "semantic"),
        )
        filtered = df[kw_mask | sem_mask]
    elif kw_mask.any():
        df["match_type"] = "keyword"
        filtered = df[kw_mask]
    elif sem_mask.any():
        df["match_type"] = "semantic"
        filtered = df[sem_mask]
    else:
        return pd.DataFrame()

    filtered = filtered.sort_values("similarity", ascending=False).head(top_k)
    return filtered


# ========================== LLM QUERY EXPANSION ==========================

def expand_query_with_llm(query: str) -> List[str]:
    """
    Use LLM to generate related terms for the user's query.
    This is what helps fix US Tech vs Nvidia vs Technology differences.
    """
    client = get_openai_client()

    prompt = f"""
You are helping expand a financial news search query.

Original query: "{query}"

Generate 8-12 alternative search terms that:
- mean the same
- are closely related industries
- include broader and narrower concepts
- include common synonyms

Return ONLY a valid JSON list of strings, like:
["term1", "term2", "term3"]
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate semantic search expansions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        content = resp.choices[0].message.content.strip()
        terms = json.loads(content)
        terms = [str(t) for t in terms if isinstance(t, str)]

        # always include original query as first term
        all_terms = [query] + terms

        # deduplicate while preserving order
        seen = set()
        cleaned = []
        for t in all_terms:
            t_clean = t.strip()
            if not t_clean:
                continue
            low = t_clean.lower()
            if low in seen:
                continue
            seen.add(low)
            cleaned.append(t_clean)

        return cleaned[:12]
    except Exception:
        # fallback: just the original term
        return [query]


# ========================== LLM ARTICLE SELECTION ==========================

def llm_select_articles(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Let the LLM decide which of the already-matched articles
    are truly relevant to the user's topic.
    We send only (id, title, summary) to the LLM to keep it light.
    """
    if df.empty:
        return df

    client = get_openai_client()

    # cap to avoid huge prompts (cost + speed)
    df_small = df.head(200).copy()

    # Prepare lightweight list
    items = [
        {
            "id": int(row["id"]),
            "title": row["title"],
            "summary": (row["summary"] or "")[:250],
        }
        for _, row in df_small.iterrows()
    ]

    prompt = f"""
You are an article relevance filter for a financial news dashboard.

The user topic is: "{query}"

Below is a list of articles with IDs, titles, and summaries.

Decide which articles are truly relevant to this topic.
Return ONLY a JSON list of the IDs to KEEP.
Do NOT include any explanation text, just the JSON list.

Articles:
{json.dumps(items, indent=2)}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You select relevant financial news articles."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content.strip()
        selected_ids = json.loads(content)
        if not isinstance(selected_ids, list):
            return df_small
        selected_ids = set(int(i) for i in selected_ids)
        return df_small[df_small["id"].isin(selected_ids)]
    except Exception:
        # if anything goes wrong, just return the truncated df
        return df_small


# ========================== KEYWORDS & SENTIMENT INDEX ==========================

def extract_top_keywords(titles: List[str], n: int = 20) -> List[Tuple[str, int]]:
    text = " ".join(titles).lower()
    words = re.findall(r"[a-zA-Z]{4,}", text)
    stop = {
        "this",
        "that",
        "with",
        "from",
        "will",
        "have",
        "been",
        "into",
        "after",
        "over",
        "under",
        "they",
        "them",
        "your",
        "their",
        "about",
        "which",
        "there",
        "where",
        "when",
        "than",
        "because",
        "while",
        "before",
        "through",
        "within",
        "without",
    }
    words = [w for w in words if w not in stop]
    counter = Counter(words)
    return counter.most_common(n)


def calculate_sentiment_index(df: pd.DataFrame) -> float:
    """
    Weighted sentiment index:
    - Only English articles with positive/neutral/negative labels are used
    - Weight by similarity, recency, and sentiment confidence
    - Output scaled to [-95, 95] to avoid misleading 100% extremes
    """
    if df.empty or "sentiment_label" not in df.columns:
        return 0.0

    # Only English with scored labels
    df_en = df[(df["language"] == "en") & df["sentiment_label"].isin(["positive", "neutral", "negative"])].copy()
    if df_en.empty:
        return 0.0

    today = dt.date.today()
    df_en["published_dt"] = pd.to_datetime(df_en["published"], errors="coerce")
    df_en["age_days"] = (today - df_en["published_dt"].dt.date).dt.days
    df_en["age_days"] = df_en["age_days"].clip(lower=0).fillna(365)

    # recency weight: exponential decay ~2 months
    df_en["recency_weight"] = np.exp(-df_en["age_days"] / 60.0)

    # similarity + confidence
    df_en["sim_weight"] = df_en["similarity"].clip(lower=0.1)
    df_en["conf_weight"] = df_en["sentiment_score"].fillna(0.5).clip(lower=0.3)

    df_en["weight"] = df_en["recency_weight"] * df_en["sim_weight"] * df_en["conf_weight"]

    sent_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    df_en["sent_value"] = df_en["sentiment_label"].map(sent_map).fillna(0.0)

    num = (df_en["sent_value"] * df_en["weight"]).sum()
    den = df_en["weight"].sum()
    if den == 0:
        return 0.0

    index = (num / den) * 100.0
    index = float(np.clip(index, -95.0, 95.0))
    return round(index, 2)


# ========================== STREAMLIT APP ==========================

def run_app():
    st.set_page_config(
        page_title="Financial News Sentiment Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Title
    st.title("📈 Financial News Sentiment Dashboard")
    st.markdown(
        "RSS + Google News + GDELT → **Hybrid + LLM Query Expansion + LLM Article Selection** → FinBERT Sentiment"
    )

    # Sidebar toggle (optional)
    if "show_sidebar" not in st.session_state:
        st.session_state["show_sidebar"] = True

    top_col1, top_col2 = st.columns([4, 1])
    with top_col2:
        show_sidebar = st.checkbox(
            "Show sidebar", value=st.session_state["show_sidebar"]
        )
        st.session_state["show_sidebar"] = show_sidebar

    if not st.session_state["show_sidebar"]:
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] {display: none;}
            [data-testid="stAppViewContainer"] {margin-left: 0;}
            </style>
            """,
            unsafe_allow_html=True,
        )

    # SIDEBAR
    with st.sidebar:
        st.header("Settings")

        if st.button("Fetch base RSS articles"):
            with st.spinner("Fetching RSS feeds…"):
                n = fetch_rss_articles()
            st.success(f"RSS update done. New articles added: {n}")

        today = dt.date.today()
        default_start = today - dt.timedelta(days=30)
        start_date = st.date_input("Start date", default_start)
        end_date = st.date_input("End date", today)

        st.subheader("Extra sources for this topic")
        use_gdelt = st.checkbox("Include GDELT", True)
        use_gnews = st.checkbox("Include Google News", True)

        max_articles = st.slider(
            "Max articles AFTER search",
            min_value=50,
            max_value=500,
            value=MAX_ARTICLES_DEFAULT,
            step=50,
        )

    # MAIN QUERY
    st.markdown("### 🔍 Search Topic")
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        query = st.text_input(
            "Enter topic (e.g. 'US Tech', 'Germany', 'European textiles', 'oil', 'Nvidia')",
            key="main_query",
        )

    analyze_clicked = st.button("Analyze", type="primary")

    if not analyze_clicked:
        st.info("Enter a topic and click **Analyze** to run the pipeline.")
        return

    if not query.strip():
        st.warning("Please enter a non-empty topic.")
        return

    # Query expansion via LLM
    with st.spinner("Using LLM to expand your topic into related concepts…"):
        expanded_queries = expand_query_with_llm(query)

    st.write("**Expanded queries used:**")
    st.write(", ".join(expanded_queries))

    # Fetch extra data for this query from GDELT + Google News
    if use_gdelt or use_gnews:
        with st.spinner("Fetching additional articles from GDELT / Google News…"):
            total_added = 0
            # to keep cost stable, only use first few expansions for external APIs
            source_terms = expanded_queries[:5]
            for q in source_terms:
                if use_gdelt:
                    total_added += fetch_gdelt_articles(q, start_date, end_date)
                if use_gnews:
                    total_added += fetch_gnews_articles(q, start_date, end_date)
        st.success(f"Added ~{total_added} extra articles for this topic.")

    # Hybrid search for each expanded query
    with st.spinner("Running hybrid keyword + semantic search…"):
        dfs = []
        for q in expanded_queries:
            part = hybrid_search(
                query=q,
                start=start_date,
                end=end_date,
                top_k=max_articles,
                min_sim=MIN_SIMILARITY,
            )
            if not part.empty:
                part["query_term"] = q
                dfs.append(part)

        if not dfs:
            st.error(
                "No matches found for this topic in the selected date range. "
                "Try broadening the date or changing the query."
            )
            return

        df = pd.concat(dfs, ignore_index=True)
        # drop duplicates by link
        df = df.drop_duplicates(subset=["link"])

    st.write(f"🔎 Initial hybrid search found **{len(df)}** candidate articles.")

    # Optional LLM article filter / selector
    with st.spinner("Letting LLM choose the most relevant subset of articles…"):
        df = llm_select_articles(query, df)

    if df.empty:
        st.error("After LLM filtering, no strongly relevant articles remained.")
        return

    st.success(f"✅ LLM selected **{len(df)}** final articles for analysis.")

    # Ensure language for these articles
    df = ensure_language_for_df(df)

    # SENTIMENT (English only)
    with st.spinner("Running FinBERT sentiment classification on English articles…"):
        # initialize
        df["sentiment_label"] = "not_scored"
        df["sentiment_score"] = np.nan

        mask_en = df["language"] == "en"
        df_en = df[mask_en].copy()

        if not df_en.empty:
            texts = df_en["content"].fillna(df_en["summary"]).tolist()
            texts = [(t or "")[:1000] for t in texts]
            sents = finbert_sentiment(texts)
            df.loc[mask_en, "sentiment_label"] = [s["label"] for s in sents]
            df.loc[mask_en, "sentiment_score"] = [s["score"] for s in sents]

    df["relevance"] = (df["similarity"] * 100).round(1)
    df["source_domain"] = df["source"].fillna("").replace("", "unknown")

    sentiment_index = calculate_sentiment_index(df)

    # TABS
    tab_dash, tab_articles, tab_keywords, tab_download = st.tabs(
        ["📊 Dashboard", "📰 Articles", "🔑 Keywords", "📥 Download"]
    )

    # DASHBOARD TAB
    with tab_dash:
        st.subheader("Executive Sentiment Overview")

        counts = df["sentiment_label"].value_counts()
        total = int(counts.sum())
        pos = int(counts.get("positive", 0))
        neg = int(counts.get("negative", 0))
        neu = int(counts.get("neutral", 0))

        c_a, c_b, c_c, c_d, c_e = st.columns(5)
        c_a.metric("Total Articles", total)
        c_b.metric("Positive", pos, f"{(pos/total*100):.1f}%" if total else "0.0%")
        c_c.metric("Negative", neg, f"{(neg/total*100):.1f}%" if total else "0.0%")
        c_d.metric("Neutral", neu, f"{(neu/total*100):.1f}%" if total else "0.0%")

        direction = (
            "Bullish" if sentiment_index > 0 else "Bearish" if sentiment_index < 0 else "Neutral"
        )
        c_e.metric("Sentiment Index", f"{sentiment_index:.1f}", direction)

        st.markdown("---")

        # Pie chart
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown("#### Sentiment Distribution (English only + scored)")
            sentiment_count = df[df["sentiment_label"].isin(["positive", "negative", "neutral"])]["sentiment_label"].value_counts()
            if sentiment_count.empty:
                st.info("No English articles with sentiment scored.")
            else:
                fig_pie = px.pie(
                    names=sentiment_count.index,
                    values=sentiment_count.values,
                    color=sentiment_count.index,
                    color_discrete_map={
                        "positive": "#2ecc71",
                        "negative": "#e74c3c",
                        "neutral": "#95a5a6",
                    },
                    hole=0.3,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)

        # Gauge chart
        with r1c2:
            st.markdown("#### Sentiment Index Gauge")
            gauge_color = "#2ecc71" if sentiment_index > 0 else "#e74c3c"
            fig_g = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=sentiment_index,
                    title={"text": "Sentiment Index"},
                    delta={"reference": 0},
                    gauge={
                        "axis": {"range": [-100, 100]},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [-100, 0], "color": "rgba(231, 76, 60, 0.2)"},
                            {"range": [0, 100], "color": "rgba(46, 204, 113, 0.2)"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 0,
                        },
                    },
                )
            )
            fig_g.update_layout(height=400)
            st.plotly_chart(fig_g, use_container_width=True)

        # Source + confidence
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("#### Articles by Source")
            src_counts = df["source_domain"].value_counts()
            fig_src = px.bar(
                x=src_counts.values,
                y=src_counts.index,
                orientation="h",
                labels={"x": "Count", "y": "Source"},
                color=src_counts.values,
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_src, use_container_width=True)

        with r2c2:
            st.markdown("#### Average Confidence by Sentiment (English only)")
            df_conf = df[df["sentiment_label"].isin(["positive", "negative", "neutral"])]
            if df_conf.empty:
                st.info("No English articles with sentiment scored.")
            else:
                avg_conf = (df_conf.groupby("sentiment_label")["sentiment_score"].mean() * 100).round(1)
                fig_conf = px.bar(
                    x=avg_conf.index,
                    y=avg_conf.values,
                    labels={"x": "Sentiment", "y": "FinBERT confidence (%)"},
                    color=avg_conf.index,
                    color_discrete_map={
                        "positive": "#2ecc71",
                        "negative": "#e74c3c",
                        "neutral": "#95a5a6",
                    },
                )
                fig_conf.update_layout(showlegend=False)
                st.plotly_chart(fig_conf, use_container_width=True)

    # ARTICLES TAB
    with tab_articles:
        st.subheader("Articles")
        sent_filter = st.multiselect(
            "Filter by sentiment",
            options=["positive", "negative", "neutral", "not_scored"],
            default=["positive", "negative", "neutral", "not_scored"],
        )
        filtered = df[df["sentiment_label"].isin(sent_filter)].copy()
        filtered = filtered.sort_values(
            ["relevance", "sentiment_score"], ascending=[False, False]
        )
        for _, row in filtered.iterrows():
            icon_map = {"positive": "🟢", "negative": "🔴", "neutral": "🔵", "not_scored": "⚪"}
            icon = icon_map.get(row["sentiment_label"], "⚪")
            st.markdown(f"**{icon} {row['title']}**")
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            c1.caption(f"{row['source_domain']} | lang: {row.get('language','unknown')}")
            c2.caption(f"Relevance: {row['relevance']:.1f}%")
            if row["sentiment_label"] in ["positive", "negative", "neutral"]:
                c3.caption(f"Sentiment conf: {row['sentiment_score']*100:.1f}%")
            else:
                c3.caption("Sentiment: not scored")
            c4.markdown(f"[Read →]({row['link']})")
            if row["summary"]:
                st.caption(row["summary"][:280] + "…")
            st.markdown("---")

    # KEYWORDS TAB
    with tab_keywords:
        st.subheader("Trending Keywords")
        keywords = extract_top_keywords(df["title"].tolist(), n=20)
        if keywords:
            kw_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])
            col_a, col_b = st.columns(2)
            with col_a:
                fig_kw = px.bar(
                    kw_df,
                    x="Frequency",
                    y="Keyword",
                    orientation="h",
                    color="Frequency",
                    color_continuous_scale="Blues",
                )
                fig_kw.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(fig_kw, use_container_width=True)
            with col_b:
                fig_tree = px.treemap(
                    kw_df,
                    path=["Keyword"],
                    values="Frequency",
                    color="Frequency",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info("Not enough data to extract keywords.")

    # DOWNLOAD TAB
    with tab_download:
        st.subheader("Download Results")
        dl_df = df[
            [
                "title",
                "summary",
                "published",
                "link",
                "source_domain",
                "language",
                "sentiment_label",
                "sentiment_score",
                "relevance",
                "match_type",
                "query_term",
            ]
        ]
        csv = dl_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name="news_sentiment_results.csv",
            mime="text/csv",
        )
        st.write("### Data Preview")
        st.dataframe(dl_df, use_container_width=True)


if __name__ == "__main__":
    run_app()
