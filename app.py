import os
import json
import sqlite3
import time
import datetime as dt
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# ========================== CONFIG ==========================
# ============================================================

DB_PATH = "news_articles.db"

OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
FINBERT_MODEL = "ProsusAI/finbert"

# Hybrid search + index parameters
MIN_SIMILARITY = 0.30           # similarity threshold (Professor's suggestion)
MAX_ARTICLES_CAP = 500          # hard cap after threshold (for performance)
EMBED_PRECOMP_BATCH = 30        # per precompute call
SENT_PRECOMP_BATCH = 50         # per precompute call
LANG_PRECOMP_BATCH = 100        # per precompute call

DetectorFactory.seed = 0

# ============================================================
# ========================== DATABASE ========================
# ============================================================

def get_connection():
    """Create SQLite connection with proper schema & indexes."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

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

    # Backward compatibility: try adding columns if missing
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN precomputed_sentiment_label TEXT")
    except Exception:
        pass

    try:
        conn.execute("ALTER TABLE articles ADD COLUMN precomputed_sentiment_score REAL")
    except Exception:
        pass

    try:
        conn.execute("ALTER TABLE articles ADD COLUMN embedding_generated INTEGER DEFAULT 0")
    except Exception:
        pass

    try:
        conn.execute("ALTER TABLE articles ADD COLUMN sentiment_computed INTEGER DEFAULT 0")
    except Exception:
        pass

    try:
        conn.execute("ALTER TABLE articles ADD COLUMN language TEXT DEFAULT 'unknown'")
    except Exception:
        pass

    conn.execute("CREATE INDEX IF NOT EXISTS idx_published ON articles(published);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_language ON articles(language);")
    conn.commit()
    return conn


# Global connection (Streamlit will reuse)
conn = get_connection()


def safe_execute(query: str, params: tuple = ()):
    """Safe DB write with retries (avoids 'database is locked')."""
    max_retries = 5
    for _ in range(max_retries):
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            conn.commit()
            return cur
        except sqlite3.OperationalError:
            time.sleep(0.1)
            continue
    raise Exception("Database write failed")


# ============================================================
# ========================== OPENAI CLIENT ===================
# ============================================================

@st.cache_resource
def get_openai_client():
    """Return cached OpenAI client, or stop app with nice error."""
    api_key = None
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        api_key = st.secrets["openai"]["api_key"]

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("❌ OpenAI API key not found. Please set it in Streamlit secrets or environment.")
        st.stop()

    return OpenAI(api_key=api_key)


# ============================================================
# ====================== LANGUAGE DETECTION ==================
# ============================================================

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


def precompute_language_for_unknown():
    """
    Detect language for articles with language='unknown'.
    We only need a rough guess; used to avoid FinBERT on non-English.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary, content
        FROM articles
        WHERE language = 'unknown'
        LIMIT ?
        """,
        (LANG_PRECOMP_BATCH,),
    )
    rows = cur.fetchall()
    if not rows:
        return 0

    updated = 0
    for article_id, title, summary, content in rows:
        txt = f"{title or ''} {summary or ''} {content or ''}".strip()
        if not txt:
            lang = "unknown"
        else:
            lang = detect_language(txt)
        try:
            safe_execute(
                "UPDATE articles SET language = ? WHERE id = ?",
                (lang, article_id),
            )
            updated += 1
        except Exception:
            continue

    return updated


# ============================================================
# ===================== FINBERT SENTIMENT ====================
# ============================================================

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
    FinBERT with aggressive calibration so we NEVER get crazy 100% outputs.
    - Threshold: if max_prob < 0.65 → force neutral
    - Scale confidences down
    - Never return score > 0.95
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

        # Base calibration
        if max_score < 0.65:
            label = "neutral"
            score = 0.5
        else:
            label = base_label
            if max_score > 0.90:
                score = max_score * 0.7
            elif max_score > 0.80:
                score = max_score * 0.8
            else:
                score = max_score

        # Check if positive but negative also high → neutral
        if label == "positive":
            neg_prob = float(p[0])
            if neg_prob > 0.25:
                label = "neutral"
                score = 0.5

        # Final safety: never 1.0
        score = min(score, 0.95)

        results.append({"label": label, "score": score})

    return results


def precompute_sentiment_for_new_articles():
    """
    Precompute sentiment for English articles where sentiment_computed = 0.
    Professor’s suggestion: compute during ingestion / background instead
    of on every query.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary, content
        FROM articles
        WHERE sentiment_computed = 0
          AND language = 'en'
        LIMIT ?
        """,
        (SENT_PRECOMP_BATCH,),
    )
    rows = cur.fetchall()
    if not rows:
        return 0

    articles = []
    for row in rows:
        article_id, title, summary, content = row
        text = f"{title or ''} {summary or ''} {content or ''}".strip()
        text = text[:1000]  # avoid very long texts
        if not text:
            text = (title or "")[:250]
        articles.append({"id": article_id, "text": text})

    texts = [a["text"] for a in articles]
    sentiments = finbert_sentiment(texts)

    updated = 0
    for art, sent in zip(articles, sentiments):
        try:
            safe_execute(
                """
                UPDATE articles
                SET precomputed_sentiment_label = ?,
                    precomputed_sentiment_score = ?,
                    sentiment_computed = 1
                WHERE id = ?
                """,
                (sent["label"], sent["score"], art["id"]),
            )
            updated += 1
        except Exception:
            continue

    return updated


# ============================================================
# ======================= EMBEDDINGS =========================
# ============================================================

def get_embedding(text: str) -> List[float]:
    client = get_openai_client()
    text = (text or "").replace("\n", " ")
    if len(text) > 500:
        text = text[:500]
    emb = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text,
    )
    return emb.data[0].embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def precompute_embeddings_for_new_articles():
    """Generate embeddings for articles where embedding_generated = 0."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary
        FROM articles
        WHERE embedding_generated = 0
        LIMIT ?
        """,
        (EMBED_PRECOMP_BATCH,),
    )
    rows = cur.fetchall()
    if not rows:
        return 0

    updated = 0
    for article_id, title, summary in rows:
        text = f"{title or ''} {summary or ''}".strip()
        text = text[:500]
        if not text:
            continue
        try:
            emb = get_embedding(text)
            safe_execute(
                """
                UPDATE articles
                SET embedding = ?, embedding_generated = 1
                WHERE id = ?
                """,
                (json.dumps(emb), article_id),
            )
            updated += 1
        except Exception:
            continue

    return updated


# ============================================================
# =================== LOAD ARTICLES / SEARCH =================
# ============================================================

def load_articles_for_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Load all articles for time range, English + others."""
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(articles)")
    columns = [row[1] for row in cur.fetchall()]

    select_fields = [
        "id",
        "title",
        "summary",
        "published",
        "link",
        "source",
        "content",
        "embedding",
        "language",
    ]
    if "precomputed_sentiment_label" in columns:
        select_fields.append("precomputed_sentiment_label")
    if "precomputed_sentiment_score" in columns:
        select_fields.append("precomputed_sentiment_score")

    query = f"""
        SELECT {', '.join(select_fields)}
        FROM articles
        WHERE date(published) BETWEEN ? AND ?
        """

    cur.execute(query, (start.isoformat(), end.isoformat()))
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=select_fields)

    if "precomputed_sentiment_label" not in df.columns:
        df["precomputed_sentiment_label"] = None
    if "precomputed_sentiment_score" not in df.columns:
        df["precomputed_sentiment_score"] = None

    return df


def expand_query_with_llm(query: str) -> List[str]:
    """
    LLM-based query expansion as Eric requested.
    Example:
      "Germany" → ["Germany", "German", "German industry", "German economy", ...]
    """
    client = get_openai_client()

    prompt = f"""
You are helping with financial news search. For the query "{query}",
generate 5–8 related search terms that capture:
1. Same meaning in different words
2. Related industries/sectors/regions
3. Broader and narrower financial concepts
4. Common synonyms used in markets and news

Return ONLY a JSON list of strings. No explanation.
"""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Generate related financial search terms as JSON list of strings."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        content = resp.choices[0].message.content.strip()
        # Try to extract JSON list
        import re as _re
        m = _re.search(r"\[.*\]", content, flags=_re.DOTALL)
        if m:
            expansions = json.loads(m.group())
        else:
            expansions = json.loads(content)

        expansions = [str(x) for x in expansions if isinstance(x, str)]
        all_terms = [query] + expansions

        # Deduplicate while preserving order
        seen = set()
        cleaned = []
        for t in all_terms:
            t = t.strip()
            if not t:
                continue
            low = t.lower()
            if low in seen:
                continue
            seen.add(low)
            cleaned.append(t)

        return cleaned[:8]
    except Exception:
        # Fallback: just use original query
        return [query]


def hybrid_search(q: str, df: pd.DataFrame, min_sim: float = MIN_SIMILARITY) -> pd.DataFrame:
    """
    Hybrid semantic + keyword search:
    - semantic similarity via embeddings
    - small keyword boost on title + summary
    - uses similarity threshold (NOT fixed 200)
    """
    if df.empty:
        return df

    # Compute query embedding once
    q_emb = np.array(get_embedding(q))

    sims = []
    for _, row in df.iterrows():
        emb_str = row.get("embedding")
        if isinstance(emb_str, str) and emb_str not in ("", "null", "None"):
            try:
                emb_vec = np.array(json.loads(emb_str))
                sim = cosine_similarity(q_emb, emb_vec)
            except Exception:
                sim = 0.0
        else:
            sim = 0.0
        sims.append(sim)

    df = df.copy()
    df["similarity"] = sims

    # Keyword boost
    q_lower = q.lower()
    title_boost = df["title"].fillna("").str.lower().str.contains(q_lower).astype(int) * 0.2
    sum_boost = df["summary"].fillna("").str.lower().str.contains(q_lower).astype(int) * 0.1
    df["similarity"] = (df["similarity"] + title_boost + sum_boost).clip(0, 1)

    # Threshold-based selection (Eric: NOT "top 200")
    filtered = df[df["similarity"] >= min_sim].copy()
    if filtered.empty:
        return filtered

    filtered = filtered.sort_values("similarity", ascending=False)

    # Hard cap only for performance, AFTER threshold
    if len(filtered) > MAX_ARTICLES_CAP:
        filtered = filtered.head(MAX_ARTICLES_CAP)

    return filtered


# ============================================================
# ============ WEIGHTED SENTIMENT INDEX / FUSION =============
# ============================================================

def calculate_weighted_sentiment_index(df: pd.DataFrame) -> dict:
    """
    Weighted sentiment index:
    - Only English articles are used for sentiment index
    - Weight by: similarity, source reliability, sentiment confidence, recency
    - Index ∈ [-100, 100] but clipped to [-95, 95] to avoid 100% extremes
    """
    if df.empty:
        return {
            "index": 0.0,
            "confidence": 0.0,
            "total_articles": 0,
            "weighted_articles": 0,
            "english_articles": 0,
        }

    # Only English for actual FinBERT sentiment
    df_eng = df[df["language"] == "en"].copy()
    if df_eng.empty:
        return {
            "index": 0.0,
            "confidence": 0.0,
            "total_articles": len(df),
            "weighted_articles": 0,
            "english_articles": 0,
        }

    # Source reliability weights
    source_weights = {
        "reuters": 1.2,
        "bbc": 1.2,
        "cnbc": 1.1,
        "bloomberg": 1.1,
        "wsj": 1.1,
        "ft": 1.1,
        "marketwatch": 1.0,
        "techcrunch": 1.0,
        "theverge": 0.9,
        "wired": 0.9,
        "aljazeera": 0.9,
        "gdelt": 0.8,
        "google-news": 0.8,
    }

    # Domain extraction
    def source_domain(x: str) -> str:
        s = str(x or "")
        if "//" in s:
            return s.split("//")[-1].split("/")[0]
        return s[:50]

    df_eng["source_domain"] = df_eng["source"].apply(source_domain)

    df_eng["source_weight"] = df_eng["source_domain"].apply(
        lambda x: next((v for k, v in source_weights.items() if k in x.lower()), 1.0)
    )

    # Recency weight: newer articles → higher weight
    today = dt.date.today()
    df_eng["published_dt"] = pd.to_datetime(df_eng["published"], errors="coerce")
    age_days = (today - df_eng["published_dt"].dt.date).dt.days
    age_days = age_days.clip(lower=0).fillna(365)
    # exponential decay ~ 2 months half-life
    recency_weight = np.exp(-age_days / 60.0)

    # Similarity weight (avoid zeros)
    sim_weight = df_eng["similarity"].clip(lower=0.1).values

    # Sentiment mapping
    sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    sentiment_values = df_eng["sentiment_label"].map(sentiment_map).fillna(0).values
    sentiment_scores = df_eng["sentiment_score"].fillna(0.5).values

    weights = sim_weight * df_eng["source_weight"].values * recency_weight * np.maximum(
        sentiment_scores, 0.3
    )

    weighted_sentiments = sentiment_values * sentiment_scores * weights

    if weights.sum() > 0:
        composite = (weighted_sentiments.sum() / weights.sum()) * 100.0
    else:
        composite = 0.0

    # Never show exactly 100 / -100
    composite = float(np.clip(composite, -95.0, 95.0))

    avg_confidence = float(sentiment_scores.mean() * 100.0)
    avg_confidence = min(avg_confidence, 96.0)

    return {
        "index": round(composite, 2),
        "confidence": round(avg_confidence, 1),
        "total_articles": int(len(df)),
        "weighted_articles": int((weights > 0.3).sum()),
        "english_articles": int(len(df_eng)),
    }


# ============================================================
# ========================= DASHBOARD ========================
# ============================================================

def run_app():
    st.set_page_config(
        page_title="Financial News Sentiment Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ======= HEADER =======
    st.markdown(
        "<h1 style='margin-bottom:0'>📈 Financial News Sentiment Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.caption("RSS → Embeddings → LLM Query Expansion → FinBERT Sentiment → Weighted Index")

    # ======= TOP CONTROLS =======
    with st.container():
        col_q, col_t, col_btn = st.columns([2.5, 1, 1])
        with col_q:
            query = st.text_input(
                "🔍 Topic / Entity / Sector",
                placeholder="Examples: US Tech, Nvidia, Germany, European textiles, Oil, S&P 500...",
            )

        with col_t:
            days_back_label = st.selectbox(
                "Time window",
                ["7 days", "30 days", "90 days"],
                index=1,
            )

        with col_btn:
            analyze_clicked = st.button("🚀 Run Analysis", type="primary")

    # Date range
    today = dt.date.today()
    if days_back_label == "7 days":
        start_date = today - dt.timedelta(days=7)
    elif days_back_label == "30 days":
        start_date = today - dt.timedelta(days=30)
    else:
        start_date = today - dt.timedelta(days=90)

    end_date = today

    # Small global DB stats
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM articles")
    total_articles_db = cur.fetchone()[0] or 0
    cur.execute("SELECT COUNT(*) FROM articles WHERE language = 'en'")
    total_en_articles = cur.fetchone()[0] or 0

    st.caption(
        f"Database: {total_articles_db:,} articles "
        f"(English: {total_en_articles:,}) • Period: {start_date} → {end_date}"
    )

    # If no query or not clicked
    if not analyze_clicked or not query.strip():
        st.info("👆 Enter a topic and click **Run Analysis** to see sentiment.")
        st.write("---")
        st.subheader("🔧 Maintenance / Precomputation (optional)")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Precompute Embeddings", key="pre_emb"):
                with st.spinner("Generating embeddings for new articles..."):
                    count = precompute_embeddings_for_new_articles()
                st.success(f"✅ Generated embeddings for {count} articles.")
        with c2:
            if st.button("Precompute Sentiment", key="pre_sent"):
                with st.spinner("Running FinBERT on new English articles..."):
                    count = precompute_sentiment_for_new_articles()
                st.success(f"✅ Computed sentiment for {count} articles.")
        with c3:
            if st.button("Detect Languages", key="pre_lang"):
                with st.spinner("Detecting languages for unknown articles..."):
                    count = precompute_language_for_unknown()
                st.success(f"✅ Updated language for {count} articles.")

        return

    # =================== MAIN PIPELINE ===================

    # Step 0: Ensure language detection is somewhat up-to-date
    precompute_language_for_unknown()

    # Step 1: Load articles for date range
    with st.spinner("Loading articles for selected period..."):
        df_all = load_articles_for_range(start_date, end_date)

    if df_all.empty:
        st.error("No articles found in the database for this time range.")
        return

    # Step 2: LLM Query expansion
    with st.spinner("Expanding query with LLM (Germany → German industry, etc.)..."):
        expanded_terms = expand_query_with_llm(query)

    st.markdown(
        f"**Expanded search terms:** "
        + ", ".join([f"`{t}`" for t in expanded_terms])
    )

    # Step 3: Hybrid search for each expanded term
    all_results = []
    with st.spinner("Running hybrid semantic + keyword search..."):
        for q_term in expanded_terms:
            res = hybrid_search(q_term, df_all, MIN_SIMILARITY)
            if not res.empty:
                res = res.copy()
                res["query_term"] = q_term
                all_results.append(res)

    if not all_results:
        st.error(f"No relevant articles found for '{query}'. Try a broader topic or longer time window.")
        return

    df = pd.concat(all_results, ignore_index=True)
    # Deduplicate by link
    df = df.sort_values("similarity", ascending=False)
    df = df.drop_duplicates(subset=["link"]).reset_index(drop=True)

    if df.empty:
        st.error("No articles left after deduplication.")
        return

    # Enrich with domain, relevance, etc.
    def source_domain(x: str) -> str:
        s = str(x or "")
        if "//" in s:
            return s.split("//")[-1].split("/")[0]
        return s[:50]

    df["source_domain"] = df["source"].apply(source_domain)
    df["relevance"] = (df["similarity"] * 100).round(1)

    st.success(f"✅ Found **{len(df)}** relevant articles after semantic filtering.")

    # Step 4: Sentiment (precomputed if available; otherwise compute in real-time for EN only)
    with st.spinner("Applying FinBERT sentiment (English only)..."):
        has_pre = not df["precomputed_sentiment_label"].isna().all()

        # Initialize columns
        df["sentiment_label"] = None
        df["sentiment_score"] = None

        # English subset
        mask_en = df["language"] == "en"
        df_en = df[mask_en].copy()

        if df_en.empty:
            # no English articles, we skip FinBERT
            pass
        else:
            if has_pre:
                df.loc[mask_en, "sentiment_label"] = df_en["precomputed_sentiment_label"].values
                df.loc[mask_en, "sentiment_score"] = df_en["precomputed_sentiment_score"].values
                sentiment_source = "precomputed"
            else:
                texts = (df_en["content"].fillna("") + " " + df_en["summary"].fillna("")).tolist()
                texts = [t.strip()[:1000] if t.strip() else "" for t in texts]
                sentiments = finbert_sentiment(texts)
                df.loc[mask_en, "sentiment_label"] = [s["label"] for s in sentiments]
                df.loc[mask_en, "sentiment_score"] = [s["score"] for s in sentiments]
                sentiment_source = "real-time"

        # Non-English: mark as not scored
        df.loc[~mask_en, "sentiment_label"] = "not_scored"
        df.loc[~mask_en, "sentiment_score"] = np.nan

    # Step 5: Weighted sentiment index
    sentiment_results = calculate_weighted_sentiment_index(df)

    # =================== UI TABS ===================
    st.markdown("---")
    tab_overview, tab_articles, tab_sources, tab_debug = st.tabs(
        ["📊 Overview", "📰 Articles", "🏢 Sources", "🧪 Diagnostics"]
    )

    # ------------- OVERVIEW TAB -------------
    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Relevant Articles", sentiment_results["total_articles"])

        with col2:
            st.metric("Articles Used in Index (EN, weighted)", sentiment_results["weighted_articles"])

        index_val = sentiment_results["index"]
        if index_val > 10:
            idx_dir = "📈 Bullish"
        elif index_val < -10:
            idx_dir = "📉 Bearish"
        else:
            idx_dir = "➡️ Neutral"

        with col3:
            st.metric("Sentiment Index", f"{index_val:.1f}", idx_dir)

        conf_val = sentiment_results["confidence"]
        if conf_val > 70:
            conf_level = "High"
        elif conf_val > 50:
            conf_level = "Medium"
        else:
            conf_level = "Low"

        with col4:
            st.metric("Average Confidence", f"{conf_val:.1f}%", conf_level)

        st.caption(
            f"ℹ️ Index is weighted by similarity, source reliability, recency, and FinBERT confidence. "
            f"Only English articles contribute to the index."
        )

        # Sentiment distribution (only EN)
        df_en_index = df[df["language"] == "en"]
        df_en_index = df_en_index[df_en_index["sentiment_label"].isin(["positive", "neutral", "negative"])]

        col_pie, col_bar = st.columns(2)

        with col_pie:
            st.subheader("Sentiment Distribution (English only)")
            if not df_en_index.empty:
                counts = df_en_index["sentiment_label"].value_counts()
                fig_pie = px.pie(
                    names=counts.index,
                    values=counts.values,
                    hole=0.3,
                    color=counts.index,
                    color_discrete_map={
                        "positive": "#2ecc71",
                        "neutral": "#95a5a6",
                        "negative": "#e74c3c",
                    },
                )
                st.plotly_chart(fig_pie)
            else:
                st.info("No English articles with sentiment scored for this query.")

        with col_bar:
            st.subheader("Top News Sources")
            src_counts = df["source_domain"].value_counts().head(10)
            if not src_counts.empty:
                fig_bar = px.bar(
                    x=src_counts.values,
                    y=src_counts.index,
                    orientation="h",
                    labels={"x": "Article Count", "y": "Source"},
                )
                st.plotly_chart(fig_bar)
            else:
                st.info("No sources found.")

        # Optional: small timeline of average daily sentiment
        st.subheader("Daily Average Sentiment (English, if available)")
        if not df_en_index.empty:
            df_en_index["published_dt"] = pd.to_datetime(df_en_index["published"], errors="coerce")
            tmp = df_en_index.dropna(subset=["published_dt"]).copy()
            tmp["date"] = tmp["published_dt"].dt.date
            # map labels to numeric
            sent_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            tmp["sent_value"] = tmp["sentiment_label"].map(sent_map).fillna(0.0)
            # weight by confidence
            tmp["weighted"] = tmp["sent_value"] * tmp["sentiment_score"].fillna(0.5)
            daily = tmp.groupby("date")["weighted"].mean().reset_index()
            daily["index"] = daily["weighted"] * 100.0
            fig_ts = go.Figure()
            fig_ts.add_trace(
                go.Scatter(
                    x=daily["date"],
                    y=daily["index"],
                    mode="lines+markers",
                    name="Daily Sentiment",
                )
            )
            fig_ts.update_layout(
                yaxis_title="Sentiment Index (scaled)",
                xaxis_title="Date",
            )
            st.plotly_chart(fig_ts)
        else:
            st.info("Not enough English articles to plot daily sentiment.")

    # ------------- ARTICLES TAB -------------
    with tab_articles:
        st.subheader(f"Relevant Articles ({len(df)} total)")

        # Sorting controls
        sort_opt = st.selectbox(
            "Sort articles by",
            ["Relevance (high → low)", "Newest first", "Oldest first"],
            index=0,
        )

        df_display = df.copy()
        if sort_opt == "Relevance (high → low)":
            df_display = df_display.sort_values("relevance", ascending=False)
        elif sort_opt == "Newest first":
            df_display["published_dt"] = pd.to_datetime(df_display["published"], errors="coerce")
            df_display = df_display.sort_values("published_dt", ascending=False)
        else:
            df_display["published_dt"] = pd.to_datetime(df_display["published"], errors="coerce")
            df_display = df_display.sort_values("published_dt", ascending=True)

        # Limit display to top 60 to keep UI responsive
        df_display = df_display.head(60)

        for _, row in df_display.iterrows():
            sent = row.get("sentiment_label", "not_scored")
            score = row.get("sentiment_score", np.nan)

            if sent == "positive":
                icon = "🟢"
            elif sent == "negative":
                icon = "🔴"
            elif sent == "neutral":
                icon = "🔵"
            else:
                icon = "⚪"

            st.markdown(f"### {icon} {row['title']}")
            sub_col1, sub_col2, sub_col3, sub_col4 = st.columns([3, 2, 2, 1])

            sub_col1.caption(f"📰 Source: {row.get('source_domain', 'Unknown')}")
            sub_col2.caption(f"🎯 Relevance: {row.get('relevance', 0):.1f}%")
            if sent in ["positive", "neutral", "negative"]:
                sub_col3.caption(f"😊 Sentiment: {sent} ({(score or 0)*100:.1f}%)")
            else:
                sub_col3.caption("😊 Sentiment: not scored (non-English)")
            sub_col4.markdown(f"[Read →]({row['link']})")

            if row.get("published"):
                st.caption(f"📅 Published: {row['published']}")
            if row.get("summary"):
                st.write(row["summary"][:400] + ("..." if len(row["summary"]) > 400 else ""))

            st.markdown("---")

        # Download section
        st.subheader("📥 Download Articles as CSV")
        download_cols = [
            "title",
            "summary",
            "published",
            "link",
            "source_domain",
            "language",
            "relevance",
            "sentiment_label",
            "sentiment_score",
        ]
        csv_df = df[download_cols].copy()
        csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=f"sentiment_results_{query.replace(' ', '_')}.csv",
            mime="text/csv",
        )

    # ------------- SOURCES TAB -------------
    with tab_sources:
        st.subheader("Source Breakdown")

        src_group = df.groupby("source_domain").agg(
            total_articles=("link", "count"),
            avg_relevance=("relevance", "mean"),
        )
        src_group = src_group.sort_values("total_articles", ascending=False)

        st.write("Top 20 sources by article count:")
        st.dataframe(src_group.head(20).round(2))

        # Sentiment by source (EN only)
        df_en_src = df[(df["language"] == "en") & df["sentiment_label"].isin(["positive", "neutral", "negative"])]
        if not df_en_src.empty:
            st.subheader("Average Sentiment per Source (EN only)")
            sent_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            df_en_src["sent_value"] = df_en_src["sentiment_label"].map(sent_map).fillna(0.0)
            df_en_src["weighted"] = df_en_src["sent_value"] * df_en_src["sentiment_score"].fillna(0.5)

            src_sent = df_en_src.groupby("source_domain")["weighted"].mean().reset_index()
            src_sent["index"] = src_sent["weighted"] * 100.0
            src_sent = src_sent.sort_values("index", ascending=False)

            fig_src = px.bar(
                src_sent.head(20),
                x="source_domain",
                y="index",
                labels={"index": "Sentiment Index", "source_domain": "Source"},
            )
            fig_src.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_src)
        else:
            st.info("No English articles with sentiment scored to compute per-source sentiment.")

    # ------------- DEBUG / DIAGNOSTICS TAB -------------
    with tab_debug:
        st.subheader("Diagnostics & Explanation (for Professor)")

        st.markdown("**Query & Expansion**")
        st.write(f"- Original query: `{query}`")
        st.write(f"- Expanded terms: {expanded_terms}")

        st.markdown("**Article Selection Logic**")
        st.write(
            f"""
- We load **all articles** in the time window from the local SQLite DB.
- For each expanded term, we compute a query embedding and compare with each article embedding.
- We also add a small keyword boost if the term appears in title or summary.
- We then **filter by a similarity threshold of {MIN_SIMILARITY}** (not top-N).
- After thresholding, we apply a hard cap of {MAX_ARTICLES_CAP} articles **only for performance**.
- All remaining articles are deduplicated by URL and displayed.
"""
        )

        st.markdown("**Sentiment Scoring Logic**")
        st.write(
            """
- We **only run FinBERT on English articles** (language detected with `langdetect`).
- If sentiment was precomputed, we reuse it; otherwise we run FinBERT at query time.
- We calibrate FinBERT outputs:
  - If max probability < 0.65 → force label = neutral, confidence = 0.5  
  - High probabilities are scaled down (no 1.0 scores).
  - Final sentiment scores are capped at 0.95 so you never see 100% confidence.
"""
        )

        st.markdown("**Composite Sentiment Index**")
        st.write(
            """
- Only English articles with sentiment contribute to the index.
- Each article gets a weight = similarity × source reliability × recency × sentiment confidence.
- We map sentiment labels: positive=+1, neutral=0, negative=-1, multiply by confidence and weights,
  and average to form the final index in **[-100, 100]**.
- To avoid misleading extremes, we clip the index to **[-95, +95]**, so you never see 100%.
"""
        )

        st.markdown("**Quick Maintenance Buttons**")
        st.info(
            "Use the buttons on the main page (when no query is running) to precompute embeddings, "
            "sentiment, and languages in the background."
        )


if __name__ == "__main__":
    run_app()
