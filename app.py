import os
import json
import sqlite3
import time
import datetime as dt
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import feedparser
import requests
import torch
from collections import Counter
import re
import plotly.express as px

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

MIN_SIMILARITY = 0.25  # not heavily used, but kept
MIN_RELEVANCE_SCORE = 0.30  # default for slider
MAX_ARTICLES_DEFAULT = 150  # not directly used now, kept for future
GDELT_MAX_RECORDS = 40       # reduced for speed + less noise
GNEWS_MAX_RESULTS = 40       # reduced for speed + less noise

DetectorFactory.seed = 0

# ========================== DB SETUP WITH SENTIMENT PRECOMPUTATION ==========================

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    
    # Create main table
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
            sentiment_computed INTEGER DEFAULT 0,
            embedding_generated INTEGER DEFAULT 0
        );
        """
    )
    
    # Create indices
    conn.execute("CREATE INDEX IF NOT EXISTS idx_published ON articles(published);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_language ON articles(language);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_computed ON articles(sentiment_computed);")
    
    conn.commit()
    return conn


conn = get_connection()


def safe_execute(query: str, params: tuple = ()):
    max_retries = 10
    for _ in range(max_retries):
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            conn.commit()
            return cur
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                time.sleep(0.3)
                continue
            raise
    raise Exception("Database write failed")


# ========================== OPENAI CLIENT ==========================

@st.cache_resource
def get_openai_client():
    api_key = None
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        api_key = st.secrets["openai"]["api_key"]

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "❌ OpenAI API key missing.\n"
            "Add it to .streamlit/secrets.toml under [openai].api_key\n"
            "or set environment variable OPENAI_API_KEY."
        )
        st.stop()

    return OpenAI(api_key=api_key)


# ========================== LANGUAGE DETECTION ==========================

def detect_language(text: str) -> str:
    text = (text or "").strip()
    if not text or len(text) < 10:
        return "unknown"
    try:
        return detect(text)
    except (LangDetectException, Exception):
        return "unknown"


def ensure_language(df: pd.DataFrame) -> pd.DataFrame:
    if "language" not in df.columns:
        df["language"] = "unknown"

    mask = df["language"].isna() | df["language"].eq("unknown") | df["language"].eq("")
    for idx, row in df[mask].iterrows():
        text = f"{row.get('title','')} {row.get('summary','')}"
        lang = detect_language(text)
        df.at[idx, "language"] = lang
        try:
            safe_execute(
                "UPDATE articles SET language=? WHERE id=?",
                (lang, int(row["id"]))
            )
        except Exception:
            pass
    return df


# ========================== PROPER FINBERT WITH CALIBRATION ==========================

@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def finbert_sentiment_with_calibration(texts: List[str]) -> List[Dict]:
    """Proper FinBERT with calibration to avoid 100% confidence"""
    if not texts:
        return []
    
    try:
        tokenizer, model, device = load_finbert()
        
        # Clean texts
        texts = [str(t)[:500] if t else "" for t in texts]
        
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        id2label = {0: "negative", 1: "neutral", 2: "positive"}
        results = []
        
        for p in probs:
            idx = int(np.argmax(p))
            max_score = float(p[idx])
            label = id2label[idx]
            
            # PROPER CALIBRATION
            if max_score > 0.95:
                score = 0.85  # Cap extreme confidence
            elif max_score > 0.90:
                score = max_score * 0.9
            elif max_score > 0.80:
                score = max_score * 0.95
            elif max_score < 0.60:
                # Low confidence -> neutral with moderate score
                label = "neutral"
                score = 0.65
            else:
                score = max_score
            
            # Check for balanced probabilities
            sorted_probs = np.sort(p)[::-1]
            if sorted_probs[0] - sorted_probs[1] < 0.20:  # Close competition
                if label != "neutral":  # If not already neutral
                    label = "neutral"
                    score = 0.65
            
            results.append({"label": label, "score": score})
        
        return results
        
    except Exception as e:
        st.error(f"FinBERT error: {str(e)[:100]}")
        return [{"label": "neutral", "score": 0.65} for _ in texts]


# ========================== PRE-COMPUTE SENTIMENT ==========================

def precompute_sentiment_for_new_articles():
    """Pre-compute sentiment for new English articles (for speed)"""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary, content 
        FROM articles 
        WHERE sentiment_computed = 0 
        AND language = 'en'
        LIMIT 50
        """
    )
    rows = cur.fetchall()
    
    if not rows:
        return 0
    
    articles_data = []
    for row in rows:
        articles_data.append({
            'id': row[0],
            'text': f"{row[1]} {row[2]}"[:800]
        })
    
    texts = [a['text'] for a in articles_data]
    sentiments = finbert_sentiment_with_calibration(texts)
    
    updated = 0
    for i, sentiment in enumerate(sentiments):
        try:
            safe_execute(
                """
                UPDATE articles 
                SET precomputed_sentiment_label = ?, 
                    precomputed_sentiment_score = ?,
                    sentiment_computed = 1
                WHERE id = ?
                """,
                (sentiment['label'], sentiment['score'], articles_data[i]['id'])
            )
            updated += 1
        except:
            continue
    
    return updated


# ========================== EMBEDDINGS ==========================

def get_embedding(text: str) -> List[float]:
    client = get_openai_client()
    text = (text or "").replace("\n", " ")
    if len(text) > 500:
        text = text[:500]
    emb = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text
    )
    return emb.data[0].embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def precompute_embeddings_for_new_articles(limit: int = 100) -> int:
    """
    Generate embeddings for articles that don't have them yet (English only).
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary 
        FROM articles
        WHERE (embedding IS NULL OR embedding = '')
        AND language = 'en'
        LIMIT ?
        """,
        (limit,)
    )
    rows = cur.fetchall()
    if not rows:
        return 0

    for art_id, title, summary in rows:
        text = f"{title} {summary}".strip()
        if not text:
            continue
        try:
            emb = get_embedding(text)
            safe_execute(
                "UPDATE articles SET embedding = ?, embedding_generated = 1 WHERE id = ?",
                (json.dumps(emb), art_id)
            )
        except Exception:
            continue

    return len(rows)


# ========================== INGESTION ==========================

def fetch_rss_articles() -> int:
    new_count = 0
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue

        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            link = entry.get("link", "")

            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub = dt.datetime(*entry.published_parsed[:6])
            else:
                pub = dt.datetime.utcnow()
            published = pub.isoformat()

            content = summary
            
            # Get language
            text_for_lang = f"{title} {summary}"
            lang = detect_language(text_for_lang)
            
            try:
                safe_execute(
                    """
                    INSERT OR IGNORE INTO articles
                    (title, summary, published, link, source, content, language)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (title, summary, published, link, url, content, lang),
                )
                new_count += 1
            except Exception:
                pass
    return new_count


def fetch_gdelt_articles(query: str, start: dt.date, end: dt.date) -> int:
    new_count = 0
    start_str = start.strftime("%Y%m%d000000")
    end_str = end.strftime("%Y%m%d235959")
    q_encoded = requests.utils.quote(query)

    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={q_encoded}&mode=artlist&maxrecords={GDELT_MAX_RECORDS}"
        f"&format=json&startdatetime={start_str}&enddatetime={end_str}"
    )

    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return 0
        data = r.json()
        articles = data.get("articles", [])
    except Exception:
        return 0

    for a in articles:
        title = a.get("title", "")
        link = a.get("url", "")
        src = a.get("domain", "gdelt")
        snippet = a.get("snippet", "") or ""
        seendate = a.get("seendate", "")

        try:
            published = dt.datetime.strptime(seendate, "%Y%m%d%H%M%S").isoformat()
        except Exception:
            published = dt.datetime.utcnow().isoformat()

        # Get language
        lang = detect_language(f"{title} {snippet}")

        try:
            safe_execute(
                """
                INSERT OR IGNORE INTO articles
                (title, summary, published, link, source, content, language)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (title, snippet, published, link, src, snippet, lang),
            )
            new_count += 1
        except Exception:
            pass

    return new_count


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
        desc = r.get("description", "")
        source = r.get("publisher", {}).get("title", "google-news")
        pub = r.get("published date") or r.get("published_date")

        try:
            published = str(pub) if pub else dt.datetime.utcnow().isoformat()
        except Exception:
            published = dt.datetime.utcnow().isoformat()

        # GNews is mostly English
        lang = "en"

        try:
            safe_execute(
                """
                INSERT OR IGNORE INTO articles
                (title, summary, published, link, source, content, language)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (title, desc, published, link, source, desc, lang),
            )
            new_count += 1
        except Exception:
            pass

    return new_count


# ========================== LOAD ARTICLES ==========================

def load_articles_for_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Use substr(published,1,10) because published is ISO with 'T' and SQLite date()
    doesn't handle that well.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary, published, link, source, content, 
               embedding, language, precomputed_sentiment_label, 
               precomputed_sentiment_score
        FROM articles
        WHERE substr(published, 1, 10) BETWEEN ? AND ?
        """,
        (start.isoformat(), end.isoformat()),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "id", "title", "summary", "published", "link", "source", "content",
            "embedding", "language", "precomputed_sentiment_label", 
            "precomputed_sentiment_score"
        ],
    )
    return df


# ========================== HYBRID SEARCH WITH RELEVANCE THRESHOLD ==========================

def hybrid_search_with_threshold(query: str, start: dt.date, end: dt.date, relevance_threshold: float) -> pd.DataFrame:
    """
    Hybrid semantic + keyword search with configurable relevance threshold.
    """
    df = load_articles_for_range(start, end)
    if df.empty:
        return df
    
    # Get query embedding
    q_emb = np.array(get_embedding(query))
    
    # Calculate similarities
    sims = []
    for _, row in df.iterrows():
        if row["embedding"] and row["embedding"] not in ["NULL", "null", ""]:
            try:
                emb_vec = np.array(json.loads(row["embedding"]))
                sim = cosine_similarity(q_emb, emb_vec)
            except Exception:
                sim = 0.0
        else:
            sim = 0.0
        sims.append(sim)
    
    df["similarity"] = sims
    
    # Keyword boost
    q_lower = query.lower()
    keyword_boost = (
        df["title"].str.lower().str.contains(q_lower, na=False).astype(int) * 0.1
    )
    
    df["relevance_score"] = df["similarity"] + keyword_boost
    df["relevance_score"] = df["relevance_score"].clip(0, 1)
    
    # Use the dynamic threshold from UI
    filtered = df[df["relevance_score"] >= relevance_threshold].copy()
    
    if filtered.empty:
        # Fallback to top 100 by similarity
        filtered = df.sort_values("similarity", ascending=False).head(100)
    
    # Sort by relevance
    filtered = filtered.sort_values("relevance_score", ascending=False)
    
    return filtered


# ========================== LLM QUERY EXPANSION ==========================

def expand_query_with_llm(query: str) -> List[str]:
    client = get_openai_client()
    
    prompt = f"""
You are a financial news search assistant. For the query "{query}", 
generate 5-7 related search terms that capture:
1. Synonyms and related concepts
2. Broader and narrower terms
3. Related industries/sectors
4. Common phrases in financial news

Return ONLY a JSON array of strings.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Generate related financial search terms. Return JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        content = resp.choices[0].message.content.strip()
        
        # Extract JSON
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            expansions = json.loads(json_match.group())
        else:
            expansions = json.loads(content)
        
        expansions = [str(t).strip() for t in expansions if isinstance(t, str)]
        
        # Combine with original
        all_terms = [query] + expansions
        
        # Deduplicate
        seen = set()
        cleaned = []
        for t in all_terms:
            t = t.strip()
            if not t:
                continue
            if t.lower() in seen:
                continue
            seen.add(t.lower())
            cleaned.append(t)
        
        return cleaned[:7]
        
    except Exception:
        return [query]


# ========================== CALCULATE WEIGHTED SENTIMENT INDEX ==========================

def calculate_weighted_sentiment_index(df: pd.DataFrame) -> Dict:
    """
    Proper weighted sentiment index:
    - Only use English articles
    - Weight by relevance and source reliability
    - Use sentiment scores as confidence weights
    """
    if df.empty:
        return {
            "index": 0.0,
            "confidence": 0.0,
            "total_articles": 0,
            "english_articles": 0,
            "weighted_articles": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
    
    # Only English for sentiment
    df_english = df[df["language"] == "en"].copy()
    
    if df_english.empty:
        return {
            "index": 0.0,
            "confidence": 0.0,
            "total_articles": len(df),
            "english_articles": 0,
            "weighted_articles": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
    
    # Weight by relevance
    weights = df_english["relevance_score"].clip(lower=0.1).values
    
    # Source reliability weights
    source_weights = {
        'reuters': 1.2,
        'bbc': 1.2,
        'cnbc': 1.1,
        'bloomberg': 1.1,
        'wsj': 1.1,
        'ft': 1.1,
        'marketwatch': 1.0,
        'techcrunch': 1.0,
        'theverge': 0.9,
        'wired': 0.9,
        'aljazeera': 0.9,
        'gdelt': 0.8,
        'google-news': 0.8,
    }
    
    df_english['source_weight'] = df_english['source'].apply(
        lambda x: next((v for k, v in source_weights.items() if k in str(x).lower()), 1.0)
    )
    weights = weights * df_english['source_weight'].values
    
    # Get sentiment data (precomputed or computed)
    if 'precomputed_sentiment_label' in df_english.columns and not df_english['precomputed_sentiment_label'].isna().all():
        sentiment_labels = df_english['precomputed_sentiment_label'].values
        sentiment_scores = df_english['precomputed_sentiment_score'].fillna(0.5).values
    elif 'sentiment_label' in df_english.columns:
        sentiment_labels = df_english['sentiment_label'].values
        sentiment_scores = df_english['sentiment_score'].fillna(0.5).values
    else:
        return {
            "index": 0.0,
            "confidence": 0.0,
            "total_articles": len(df),
            "english_articles": len(df_english),
            "weighted_articles": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
    
    # Sentiment values
    sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    sentiment_values = np.array([sentiment_map.get(l, 0.0) for l in sentiment_labels])
    
    # Apply confidence weighting
    weighted_sentiments = sentiment_values * sentiment_scores * weights
    
    # Calculate composite index
    if weights.sum() > 0:
        composite = (weighted_sentiments.sum() / weights.sum()) * 100
        composite = np.clip(composite, -100, 100)
    else:
        composite = 0.0
    
    # Counts
    positive = np.sum((sentiment_labels == "positive") & (weights > 0.1))
    negative = np.sum((sentiment_labels == "negative") & (weights > 0.1))
    neutral = np.sum((sentiment_labels == "neutral") & (weights > 0.1))
    
    # Average confidence
    avg_confidence = np.mean(sentiment_scores[weights > 0.1]) * 100 if np.any(weights > 0.1) else 0.0
    
    return {
        "index": round(composite, 1),
        "confidence": round(avg_confidence, 1),
        "total_articles": len(df),
        "english_articles": len(df_english),
        "weighted_articles": int(np.sum(weights > 0.3)),
        "positive": int(positive),
        "negative": int(negative),
        "neutral": int(neutral)
    }


# ========================== KEYWORDS ==========================

def extract_top_keywords(titles: List[str], n: int = 15) -> List[Tuple[str, int]]:
    text = " ".join(titles).lower()
    words = re.findall(r"[a-zA-Z]{4,}", text)
    stop = {
        "this", "that", "with", "from", "have", "will", "been", "into", "after", "over",
        "under", "they", "them", "your", "their", "about", "which", "there", "where",
        "when", "than", "because", "while", "before", "through", "within", "without",
    }
    words = [w for w in words if w not in stop and not w.isdigit()]
    counter = Counter(words)
    return counter.most_common(n)


# ========================== STREAMLIT APP ==========================

def run_app():
    st.set_page_config(
        page_title="Financial News Sentiment Dashboard",
        layout="wide",
    )

    st.title("📊 Financial News Sentiment Dashboard")
    st.markdown("**RSS → Semantic Search → FinBERT Sentiment Analysis**")
    
    # SIDEBAR
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Pre-compute sentiment button
        if st.button("🔄 Pre-compute Sentiment", use_container_width=True):
            with st.spinner("Pre-computing sentiment for new articles..."):
                count = precompute_sentiment_for_new_articles()
                if count > 0:
                    st.success(f"✓ Pre-computed sentiment for {count} articles.")
                else:
                    st.info("No new articles need pre-computation.")
        
        # Pre-compute embeddings button
        if st.button("🧠 Generate Embeddings", use_container_width=True):
            with st.spinner("Generating embeddings for new articles..."):
                count = precompute_embeddings_for_new_articles()
                if count > 0:
                    st.success(f"✓ Generated embeddings for {count} articles.")
                else:
                    st.info("No new articles need embeddings.")
        
        st.markdown("---")
        
        # Date range
        today = dt.date.today()
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", today - dt.timedelta(days=30))
        with col2:
            end_date = st.date_input("End date", today)
        
        st.markdown("---")
        
        # Source selection
        use_gdelt = st.checkbox("🌐 Use GDELT", value=True)
        use_gnews = st.checkbox("🔍 Use Google News", value=True)
        
        st.markdown("---")
        
        # Relevance threshold slider
        st.markdown("### 🎯 Relevance Settings")
        relevance_threshold = st.slider(
            "Minimum relevance score",
            0.1, 0.5, MIN_RELEVANCE_SCORE, 0.05,
            help="Articles below this relevance score will be excluded"
        )
        
        st.markdown("---")
        
        # Database stats
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM articles")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM articles WHERE language = 'en'")
        english = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM articles WHERE sentiment_computed = 1")
        precomputed = cur.fetchone()[0]
        
        st.metric("📊 Total Articles", f"{total:,}")
        st.metric("🇬🇧 English Articles", f"{english:,}")
        st.metric("⚡ Pre-computed", f"{precomputed:,}")

    # MAIN AREA
    st.markdown("### 🔍 Search Topic")
    
    query = st.text_input(
        "Enter topic (e.g., 'Germany', 'US Tech', 'Oil Prices')",
        placeholder="Type your topic here...",
        label_visibility="collapsed"
    )
    
    analyze_clicked = st.button("🚀 Analyze", type="primary", use_container_width=True)

    if not analyze_clicked or not query.strip():
        st.info("👆 Enter a topic and click **Analyze** to begin.")
        return

    # ========== MAIN ANALYSIS PIPELINE ==========

    # Step 1: Query Expansion
    with st.spinner("🔄 Expanding query..."):
        expanded_terms = expand_query_with_llm(query)
    
    if len(expanded_terms) > 1:
        st.markdown("### 🔍 Expanded terms:")
        st.write(", ".join(expanded_terms[:5]) + ("..." if len(expanded_terms) > 5 else ""))

    # Step 2: Fetch additional articles
    total_added = 0
    if use_gdelt or use_gnews:
        with st.spinner("🌐 Fetching additional articles..."):
            # Limit to first 4 expansion terms to control load
            for q in expanded_terms[:4]:
                if use_gdelt:
                    added = fetch_gdelt_articles(q, start_date, end_date)
                    total_added += added
                if use_gnews:
                    added = fetch_gnews_articles(q, start_date, end_date)
                    total_added += added
        
        if total_added > 0:
            st.success(f"✅ Added {total_added} extra articles.")

    # Step 3: Hybrid Search with threshold
    with st.spinner("🔎 Running hybrid search with relevance threshold..."):
        all_results = []
        for q in expanded_terms:
            results = hybrid_search_with_threshold(q, start_date, end_date, relevance_threshold)
            if not results.empty:
                results["query_term"] = q
                all_results.append(results)
        
        if not all_results:
            st.error("❌ No relevant articles found.")
            return
        
        df = pd.concat(all_results, ignore_index=True).drop_duplicates(subset=["link"])
        
        if df.empty:
            st.error("❌ No articles found after deduplication.")
            return

        # Ensure final filter by current threshold (extra safety)
        if "relevance_score" in df.columns:
            df = df[df["relevance_score"] >= relevance_threshold].copy()
            if df.empty:
                st.error("❌ No articles meet the current relevance threshold.")
                return
    
    st.success(f"✅ Found **{len(df)}** relevant articles (relevance ≥ {relevance_threshold:.2f}).")

    # Step 4: Ensure language detection
    df = ensure_language(df)

    # Step 5: Sentiment Analysis
    with st.spinner("😊 Analyzing sentiment..."):
        # Use precomputed sentiment if available
        if 'precomputed_sentiment_label' in df.columns and not df['precomputed_sentiment_label'].isna().all():
            df["sentiment_label"] = df["precomputed_sentiment_label"]
            df["sentiment_score"] = df["precomputed_sentiment_score"]
            st.info("⚡ Using pre-computed sentiment data.")
        else:
            # Analyze English articles only
            df["sentiment_label"] = "not_scored"
            df["sentiment_score"] = 0.5
            
            mask_english = df["language"] == "en"
            df_english = df[mask_english].copy()
            
            if not df_english.empty:
                texts = []
                indices = []
                for idx, row in df_english.iterrows():
                    text = f"{row.get('title', '')}. {row.get('summary', '')}"
                    if text.strip():
                        texts.append(text[:600])
                        indices.append(idx)
                
                if texts:
                    sentiments = finbert_sentiment_with_calibration(texts)
                    
                    for i, idx in enumerate(indices):
                        if i < len(sentiments):
                            df.at[idx, "sentiment_label"] = sentiments[i]["label"]
                            df.at[idx, "sentiment_score"] = sentiments[i]["score"]
            
            # Mark non-English as not scored
            df.loc[df["language"] != "en", "sentiment_label"] = "not_scored"

    # Calculate relevance percentage for display
    df["relevance_percent"] = (df["relevance_score"] * 100).round(1)
    
    # Calculate weighted sentiment metrics
    sentiment_metrics = calculate_weighted_sentiment_index(df)
    
    # ================== TABS ==================
    
    tab_dash, tab_articles, tab_keywords, tab_download = st.tabs(
        ["📈 Dashboard", "📰 Articles", "🔑 Keywords", "📥 Download"]
    )

    # ----- Dashboard tab -----
    with tab_dash:
        st.subheader("📊 Sentiment Overview")
        
        # Display key metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("📄 Total Articles", sentiment_metrics["total_articles"])
        
        with col2:
            st.metric("🇬🇧 English Articles", sentiment_metrics["english_articles"])
        
        with col3:
            st.metric("🟢 Positive", sentiment_metrics["positive"])
        
        with col4:
            st.metric("🔴 Negative", sentiment_metrics["negative"])
        
        with col5:
            st.metric("🔵 Neutral", sentiment_metrics["neutral"])
        
        with col6:
            # Sentiment index with direction
            index_val = sentiment_metrics["index"]
            if index_val > 20:
                direction = "📈 Strongly Bullish"
                delta_color = "normal"
            elif index_val > 10:
                direction = "📈 Bullish"
                delta_color = "normal"
            elif index_val < -20:
                direction = "📉 Strongly Bearish"
                delta_color = "inverse"
            elif index_val < -10:
                direction = "📉 Bearish"
                delta_color = "inverse"
            else:
                direction = "➡️ Neutral"
                delta_color = "off"
            
            st.metric(
                "📊 Sentiment Index", 
                f"{index_val:.1f}", 
                direction,
                delta_color=delta_color
            )
        
        st.markdown("---")
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### 📊 Sentiment Distribution (English Only)")
            
            # Filter for scored sentiments
            scored_df = df[(df["language"] == "en") & (df["sentiment_label"].isin(["positive", "negative", "neutral"]))]
            
            if not scored_df.empty:
                dist = scored_df["sentiment_label"].value_counts()
                fig_pie = px.pie(
                    names=dist.index,
                    values=dist.values,
                    color=dist.index,
                    color_discrete_map={
                        "positive": "#2ecc71",
                        "negative": "#e74c3c",
                        "neutral": "#3498db",
                    },
                    hole=0.4,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No English articles with sentiment scores.")
        
        with col_chart2:
            st.markdown("#### 🌐 Language Distribution")
            lang_counts = df["language"].value_counts().head(8)
            if not lang_counts.empty:
                fig_lang = px.bar(
                    x=lang_counts.values,
                    y=lang_counts.index,
                    orientation="h",
                    labels={"x": "Count", "y": "Language"},
                    color=lang_counts.values,
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig_lang, use_container_width=True)
        
        # Confidence distribution
        st.markdown("#### 📈 Sentiment Confidence Distribution")
        scored_df = df[df["sentiment_label"].isin(["positive", "negative", "neutral"])]
        if not scored_df.empty:
            fig_conf = px.histogram(
                scored_df,
                x="sentiment_score",
                nbins=20,
                labels={"sentiment_score": "Confidence Score", "count": "Articles"},
                color_discrete_sequence=["#3498db"]
            )
            fig_conf.update_layout(
                xaxis_title="Confidence Score (0-1)",
                yaxis_title="Number of Articles",
                bargap=0.1
            )
            st.plotly_chart(fig_conf, use_container_width=True)

    # ----- Articles tab -----
    with tab_articles:
        st.subheader(f"📰 Relevant Articles ({len(df)} total)")
        
        # Filters
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            sent_filter = st.multiselect(
                "Filter by sentiment",
                options=["positive", "negative", "neutral", "not_scored"],
                default=["positive", "negative", "neutral"],
            )
        
        with col_filter2:
            lang_filter = st.multiselect(
                "Filter by language",
                options=df["language"].unique().tolist(),
                default=["en"] if "en" in df["language"].values else df["language"].unique().tolist()[:3],
            )
        
        # Apply filters
        df_art = df[df["sentiment_label"].isin(sent_filter) & df["language"].isin(lang_filter)].copy()
        
        # Sort by relevance
        df_art = df_art.sort_values("relevance_score", ascending=False)
        
        # Display articles
        for idx, row in df_art.iterrows():
            icon_map = {
                "positive": "🟢",
                "negative": "🔴",
                "neutral": "🔵",
                "not_scored": "⚪",
            }
            icon = icon_map.get(row["sentiment_label"], "⚪")
            
            with st.container():
                col_title, col_link = st.columns([5, 1])
                with col_title:
                    st.markdown(f"**{icon} {row['title']}**")
                with col_link:
                    st.markdown(f"[📖 Read]({row['link']})")
                
                # Metadata
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                with col_meta1:
                    st.caption(f"📰 {str(row['source'])[:30]} | {row['language']}")
                with col_meta2:
                    st.caption(f"🎯 {row['relevance_percent']:.1f}% relevant")
                with col_meta3:
                    if row["sentiment_label"] in ["positive", "negative", "neutral"]:
                        score_pct = row['sentiment_score'] * 100
                        st.caption(f"{row['sentiment_label'].title()} ({score_pct:.0f}%)")
                    else:
                        st.caption("Not scored (non-English)")
                
                # Summary
                if row.get("summary"):
                    summary = str(row["summary"])
                    st.caption(summary[:200] + ("..." if len(summary) > 200 else ""))
                
                st.markdown("---")

    # ----- Keywords tab -----
    with tab_keywords:
        st.subheader("🔑 Trending Keywords")
        
        keywords = extract_top_keywords(df["title"].tolist(), n=15)
        
        if keywords:
            # Display as metrics
            st.markdown("#### Top Keywords in Titles")
            cols = st.columns(3)
            for i, (word, freq) in enumerate(keywords[:9]):
                with cols[i % 3]:
                    st.metric(word.title(), freq)
            
            # Bar chart
            st.markdown("#### Keyword Frequency")
            kw_df = pd.DataFrame(keywords[:12], columns=["Keyword", "Frequency"])
            fig_kw = px.bar(
                kw_df,
                x="Frequency",
                y="Keyword",
                orientation="h",
                color="Frequency",
                color_continuous_scale="Viridis",
            )
            fig_kw.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("Not enough data for keyword analysis.")

    # ----- Download tab -----
    with tab_download:
        st.subheader("📥 Download Results")
        
        # Prepare data for download
        dl_df = df[
            [
                "title",
                "summary",
                "published",
                "link",
                "source",
                "language",
                "sentiment_label",
                "sentiment_score",
                "relevance_percent",
                "query_term",
            ]
        ].copy()
        
        # Format date
        dl_df["published"] = pd.to_datetime(dl_df["published"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        
        # Download button
        csv = dl_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name=f"sentiment_{query.replace(' ', '_')}_{dt.date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("#### 📋 Data Preview")
        st.dataframe(
            dl_df.head(10),
            use_container_width=True,
            hide_index=True,
            column_config={
                "link": st.column_config.LinkColumn("Link"),
                "sentiment_score": st.column_config.ProgressColumn(
                    "Sentiment Score",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "relevance_percent": st.column_config.ProgressColumn(
                    "Relevance",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                )
            }
        )


if __name__ == "__main__":
    run_app()
