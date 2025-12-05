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

MIN_SIMILARITY = 0.25
MIN_RELEVANCE_SCORE = 0.30
MAX_ARTICLES_DEFAULT = 150
GDELT_MAX_RECORDS = 150
GNEWS_MAX_RESULTS = 150

DetectorFactory.seed = 0

# ========================== DB SETUP WITH BACKWARD COMPATIBILITY ==========================

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    
    # Create main table with basic columns
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
            language TEXT DEFAULT 'unknown'
        );
        """
    )
    
    # Add new columns if they don't exist (backward compatibility)
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN precomputed_sentiment_label TEXT")
    except:
        pass  # Column already exists
    
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN precomputed_sentiment_score REAL")
    except:
        pass
    
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN sentiment_computed INTEGER DEFAULT 0")
    except:
        pass
    
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN embedding_generated INTEGER DEFAULT 0")
    except:
        pass
    
    # Create indices safely
    conn.execute("CREATE INDEX IF NOT EXISTS idx_published ON articles(published);")
    
    # Check if sentiment_computed column exists before creating index
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(articles)")
    columns = [row[1] for row in cur.fetchall()]
    
    if 'sentiment_computed' in columns:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_computed ON articles(sentiment_computed);")
    
    if 'language' in columns:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_language ON articles(language);")
    
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


# ========================== SIMPLE LANGUAGE DETECTION ==========================

def detect_language_simple(text: str) -> str:
    text = (text or "").strip()
    if not text or len(text) < 10:
        return "unknown"
    
    # Simple English detection (good enough for most cases)
    english_words = {'the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'for', 'it'}
    words = text.lower().split()
    english_count = sum(1 for word in words[:20] if word in english_words)
    
    if english_count > 2:
        return "en"
    
    # Check for common non-English patterns
    if any(char in text for char in ['ä', 'ö', 'ü', 'ß']):
        return "de"  # German
    elif any(char in text for char in ['é', 'è', 'ê', 'à', 'ç']):
        return "fr"  # French
    elif any(char in text for char in ['á', 'é', 'í', 'ó', 'ú', 'ñ']):
        return "es"  # Spanish
    elif any(char in text for char in ['的', '是', '在', '和']):
        return "zh"  # Chinese
    
    return "unknown"


def ensure_language(df: pd.DataFrame) -> pd.DataFrame:
    if "language" not in df.columns:
        df["language"] = "unknown"

    mask = df["language"].isna() | df["language"].eq("unknown") | df["language"].eq("")
    for idx, row in df[mask].iterrows():
        text = f"{row.get('title','')} {row.get('summary','')}"
        lang = detect_language_simple(text)
        df.at[idx, "language"] = lang
        try:
            safe_execute(
                "UPDATE articles SET language=? WHERE id=?",
                (lang, int(row["id"]))
            )
        except Exception:
            pass
    return df


# ========================== SIMPLE FINBERT ==========================

@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def finbert_sentiment_simple(texts: List[str]) -> List[Dict]:
    """Simple FinBERT - minimal calibration"""
    if not texts:
        return []
    
    try:
        tokenizer, model, device = load_finbert()
        
        # Clean texts
        texts = [str(t)[:400] if t else "" for t in texts]
        
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
            label = id2label[idx]
            
            # Simple calibration to avoid extremes
            if max_score > 0.95:
                score = 0.85
            elif max_score < 0.60:
                label = "neutral"
                score = 0.65
            else:
                score = max_score
            
            results.append({"label": label, "score": score})
        
        return results
        
    except Exception as e:
        return [{"label": "neutral", "score": 0.65} for _ in texts]


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


# ========================== SIMPLE INGESTION ==========================

def fetch_rss_articles() -> int:
    new_count = 0
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue

        for entry in feed.entries[:10]:  # Limit to 10 per feed
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            link = entry.get("link", "")

            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub = dt.datetime(*entry.published_parsed[:6])
            else:
                pub = dt.datetime.utcnow()
            published = pub.isoformat()

            content = summary
            lang = detect_language_simple(f"{title} {summary}")
            
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
        f"?query={q_encoded}&mode=artlist&maxrecords=100"
        f"&format=json&startdatetime={start_str}&enddatetime={end_str}"
    )

    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return 0
        data = r.json()
        articles = data.get("articles", [])[:50]  # Limit to 50
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

        lang = detect_language_simple(f"{title} {snippet}")

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
        gn = GNews(language="en", max_results=50)
        gn.start_date = (start.year, start.month, start.day)
        gn.end_date = (end.year, end.month, end.day)
        results = gn.get_news(query)
    except Exception:
        return 0

    for r in results[:30]:  # Limit to 30
        title = r.get("title", "")
        link = r.get("url") or r.get("link", "")
        desc = r.get("description", "")
        source = r.get("publisher", {}).get("title", "google-news")
        pub = r.get("published date") or r.get("published_date")

        try:
            published = str(pub) if pub else dt.datetime.utcnow().isoformat()
        except Exception:
            published = dt.datetime.utcnow().isoformat()

        try:
            safe_execute(
                """
                INSERT OR IGNORE INTO articles
                (title, summary, published, link, source, content, language)
                VALUES (?, ?, ?, ?, ?, ?, 'en')
                """,
                (title, desc, published, link, source, desc),
            )
            new_count += 1
        except Exception:
            pass

    return new_count


# ========================== LOAD ARTICLES ==========================

def load_articles_for_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary, published, link, source, content, 
               embedding, language
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
            "id", "title", "summary", "published", "link", "source", "content",
            "embedding", "language"
        ],
    )
    return df


# ========================== SIMPLE HYBRID SEARCH ==========================

def hybrid_search_simple(query: str, start: dt.date, end: dt.date) -> pd.DataFrame:
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
            except:
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
    
    # Filter by relevance threshold
    filtered = df[df["relevance_score"] >= MIN_RELEVANCE_SCORE].copy()
    
    if filtered.empty:
        # Fallback to top 100 by similarity
        filtered = df.sort_values("similarity", ascending=False).head(100)
    
    # Sort and limit
    filtered = filtered.sort_values("relevance_score", ascending=False).head(MAX_ARTICLES_DEFAULT)
    
    return filtered


# ========================== SIMPLE QUERY EXPANSION ==========================

def expand_query_simple(query: str) -> List[str]:
    # Simple rule-based expansion for common queries
    expansions_map = {
        "germany": ["German", "Berlin", "German economy", "Germany stock market"],
        "us tech": ["US technology", "American tech", "Silicon Valley", "tech stocks"],
        "oil": ["crude oil", "oil prices", "energy market", "petroleum"],
        "china": ["Chinese", "Beijing", "China economy", "Chinese market"],
        "inflation": ["inflation rate", "price increase", "CPI", "economic inflation"],
        "interest rates": ["interest rate", "Fed rates", "monetary policy", "central bank"],
    }
    
    query_lower = query.lower()
    expansions = expansions_map.get(query_lower, [])
    
    # Always include the original query
    return [query] + expansions[:4]


# ========================== SENTIMENT ANALYSIS ==========================

def analyze_sentiment_simple(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()
    
    # Initialize sentiment columns
    df_result["sentiment_label"] = "not_scored"
    df_result["sentiment_score"] = 0.5
    
    # Only analyze English articles
    mask_english = df_result["language"] == "en"
    df_english = df_result[mask_english].copy()
    
    if not df_english.empty:
        # Prepare texts
        texts = []
        indices = []
        for idx, row in df_english.iterrows():
            text = f"{row.get('title', '')}. {row.get('summary', '')}"
            if text.strip():
                texts.append(text[:300])
                indices.append(idx)
        
        # Analyze
        if texts:
            sentiments = finbert_sentiment_simple(texts)
            
            for i, idx in enumerate(indices):
                if i < len(sentiments):
                    df_result.at[idx, "sentiment_label"] = sentiments[i]["label"]
                    df_result.at[idx, "sentiment_score"] = sentiments[i]["score"]
    
    return df_result


def calculate_sentiment_metrics_simple(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {
            "total_articles": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "not_scored": 0,
            "sentiment_index": 0.0
        }
    
    total = len(df)
    
    # Count English articles with sentiment
    scored_df = df[df["sentiment_label"].isin(["positive", "negative", "neutral"])]
    pos = len(scored_df[scored_df["sentiment_label"] == "positive"])
    neg = len(scored_df[scored_df["sentiment_label"] == "negative"])
    neu = len(scored_df[scored_df["sentiment_label"] == "neutral"])
    not_scored = total - (pos + neg + neu)
    
    # Calculate sentiment index (only scored articles)
    if len(scored_df) > 0:
        sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        scored_df["sentiment_value"] = scored_df["sentiment_label"].map(sentiment_map)
        
        # Weight by confidence
        weights = scored_df["sentiment_score"].values
        values = scored_df["sentiment_value"].values
        
        weighted_sum = np.sum(values * weights)
        weight_sum = np.sum(weights)
        
        if weight_sum > 0:
            sentiment_index = (weighted_sum / weight_sum) * 100
        else:
            sentiment_index = 0.0
        
        sentiment_index = round(sentiment_index, 1)
    else:
        sentiment_index = 0.0
    
    return {
        "total_articles": total,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "not_scored": not_scored,
        "sentiment_index": sentiment_index
    }


# ========================== SIMPLE KEYWORDS ==========================

def extract_top_keywords_simple(titles: List[str], n: int = 10) -> List[Tuple[str, int]]:
    text = " ".join(titles).lower()
    words = re.findall(r"[a-zA-Z]{4,}", text)
    stop = {"this", "that", "with", "from", "have", "will", "been", "into"}
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
    st.markdown("**Hybrid Search → FinBERT Sentiment Analysis**")
    
    # SIDEBAR
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if st.button("🔄 Fetch RSS", use_container_width=True):
            with st.spinner("Fetching RSS feeds..."):
                n = fetch_rss_articles()
            if n > 0:
                st.success(f"✓ Added {n} articles.")
            else:
                st.info("No new articles.")
        
        st.markdown("---")
        
        today = dt.date.today()
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", today - dt.timedelta(days=30))
        with col2:
            end_date = st.date_input("End date", today)
        
        st.markdown("---")
        
        use_gdelt = st.checkbox("🌐 Use GDELT", value=True)
        use_gnews = st.checkbox("🔍 Use Google News", value=True)
        
        st.markdown("---")
        
        # Database stats
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM articles")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM articles WHERE language = 'en'")
        english = cur.fetchone()[0]
        
        st.metric("📊 Total Articles", f"{total:,}")
        st.metric("🇬🇧 English Articles", f"{english:,}")

    # MAIN AREA
    st.markdown("### 🔍 Search Topic")
    
    query = st.text_input(
        "Enter topic (e.g., 'Germany', 'US Tech', 'Oil')",
        placeholder="Type your topic here...",
        label_visibility="collapsed"
    )
    
    analyze_clicked = st.button("🚀 Analyze", type="primary", use_container_width=True)

    if not analyze_clicked or not query.strip():
        st.info("👆 Enter a topic and click **Analyze** to begin.")
        return

    # ========== MAIN ANALYSIS ==========
    
    # Query Expansion
    with st.spinner("🔄 Expanding query..."):
        expanded_terms = expand_query_simple(query)
    
    if len(expanded_terms) > 1:
        st.markdown("### 🔍 Expanded terms:")
        st.write(", ".join(expanded_terms))

    # Fetch additional articles
    total_added = 0
    if use_gdelt or use_gnews:
        with st.spinner("🌐 Fetching additional articles..."):
            for q in expanded_terms[:3]:
                if use_gdelt:
                    added = fetch_gdelt_articles(q, start_date, end_date)
                    total_added += added
                if use_gnews:
                    added = fetch_gnews_articles(q, start_date, end_date)
                    total_added += added
        
        if total_added > 0:
            st.success(f"✅ Added {total_added} extra articles.")

    # Hybrid Search
    with st.spinner("🔎 Running hybrid search..."):
        all_results = []
        for q in expanded_terms:
            results = hybrid_search_simple(q, start_date, end_date)
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
    
    st.success(f"✅ Found **{len(df)}** relevant articles.")

    # Ensure language
    df = ensure_language(df)

    # Sentiment Analysis
    with st.spinner("😊 Analyzing sentiment..."):
        df = analyze_sentiment_simple(df)
    
    # Calculate relevance percentage
    df["relevance_percent"] = (df["relevance_score"] * 100).round(1)
    df["source_domain"] = df["source"].apply(
        lambda x: str(x).split("//")[-1].split("/")[0] if "//" in str(x) else str(x)[:20]
    )

    # Calculate metrics
    metrics = calculate_sentiment_metrics_simple(df)
    
    # ================== TABS ==================
    
    tab_dash, tab_articles, tab_keywords, tab_download = st.tabs(
        ["📈 Dashboard", "📰 Articles", "🔑 Keywords", "📥 Download"]
    )

    # ----- Dashboard tab -----
    with tab_dash:
        st.subheader("📊 Sentiment Overview")
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📄 Total Articles", metrics["total_articles"])
        
        with col2:
            st.metric("🟢 Positive", metrics["positive"])
        
        with col3:
            st.metric("🔴 Negative", metrics["negative"])
        
        with col4:
            st.metric("🔵 Neutral", metrics["neutral"])
        
        with col5:
            # Sentiment index
            index_val = metrics["sentiment_index"]
            if index_val > 20:
                direction = "📈 Bullish"
            elif index_val < -20:
                direction = "📉 Bearish"
            else:
                direction = "➡️ Neutral"
            
            st.metric(
                "📊 Sentiment Index", 
                f"{index_val:.1f}", 
                direction
            )
        
        st.markdown("---")
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### 📊 Sentiment Distribution")
            
            # Sentiment counts for chart
            sentiment_counts = pd.Series({
                "Positive": metrics["positive"],
                "Negative": metrics["negative"],
                "Neutral": metrics["neutral"]
            })
            
            if sentiment_counts.sum() > 0:
                fig_pie = px.pie(
                    names=sentiment_counts.index,
                    values=sentiment_counts.values,
                    color=sentiment_counts.index,
                    color_discrete_map={
                        "Positive": "#2ecc71",
                        "Negative": "#e74c3c",
                        "Neutral": "#3498db",
                    },
                    hole=0.4,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No sentiment-scored articles.")
        
        with col_chart2:
            st.markdown("#### 🌐 Language Distribution")
            lang_counts = df["language"].value_counts().head(5)
            if not lang_counts.empty:
                fig_lang = px.bar(
                    x=lang_counts.values,
                    y=lang_counts.index,
                    orientation="h",
                    labels={"x": "Count", "y": "Language"},
                )
                st.plotly_chart(fig_lang, use_container_width=True)

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
            sort_by = st.selectbox(
                "Sort by",
                ["Relevance", "Date"],
                index=0
            )
        
        # Apply filters
        df_art = df[df["sentiment_label"].isin(sent_filter)].copy()
        
        if sort_by == "Relevance":
            df_art = df_art.sort_values("relevance_score", ascending=False)
        else:
            df_art = df_art.sort_values("published", ascending=False)
        
        # Display
        for idx, row in df_art.iterrows():
            icon_map = {
                "positive": "🟢",
                "negative": "🔴",
                "neutral": "🔵",
                "not_scored": "⚪",
            }
            icon = icon_map.get(row["sentiment_label"], "⚪")
            
            st.markdown(f"**{icon} {row['title']}**")
            
            cols = st.columns([3, 2, 2, 1])
            cols[0].caption(f"📰 {row['source_domain']} | {row['language']}")
            cols[1].caption(f"🎯 {row['relevance_percent']:.1f}% relevant")
            if row["sentiment_label"] != "not_scored":
                score_pct = row['sentiment_score'] * 100
                cols[2].caption(f"{row['sentiment_label'].title()} ({score_pct:.0f}%)")
            else:
                cols[2].caption("Not scored")
            cols[3].markdown(f"[📖 Read]({row['link']})")
            
            if row.get("summary"):
                summary = str(row["summary"])
                st.caption(summary[:200] + ("..." if len(summary) > 200 else ""))
            
            st.markdown("---")

    # ----- Keywords tab -----
    with tab_keywords:
        st.subheader("🔑 Trending Keywords")
        
        keywords = extract_top_keywords_simple(df["title"].tolist())
        
        if keywords:
            kw_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])
            fig_kw = px.bar(
                kw_df,
                x="Frequency",
                y="Keyword",
                orientation="h",
            )
            fig_kw.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("Not enough data for keywords.")

    # ----- Download tab -----
    with tab_download:
        st.subheader("📥 Download Results")
        
        # Prepare data
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
        dl_df["published"] = pd.to_datetime(dl_df["published"], errors="coerce").dt.strftime("%Y-%m-%d")
        
        # Download button
        csv = dl_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name=f"sentiment_{query.replace(' ', '_')}.csv",
            mime="text/csv",
        )
        
        st.markdown("---")
        st.markdown("#### Preview")
        st.dataframe(dl_df.head(10), use_container_width=True)


if __name__ == "__main__":
    run_app()
