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

MIN_SIMILARITY = 0.20
MAX_ARTICLES_DEFAULT = 200
GDELT_MAX_RECORDS = 250
GNEWS_MAX_RESULTS = 200

DetectorFactory.seed = 0

# ========================== DB SETUP ==========================

def get_connection():
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
            language TEXT DEFAULT 'unknown'
        );
        """
    )
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
    if not text:
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
        text = f"{row.get('title','')} {row.get('summary','')} {row.get('content','')}"
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


# ========================== SIMPLE FINBERT SENTIMENT ==========================

@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def finbert_sentiment(texts: List[str]) -> List[dict]:
    """Simple FinBERT sentiment - no complex calibration"""
    if not texts:
        return []
    
    try:
        tokenizer, model, device = load_finbert()
        
        # Process in small batches
        batch_size = 32
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Clean texts
            batch_texts = [str(t)[:500] if t else "" for t in batch_texts]
            
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**enc)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            
            id2label = {0: "negative", 1: "neutral", 2: "positive"}
            
            for p in probs:
                idx = int(np.argmax(p))
                max_score = float(p[idx])
                label = id2label[idx]
                
                # Very simple logic
                if max_score > 0.60:  # Good confidence
                    score = max_score
                else:  # Low confidence
                    label = "neutral"
                    score = 0.5
                
                all_results.append({"label": label, "score": score})
        
        return all_results
        
    except Exception as e:
        # Return neutral for all if error
        return [{"label": "neutral", "score": 0.5} for _ in texts]


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
            try:
                safe_execute(
                    """
                    INSERT OR IGNORE INTO articles
                    (title, summary, published, link, source, content, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (title, summary, published, link, url, content),
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

        try:
            safe_execute(
                """
                INSERT OR IGNORE INTO articles
                (title, summary, published, link, source, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (title, snippet, published, link, src, snippet),
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

        try:
            safe_execute(
                """
                INSERT OR IGNORE INTO articles
                (title, summary, published, link, source, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (title, desc, published, link, source, desc),
            )
            new_count += 1
        except Exception:
            pass

    return new_count


# ========================== LOAD + ENSURE EMBEDDINGS ==========================

def load_articles_for_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary, published, link, source, content, embedding, language
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
        ],
    )
    return df


def ensure_embeddings(df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        emb_str = row.get("embedding")
        if emb_str in (None, "", "NULL", "null"):
            text = f"{row.get('title','')} {row.get('summary','')}"
            if not text.strip():
                continue
            try:
                emb = get_embedding(text)
                safe_execute(
                    "UPDATE articles SET embedding=? WHERE id=?",
                    (json.dumps(emb), int(row["id"])),
                )
            except Exception:
                pass


# ========================== HYBRID SEARCH ==========================

def hybrid_search(query: str, start: dt.date, end: dt.date, top_k: int) -> pd.DataFrame:
    df = load_articles_for_range(start, end)
    if df.empty:
        return df

    ensure_embeddings(df)
    df = load_articles_for_range(start, end)
    if df.empty:
        return df

    q_emb = np.array(get_embedding(query))
    sims = []
    for _, row in df.iterrows():
        try:
            emb_str = row["embedding"]
            if emb_str and emb_str not in ["NULL", "null", ""]:
                emb_vec = np.array(json.loads(emb_str))
                sims.append(cosine_similarity(q_emb, emb_vec))
            else:
                sims.append(0.0)
        except Exception:
            sims.append(0.0)
    df["similarity"] = sims

    q_lower = query.lower().strip()
    kw_mask = (
        df["title"].str.lower().str.contains(q_lower, na=False)
        | df["summary"].str.lower().str.contains(q_lower, na=False)
        | df["content"].str.lower().str.contains(q_lower, na=False)
    )
    sem_mask = df["similarity"] >= MIN_SIMILARITY

    if kw_mask.any() or sem_mask.any():
        filtered = df[kw_mask | sem_mask]
    else:
        return pd.DataFrame()

    filtered = filtered.sort_values("similarity", ascending=False).head(top_k)
    return filtered


# ========================== LLM QUERY EXPANSION ==========================

def expand_query(query: str) -> List[str]:
    client = get_openai_client()
    prompt = f"""Generate 5-8 financial news search terms related to: "{query}"

Return ONLY a JSON array of strings like ["term1", "term2", "term3"]"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return JSON arrays only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200,
        )
        content = resp.choices[0].message.content.strip()
        
        # Clean response
        content = content.replace('```json', '').replace('```', '').strip()
        
        expansions = json.loads(content)
        
        if not isinstance(expansions, list):
            return [query]
        
        expansions = [str(t).strip() for t in expansions if t and str(t).strip()]
        
        all_terms = [query.strip()] + expansions
        seen = set()
        cleaned = []
        for t in all_terms:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(t)
        
        return cleaned[:8]
        
    except Exception:
        return [query]


# ========================== SENTIMENT CALCULATION ==========================

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze sentiment for all articles"""
    df_result = df.copy()
    
    # Initialize sentiment columns
    df_result["sentiment_label"] = "neutral"
    df_result["sentiment_score"] = 0.5
    
    # Prepare texts for sentiment analysis
    texts = []
    indices = []
    
    for idx, row in df_result.iterrows():
        # Use title + summary for context
        text = f"{row.get('title', '')}. {row.get('summary', '')}"
        if text.strip() and len(text.strip()) > 10:
            texts.append(text[:400])
            indices.append(idx)
    
    # If we have texts to analyze
    if texts:
        try:
            sentiments = finbert_sentiment(texts)
            
            # Apply sentiments to DataFrame
            for i, idx in enumerate(indices):
                if i < len(sentiments):
                    df_result.at[idx, "sentiment_label"] = sentiments[i]["label"]
                    df_result.at[idx, "sentiment_score"] = sentiments[i]["score"]
        
        except Exception:
            pass  # Keep default neutral
    
    return df_result


def calculate_sentiment_metrics(df: pd.DataFrame) -> dict:
    """Calculate all sentiment metrics"""
    if df.empty:
        return {
            "total_articles": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "sentiment_index": 0.0,
            "avg_confidence": 0.0
        }
    
    total = len(df)
    
    # Count sentiments
    counts = df["sentiment_label"].value_counts()
    pos = int(counts.get("positive", 0))
    neg = int(counts.get("negative", 0))
    neu = int(counts.get("neutral", 0))
    
    # Calculate sentiment index
    if total > 0:
        # Simple calculation: (positive% - negative%) * 100
        pos_pct = (pos / total) * 100
        neg_pct = (neg / total) * 100
        sentiment_index = (pos_pct - neg_pct)
        sentiment_index = round(sentiment_index, 1)
        
        # Calculate average confidence
        avg_confidence = df["sentiment_score"].mean() * 100
        avg_confidence = round(avg_confidence, 1)
    else:
        sentiment_index = 0.0
        avg_confidence = 0.0
    
    return {
        "total_articles": total,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "sentiment_index": sentiment_index,
        "avg_confidence": avg_confidence
    }


# ========================== KEYWORDS ==========================

def extract_top_keywords(titles: List[str], n: int = 20) -> List[Tuple[str, int]]:
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
    st.markdown("**Hybrid Search → FinBERT Sentiment Analysis**")

    # SIDEBAR
    with st.sidebar:
        st.header("📥 Update Sources")
        
        if st.button("🔄 Fetch RSS", use_container_width=True):
            with st.spinner("Fetching RSS feeds..."):
                n = fetch_rss_articles()
            if n > 0:
                st.success(f"✓ Added {n} RSS articles.")
            else:
                st.info("No new RSS articles.")
        
        st.markdown("---")
        
        today = dt.date.today()
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", today - dt.timedelta(days=30))
        with col2:
            end_date = st.date_input("End date", today)
        
        st.markdown("---")
        
        use_gdelt = st.checkbox("✅ Use GDELT", value=True)
        use_gnews = st.checkbox("✅ Use Google News", value=True)
        
        st.markdown("---")
        
        max_articles = st.slider(
            "📊 Max articles per term",
            50, 500, MAX_ARTICLES_DEFAULT, 50,
        )

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
        
        # Show database stats
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM articles")
        total = cur.fetchone()[0]
        
        st.markdown("---")
        st.metric("📊 Total Articles in Database", f"{total:,}")
        
        return

    # ========== MAIN ANALYSIS ==========
    
    # Query Expansion
    with st.spinner("🔄 Expanding query..."):
        expanded_terms = expand_query(query)
    
    if len(expanded_terms) > 1:
        st.markdown("### 🔍 Expanded terms:")
        st.write(", ".join(expanded_terms[:6]))

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
        dfs = []
        for q in expanded_terms:
            part = hybrid_search(q, start_date, end_date, max_articles)
            if not part.empty:
                part["query_term"] = q
                dfs.append(part)

        if not dfs:
            st.error("❌ No relevant articles found.")
            return

        df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["link"])
        
        if df.empty:
            st.error("❌ No articles found after deduplication.")
            return
    
    st.success(f"✅ Found **{len(df)}** candidate articles.")

    # Select top articles
    target_articles = 150
    if len(df) > target_articles:
        df = df.head(target_articles)
        st.info(f"📊 Using top {target_articles} most relevant articles.")
    else:
        st.success(f"✅ Using all {len(df)} articles.")

    # Language detection
    df = ensure_language(df)

    # Sentiment Analysis
    with st.spinner("😊 Analyzing sentiment..."):
        df = analyze_sentiment(df)
    
    # Calculate relevance and source domain
    df["relevance"] = (df["similarity"] * 100).round(1)
    df["source_domain"] = df["source"].apply(
        lambda x: x.split("//")[-1].split("/")[0]
        if isinstance(x, str) and "//" in x
        else (str(x)[:30] if x else "unknown")
    )

    # Calculate sentiment metrics
    metrics = calculate_sentiment_metrics(df)
    
    # ================== TABS ==================
    
    tab_dash, tab_articles, tab_keywords, tab_download = st.tabs(
        ["📈 Dashboard", "📰 Articles", "🔑 Keywords", "📥 Download"]
    )

    # ----- Dashboard tab -----
    with tab_dash:
        st.subheader("📊 Sentiment Overview")
        
        # Display metrics
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
            # Determine sentiment direction
            sentiment_index = metrics["sentiment_index"]
            if sentiment_index > 15:
                direction = "📈 Bullish"
            elif sentiment_index < -15:
                direction = "📉 Bearish"
            else:
                direction = "➡️ Neutral"
            
            st.metric(
                "📊 Sentiment Index", 
                f"{sentiment_index:.1f}", 
                direction
            )
        
        st.markdown("---")
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### 📊 Sentiment Distribution")
            
            # Get sentiment counts for chart
            sentiment_counts = pd.Series({
                "Positive": metrics["positive"],
                "Negative": metrics["negative"],
                "Neutral": metrics["neutral"]
            })
            
            # Create pie chart
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
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+label"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_chart2:
            st.markdown("#### 📰 Top Sources")
            src_counts = df["source_domain"].value_counts().head(8)
            if not src_counts.empty:
                fig_src = px.bar(
                    x=src_counts.values,
                    y=src_counts.index,
                    orientation="h",
                    labels={"x": "Articles", "y": "Source"},
                )
                st.plotly_chart(fig_src, use_container_width=True)

    # ----- Articles tab -----
    with tab_articles:
        st.subheader(f"📰 Relevant Articles ({len(df)} total)")
        
        # Filters
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            sent_filter = st.multiselect(
                "Filter by sentiment",
                options=["positive", "negative", "neutral"],
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
            df_art = df_art.sort_values("relevance", ascending=False)
        else:
            df_art = df_art.sort_values("published", ascending=False)
        
        # Display articles
        for idx, row in df_art.iterrows():
            icon_map = {
                "positive": "🟢",
                "negative": "🔴",
                "neutral": "🔵",
            }
            icon = icon_map.get(row["sentiment_label"], "🔵")
            
            st.markdown(f"**{icon} {row['title']}**")
            
            cols = st.columns([3, 2, 2, 1])
            cols[0].caption(f"📰 {row['source_domain']}")
            cols[1].caption(f"🎯 {row['relevance']:.1f}% relevant")
            cols[2].caption(f"{row['sentiment_label'].title()} ({row['sentiment_score']*100:.0f}%)")
            cols[3].markdown(f"[📖 Read]({row['link']})")
            
            if row.get("summary"):
                summary = str(row["summary"])
                st.caption(summary[:200] + ("..." if len(summary) > 200 else ""))
            
            st.markdown("---")

    # ----- Keywords tab -----
    with tab_keywords:
        st.subheader("🔑 Trending Keywords")
        
        keywords = extract_top_keywords(df["title"].tolist(), n=15)
        
        if keywords:
            # Bar chart
            kw_df = pd.DataFrame(keywords[:12], columns=["Keyword", "Frequency"])
            fig_kw = px.bar(
                kw_df,
                x="Frequency",
                y="Keyword",
                orientation="h",
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
                "source_domain",
                "language",
                "sentiment_label",
                "sentiment_score",
                "relevance",
                "query_term",
            ]
        ].copy()
        
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
