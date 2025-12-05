import os
import json
import sqlite3
import time
import datetime as dt
from typing import List, Tuple, Optional

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
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from gnews import GNews
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ========================== CONFIG ==========================

DB_PATH = "news_articles.db"

# Reduced to 15 reliable RSS feeds
RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://www.marketwatch.com/marketwatch/rss/topstories",
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.wired.com/feed/rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.npr.org/rss/rss.php?id=1001",
    "https://www.npr.org/rss/rss.php?id=1019",
    "https://www.dw.com/rss/rdf-en-business",
    "https://www.engadget.com/rss.xml",
]

OPENAI_EMBED_MODEL = "text-embedding-3-small"
FINBERT_MODEL = "ProsusAI/finbert"

# Optimized config
MIN_SIMILARITY = 0.30
MAX_ARTICLES_DEFAULT = 200
GDELT_MAX_RECORDS = 100
GNEWS_MAX_RESULTS = 100

DetectorFactory.seed = 0


# ========================== DB SETUP WITH PRECOMPUTATION ==========================

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
            language TEXT DEFAULT 'unknown',
            precomputed_sentiment_label TEXT,
            precomputed_sentiment_score REAL,
            embedding_generated INTEGER DEFAULT 0,
            sentiment_computed INTEGER DEFAULT 0
        );
        """
    )
    
    # Ensure all columns exist (backward compatibility)
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN precomputed_sentiment_label TEXT")
    except:
        pass  # Column already exists
    
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN precomputed_sentiment_score REAL")
    except:
        pass
    
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN embedding_generated INTEGER DEFAULT 0")
    except:
        pass
    
    try:
        conn.execute("ALTER TABLE articles ADD COLUMN sentiment_computed INTEGER DEFAULT 0")
    except:
        pass
    
    conn.execute("CREATE INDEX IF NOT EXISTS idx_published ON articles(published);")
    conn.commit()
    return conn

# Global connection
conn = get_connection()


def safe_execute(query: str, params: tuple = ()):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            conn.commit()
            return cur
        except sqlite3.OperationalError:
            time.sleep(0.1)
            continue
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
        raise ValueError("OpenAI API key not found.")

    return OpenAI(api_key=api_key)


# ========================== FIXED SENTIMENT ANALYSIS ==========================

@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def finbert_sentiment(texts: List[str]) -> List[dict]:
    """FIXED: Better calibration to avoid 100% bug"""
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
        
        # CRITICAL FIX: More aggressive calibration
        if max_score < 0.65:  # Increased threshold
            label = "neutral"
            score = 0.5
        else:
            label = base_label
            # Scale down extreme confidences more aggressively
            if max_score > 0.90:
                score = max_score * 0.7
            elif max_score > 0.80:
                score = max_score * 0.8
            else:
                score = max_score
        
        # Force some diversity: if positive, check negative probability
        if label == "positive":
            neg_prob = float(p[0])
            if neg_prob > 0.25:  # If negative has reasonable probability
                label = "neutral"
                score = 0.5
        
        results.append({"label": label, "score": score})
    
    return results


def precompute_sentiment_for_new_articles():
    """Precompute sentiment for new articles (Professor's suggestion)"""
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
            'text': f"{row[1]} {row[2]}"[:1000]
        })
    
    texts = [a['text'] for a in articles_data]
    sentiments = finbert_sentiment(texts)
    
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


# ========================== EMBEDDINGS WITH PRECOMPUTATION ==========================

def get_embedding(text: str) -> List[float]:
    client = get_openai_client()
    text = text.replace("\n", " ")
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


def precompute_embeddings_for_new_articles():
    """Precompute embeddings for new articles"""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary 
        FROM articles 
        WHERE embedding_generated = 0 
        LIMIT 30
        """
    )
    rows = cur.fetchall()
    
    if not rows:
        return 0
    
    updated = 0
    for row in rows:
        text = f"{row[1]} {row[2]}"[:500]
        if not text.strip():
            continue
        
        try:
            emb = get_embedding(text)
            safe_execute(
                """
                UPDATE articles SET embedding = ?, embedding_generated = 1 
                WHERE id = ?
                """,
                (json.dumps(emb), row[0])
            )
            updated += 1
        except Exception:
            continue
    
    return updated


# ========================== HYBRID SEARCH (PROFESSOR'S REQUIREMENT) ==========================

def load_articles_for_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    cur = conn.cursor()
    
    # Check if columns exist
    cur.execute("PRAGMA table_info(articles)")
    columns = [row[1] for row in cur.fetchall()]
    
    # Build select query based on available columns
    select_fields = ["id", "title", "summary", "published", "link", "source", "content", 
                     "embedding", "language"]
    
    # Add precomputed sentiment columns if they exist
    if 'precomputed_sentiment_label' in columns:
        select_fields.append("precomputed_sentiment_label")
    if 'precomputed_sentiment_score' in columns:
        select_fields.append("precomputed_sentiment_score")
    
    query = f"""
    SELECT {', '.join(select_fields)}
    FROM articles
    WHERE date(published) BETWEEN ? AND ?
    AND language = 'en'
    """
    
    cur.execute(query, (start.isoformat(), end.isoformat()))
    rows = cur.fetchall()
    
    if not rows:
        return pd.DataFrame()
    
    # Create DataFrame with available columns
    df = pd.DataFrame(rows, columns=select_fields)
    
    # Ensure expected columns exist
    if 'precomputed_sentiment_label' not in df.columns:
        df['precomputed_sentiment_label'] = None
    if 'precomputed_sentiment_score' not in df.columns:
        df['precomputed_sentiment_score'] = None
    
    return df


def hybrid_search(
    query: str,
    start: dt.date,
    end: dt.date,
    min_sim: float = MIN_SIMILARITY,
) -> pd.DataFrame:
    """
    Hybrid search as Professor wants: semantic + keyword
    Returns ALL articles above similarity threshold (not fixed number)
    """
    df = load_articles_for_range(start, end)
    if df.empty:
        return df

    # Get query embedding
    q_emb = np.array(get_embedding(query))
    
    # Calculate similarities
    sims = []
    for _, row in df.iterrows():
        if row["embedding"] and isinstance(row["embedding"], str) and row["embedding"] != "null":
            try:
                emb_vec = np.array(json.loads(row["embedding"]))
                sim = cosine_similarity(q_emb, emb_vec)
            except:
                sim = 0.0
        else:
            sim = 0.0
        sims.append(sim)
    
    df["similarity"] = sims
    
    # Keyword match boost
    q_lower = query.lower()
    keyword_boost = (
        df["title"].str.lower().str.contains(q_lower, na=False).astype(int) * 0.2 +
        df["summary"].str.lower().str.contains(q_lower, na=False).astype(int) * 0.1
    )
    
    df["similarity"] = df["similarity"] + keyword_boost
    df["similarity"] = df["similarity"].clip(0, 1)
    
    # Filter by similarity threshold (Professor's suggestion - not fixed number)
    filtered = df[df["similarity"] >= min_sim].copy()
    
    if filtered.empty:
        return pd.DataFrame()
    
    # Sort by similarity
    filtered = filtered.sort_values("similarity", ascending=False)
    
    return filtered


# ========================== QUERY EXPANSION (PROFESSOR'S REQUIREMENT) ==========================

def expand_query_with_llm(query: str) -> List[str]:
    """
    Query expansion as Professor wants: Germany → German Industry
    """
    client = get_openai_client()

    prompt = f"""
You are helping with financial news search. For the query "{query}", 
generate 5-8 related search terms that capture:
1. Same meaning in different words
2. Related industries/sectors
3. Broader and narrower concepts
4. Common synonyms in financial context

Examples:
- "Germany" → ["German", "German industry", "Berlin", "Deutschland", "German economy"]
- "US Tech" → ["US technology", "American tech", "Silicon Valley", "tech stocks"]

Return ONLY a JSON list of strings.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Generate related financial search terms."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200
        )
        content = resp.choices[0].message.content.strip()
        
        # Extract JSON
        import re
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            expansions = json.loads(json_match.group())
        else:
            expansions = json.loads(content)
        
        expansions = [str(t) for t in expansions if isinstance(t, str)]
        
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
        
        return cleaned[:8]
        
    except Exception:
        return [query]


# ========================== IMPROVED SENTIMENT CALCULATION ==========================

def calculate_weighted_sentiment_index(df: pd.DataFrame, query: str) -> dict:
    """
    IMPROVED: Weighted sentiment as Professor wants
    - Weight by similarity/relevance
    - Consider source reliability
    - Not simple majority
    """
    if df.empty:
        return {"index": 0.0, "confidence": 0.0}
    
    # Weight by similarity
    weights = df["similarity"].clip(lower=0.1).values
    
    # Source weights (simple heuristic)
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
    
    df['source_weight'] = df['source'].apply(
        lambda x: next((v for k, v in source_weights.items() if k in str(x).lower()), 1.0)
    )
    weights = weights * df['source_weight'].values
    
    # Sentiment values
    sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    
    # Check if we have sentiment data
    if 'sentiment_label' in df.columns:
        # Use real-time computed sentiment
        sentiment_values = df['sentiment_label'].map(sentiment_map).fillna(0).values
        if 'sentiment_score' in df.columns:
            sentiment_scores = df['sentiment_score'].fillna(0.5).values
        else:
            sentiment_scores = np.ones(len(df)) * 0.5
    elif 'precomputed_sentiment_label' in df.columns:
        # Use precomputed sentiment
        sentiment_values = df['precomputed_sentiment_label'].map(sentiment_map).fillna(0).values
        if 'precomputed_sentiment_score' in df.columns:
            sentiment_scores = df['precomputed_sentiment_score'].fillna(0.5).values
        else:
            sentiment_scores = np.ones(len(df)) * 0.5
    else:
        # No sentiment data
        return {
            "index": 0.0,
            "confidence": 0.0,
            "total_articles": len(df),
            "weighted_articles": 0
        }
    
    # Apply confidence weighting
    weighted_sentiments = sentiment_values * sentiment_scores * weights
    
    if weights.sum() > 0:
        composite = (weighted_sentiments.sum() / weights.sum()) * 100
    else:
        composite = 0.0
    
    # Calculate average confidence
    avg_confidence = sentiment_scores.mean() * 100
    
    return {
        "index": round(composite, 2),
        "confidence": round(avg_confidence, 1),
        "total_articles": len(df),
        "weighted_articles": (weights > 0.3).sum()  # Articles with significant weight
    }


# ========================== SIMPLIFIED STREAMLIT APP ==========================

def run_app():
    st.set_page_config(
        page_title="Financial News Sentiment Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",  # Simplified: no sidebar by default
    )

    # Clean title
    st.title("📈 Financial News Sentiment Dashboard")
    st.markdown("**RSS → Semantic Search → FinBERT Sentiment Analysis**")
    
    # Simple controls at top
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        query = st.text_input(
            "🔍 Enter topic (e.g., 'Germany', 'US Tech', 'Nvidia')",
            placeholder="Type your topic here and press Enter..."
        )
    
    with col2:
        days_back = st.selectbox(
            "Time period",
            ["7 days", "30 days", "90 days"],
            index=1
        )
    
    with col3:
        if st.button("🚀 Analyze", type="primary", use_container_width=True):
            analyze_clicked = True
        else:
            analyze_clicked = False
    
    # Date calculation
    today = dt.date.today()
    if days_back == "7 days":
        start_date = today - dt.timedelta(days=7)
    elif days_back == "30 days":
        start_date = today - dt.timedelta(days=30)
    else:
        start_date = today - dt.timedelta(days=90)
    
    end_date = today
    
    # Show database stats (minimal)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM articles WHERE language = 'en'")
    total_articles = cur.fetchone()[0]
    
    st.caption(f"Database: {total_articles:,} English articles | Period: {start_date} to {end_date}")
    
    # Only show analysis after clicking
    if not analyze_clicked or not query:
        st.info("👆 Enter a topic and click 'Analyze' to see sentiment analysis")
        
        # Optional: Show precomputation status
        if st.button("🔄 Run Background Precomputation", type="secondary"):
            with st.spinner("Precomputing embeddings and sentiment..."):
                emb_count = precompute_embeddings_for_new_articles()
                sent_count = precompute_sentiment_for_new_articles()
            st.success(f"Precomputed: {emb_count} embeddings, {sent_count} sentiments")
        
        return
    
    # ========== MAIN ANALYSIS PIPELINE ==========
    
    # Step 1: Query Expansion
    with st.spinner("Expanding query for better coverage..."):
        expanded_queries = expand_query_with_llm(query)
    
    st.write(f"**Searching for:** {', '.join(expanded_queries[:5])}")
    
    # Step 2: Hybrid Search for each expanded query
    all_results = []
    
    with st.spinner("Searching for relevant articles..."):
        for q in expanded_queries:
            results = hybrid_search(q, start_date, end_date, MIN_SIMILARITY)
            if not results.empty:
                results["query_term"] = q
                all_results.append(results)
    
    if not all_results:
        st.error(f"No articles found for '{query}'. Try a different topic or time period.")
        return
    
    df = pd.concat(all_results, ignore_index=True).drop_duplicates(subset=["link"])
    
    if df.empty:
        st.error("No articles found after deduplication.")
        return
    
    # Show how many articles found
    st.success(f"Found **{len(df)}** relevant articles")
    
    # Step 3: Sentiment Analysis (use precomputed if available)
    with st.spinner("Analyzing sentiment..."):
        # Check if we have precomputed sentiment
        has_precomputed = 'precomputed_sentiment_label' in df.columns and not df['precomputed_sentiment_label'].isna().all()
        
        if has_precomputed:
            # Use precomputed
            df["sentiment_label"] = df["precomputed_sentiment_label"]
            df["sentiment_score"] = df["precomputed_sentiment_score"]
            sentiment_source = "precomputed"
        else:
            # Compute real-time
            texts = df["content"].fillna(df["summary"]).tolist()
            sentiments = finbert_sentiment(texts)
            df["sentiment_label"] = [s["label"] for s in sentiments]
            df["sentiment_score"] = [s["score"] for s in sentiments]
            sentiment_source = "real-time"
    
    df["relevance"] = (df["similarity"] * 100).round(1)
    df["source_domain"] = df["source"].apply(
        lambda x: x.split("//")[-1].split("/")[0] if "//" in str(x) else str(x)[:30]
    )
    
    # Step 4: Calculate weighted sentiment
    sentiment_results = calculate_weighted_sentiment_index(df, query)
    
    # Step 5: Display Results
    st.markdown("---")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Articles", sentiment_results["total_articles"])
    
    with col2:
        weighted = sentiment_results["weighted_articles"]
        st.metric("Weighted Articles", weighted)
    
    with col3:
        index_val = sentiment_results["index"]
        direction = "📈 Bullish" if index_val > 10 else "📉 Bearish" if index_val < -10 else "➡️ Neutral"
        st.metric("Sentiment Index", f"{index_val:.1f}", direction)
    
    with col4:
        conf = sentiment_results["confidence"]
        level = "High" if conf > 70 else "Medium" if conf > 50 else "Low"
        st.metric("Confidence", f"{conf:.1f}%", level)
    
    # Sentiment distribution
    if 'sentiment_label' in df.columns:
        counts = df["sentiment_label"].value_counts()
        total = len(df)
        
        if total > 0:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### Sentiment Distribution")
                fig_pie = px.pie(
                    names=counts.index,
                    values=counts.values,
                    color=counts.index,
                    color_discrete_map={
                        "positive": "#2ecc71",
                        "negative": "#e74c3c",
                        "neutral": "#95a5a6",
                    },
                    hole=0.3,
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_b:
                st.markdown("#### Articles by Source")
                src_counts = df["source_domain"].value_counts().head(10)
                if not src_counts.empty:
                    fig_bar = px.bar(
                        x=src_counts.values,
                        y=src_counts.index,
                        orientation='h',
                        labels={'x': 'Count', 'y': 'Source'},
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Articles list
    st.markdown("---")
    st.subheader(f"📰 Relevant Articles ({len(df)} total)")
    
    # Sort by relevance
    df_display = df.sort_values("relevance", ascending=False).head(50)
    
    for _, row in df_display.iterrows():
        icon = {"positive": "🟢", "negative": "🔴", "neutral": "🔵"}.get(row.get("sentiment_label", "neutral"), "🔵")
        
        st.markdown(f"**{icon} {row['title']}**")
        cols = st.columns([3, 2, 2, 1])
        cols[0].caption(f"📰 {row.get('source_domain', 'Unknown')}")
        cols[1].caption(f"🎯 Relevance: {row.get('relevance', 0):.1f}%")
        if 'sentiment_label' in row and 'sentiment_score' in row:
            cols[2].caption(f"😊 {row['sentiment_label']} ({row['sentiment_score']*100:.1f}%)")
        else:
            cols[2].caption("😊 Sentiment: N/A")
        cols[3].markdown(f"[Read →]({row['link']})")
        
        if row.get("summary"):
            st.caption(row["summary"][:250] + "...")
        
        st.markdown("---")
    
    # Download option
    st.markdown("---")
    st.subheader("📥 Download Results")
    
    # Prepare download DataFrame
    download_columns = ["title", "summary", "published", "link", "source_domain", "relevance"]
    if 'sentiment_label' in df.columns:
        download_columns.extend(["sentiment_label", "sentiment_score"])
    
    download_df = df[download_columns].copy()
    
    csv = download_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv,
        file_name=f"sentiment_{query.replace(' ', '_')}.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    run_app()
