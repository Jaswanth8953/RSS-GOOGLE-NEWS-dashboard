import os
import json
import sqlite3
import time
import datetime as dt
from typing import List, Dict
from contextlib import contextmanager

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

# ========================== CONFIG ==========================
DetectorFactory.seed = 0

DB_PATH = "news_articles.db"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
FINBERT_MODEL = "ProsusAI/finbert"

RELEVANCE_THRESHOLD = 0.26
MAX_RESULTS = 1000

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "https://www.marketwatch.com/rss/topstories",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/worldNews",
]

# ========================== DATABASE ==========================
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.execute("""
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
            sentiment_label TEXT,
            sentiment_score REAL
        );
    """)
    try:
        yield conn
    finally:
        conn.close()

def safe_execute(query: str, params: tuple = (), fetch=False):
    with get_db() as conn:
        cur = conn.cursor()
        for _ in range(10):
            try:
                cur.execute(query, params)
                conn.commit()
                return cur.fetchall() if fetch else cur.lastrowid
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    time.sleep(0.5)
                    continue
                raise
            except Exception as e:
                raise
        raise Exception("Database locked too long")

# ========================== MODELS ==========================
@st.cache_resource
def get_openai_client():
    api_key = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key missing! Add to secrets or env.")
        st.stop()
    return OpenAI(api_key=api_key)

@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def finbert_sentiment(texts: List[str]) -> List[Dict]:
    if not texts:
        return []
    tokenizer, model, device = load_finbert()
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
    results = []
    for p in probs:
        idx = np.argmax(p)
        score = float(p[idx])
        label = ["negative", "neutral", "positive"][idx]
        if score < 0.5:
            label = "neutral"
            score = 0.5
        results.append({"label": label, "score": round(score, 4)})
    return results

# ========================== HELPERS ==========================
def detect_language(text: str) -> str:
    text = (text or "").strip()
    if len(text) < 30:
        return "unknown"
    try:
        return detect(text)
    except:
        return "unknown"

def get_embedding(text: str) -> List[float]:
    client = get_openai_client()
    text = text.replace("\n", " ")[:1500]
    return client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text).data[0].embedding

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# ========================== INGESTION (PRE-COMPUTED) ==========================
def ingest_article(title: str, summary: str, published: str, link: str, source: str):
    text = f"{title} {summary or ''}".strip()
    lang = detect_language(text)
    embedding = json.dumps(get_embedding(text[:1000])) if text else None

    # Store only YYYY-MM-DD part for reliable filtering
    pub_date = published.split("T")[0] if "T" in published else published.split(" ")[0]

    try:
        safe_execute("""
            INSERT OR IGNORE INTO articles 
            (title, summary, published, link, source, content, embedding, language)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (title, summary, published, link, source, text, embedding, lang))

        # Get row and add sentiment if English
        rows = safe_execute("SELECT id FROM articles WHERE link = ?", (link,), fetch=True)
        if rows and lang == "en" and len(text) > 50:
            rowid = rows[0][0]
            sents = finbert_sentiment([text[:1000]])
            if sents:
                safe_execute("""
                    UPDATE articles SET sentiment_label = ?, sentiment_score = ? WHERE id = ?
                """, (sents[0]["label"], sents[0]["score"], rowid))
    except:
        pass

def fetch_rss_articles() -> int:
    added = 0
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.get("title", "")[:500]
                link = entry.get("link", "") or entry.get("id", "")
                if not title or not link:
                    continue
                summary = entry.get("summary", "")[:2000]
                pub = dt.datetime.utcnow().isoformat()
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub = dt.datetime(*entry.published_parsed[:6]).isoformat()
                ingest_article(title, summary, pub, link, url)
                added += 1
        except:
            continue
    return added

# ========================== QUERY & SEARCH ==========================
def expand_query(query: str) -> List[str]:
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": f"Give 8-10 financial search terms for: {query}\nReturn ONLY a JSON array."}],
            response_format={"type": "json_object"},
            temperature=0.4
        )
        terms = json.loads(resp.choices[0].message.content).get("terms", []) or []
        terms = [t.strip() for t in terms if t.strip()]
        return [query] + list(dict.fromkeys(terms))[:10]
    except:
        return [query]

def hybrid_search(query: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    # FIXED: Use substr() on ISO string instead of date()
    sql = """
        SELECT id, title, summary, published, link, source, embedding, language, sentiment_label, sentiment_score
        FROM articles 
        WHERE substr(published, 1, 10) BETWEEN ? AND ?
          AND embedding IS NOT NULL AND embedding != 'null' AND embedding != ''
    """
    rows = safe_execute(sql, (start_date.isoformat(), end_date.isoformat()), fetch=True)
    
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "id","title","summary","published","link","source","embedding",
        "language","sentiment_label","sentiment_score"
    ])

    q_emb = np.array(get_embedding(query))
    sims = []
    for emb_str in df["embedding"]:
        try:
            vec = np.array(json.loads(emb_str))
            sims.append(cosine_similarity(q_emb, vec))
        except:
            sims.append(0.0)
    df["similarity"] = sims
    df = df[df["similarity"] >= RELEVANCE_THRESHOLD].copy()
    df["relevance_pct"] = (df["similarity"] * 100).round(1)
    return df.sort_values("similarity", ascending=False).head(800)

def calculate_sentiment_index(df: pd.DataFrame) -> float:
    mask = (
        (df["language"] == "en") &
        (df["sentiment_label"].isin(["positive", "negative", "neutral"])) &
        (df["similarity"] >= 0.25)
    )
    scored = df[mask]
    if len(scored) < 10:
        return 0.0
    weights = scored["similarity"]
    values = scored["sentiment_label"].map({"positive": 1.0, "neutral": 0.0, "negative": -1.0})
    index = np.average(values, weights=weights) * 100
    return round(float(index), 1)

# ========================== STREAMLIT APP ==========================
def main():
    st.set_page_config(page_title="Financial News Sentiment", layout="wide")
    st.title("Financial News Sentiment Dashboard")
    st.markdown("**Real-time • Hybrid Search • Relevance-Weighted FinBERT • No 100% Bug**")

    with st.sidebar:
        st.header("Data Controls")
        if st.button("Fetch Latest News", use_container_width=True):
            with st.spinner("Updating database..."):
                n = fetch_rss_articles()
            st.success(f"Added {n} new articles")

        today = dt.date.today()
        start_date = st.date_input("Start Date", today - dt.timedelta(days=30))
        end_date = st.date_input("End Date", today)

    query = st.text_input("Search Topic", placeholder="e.g., Nvidia, US Tech, Germany, Oil Prices")
    
    if not st.button("Analyze", type="primary") or not query.strip():
        st.info("Enter a topic and click **Analyze**")
        with get_db() as conn:
            total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            en = conn.execute("SELECT COUNT(*) FROM articles WHERE language='en'").fetchone()[0]
        st.metric("Total Articles in DB", total)
        st.metric("English Articles", en)
        return

    with st.spinner("Expanding query..."):
        terms = expand_query(query)
    st.write("**Search terms:**", " • ".join(terms))

    with st.spinner("Running hybrid search..."):
        dfs = []
        for q in terms:
            part = hybrid_search(q, start_date, end_date)
            if not part.empty:
                dfs.append(part)
        if not dfs:
            st.error("No relevant articles found.")
            return
        df = pd.concat(dfs).drop_duplicates("link")

    sentiment_index = calculate_sentiment_index(df)

    st.markdown(f"## Sentiment Index: **{sentiment_index:+.1f}**")
    if sentiment_index > 20:
        st.success("Strongly Bullish")
    elif sentiment_index < -20:
        st.error("Strongly Bearish")
    else:
        st.warning("Mixed / Neutral")

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Articles", "Download"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            counts = df["sentiment_label"].value_counts()
            fig = px.pie(values=counts.values, names=counts.index, hole=0.4, title="Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            sources = df["source"].apply(lambda x: x.split("//")[-1].split("/")[0] if isinstance(x, str) else "unknown")
            top = sources.value_counts().head(10)
            fig2 = px.bar(y=top.index, x=top.values, orientation='h', title="Top Sources")
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        for _, r in df.head(150).iterrows():
            icon = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}.get(r["sentiment_label"], "?")
            st.markdown(f"**{icon} {r['title']}**")
            cols = st.columns([3, 1, 1])
            cols[0].caption(f"Source: {r['source'][:60]}")
            cols[1].caption(f"Relevance: {r['relevance_pct']:.1f}%")
            cols[2].markdown(f"[Link]({r['link']})")
            if r.get("summary"):
                st.caption(r["summary"][:300] + ("..." if len(r["summary"]) > 300 else ""))
            st.divider()

    with tab3:
        csv = df.to_csv(index=False).encode()
        st.download_button("Download Full Results (CSV)", csv, f"sentiment_{query.replace(' ', '_')}.csv", "text/csv")

if __name__ == "__main__":
    main()
