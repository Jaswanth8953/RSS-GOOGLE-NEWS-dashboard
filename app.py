import os
import json
import sqlite3
import time
import datetime as dt
from typing import List, Tuple, Dict
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

from gnews import GNews
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, DetectorFactory

# ========================== CONFIG ==========================
DetectorFactory.seed = 0

DB_PATH = "news_articles.db"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
FINBERT_MODEL = "ProsusAI/finbert"

MIN_SIMILARITY = 0.25
MAX_ARTICLES_PER_TERM = 300
RELEVANCE_THRESHOLD = 0.28

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
        for _ in range(5):
            try:
                cur.execute(query, params)
                conn.commit()
                return cur.fetchall() if fetch else cur.lastrowid
            except sqlite3.OperationalError as e:
                if "locked" in str(e):
                    time.sleep(0.5)
                    continue
                raise
        raise Exception("DB locked too long")

# ========================== OPENAI & MODELS ==========================
@st.cache_resource
def get_openai_client():
    api_key = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key missing!")
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
        idx = int(np.argmax(p))
        score = float(p[idx])
        label = ["negative", "neutral", "positive"][idx]
        if score < 0.5:
            label = "neutral"
            score = 0.5
        results.append({"label": label, "score": round(score, 4)})
    return results

# ========================== LANGUAGE & EMBEDDINGS ==========================
def detect_language(text: str) -> str:
    text = (text or "").strip()
    if not text or len(text) < 20:
        return "unknown"
    try:
        return detect(text)
    except:
        return "unknown"

def get_embedding(text: str) -> List[float]:
    client = get_openai_client()
    text = text.replace("\n", " ")[:1000]
    return client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text).data[0].embedding

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

# ========================== INGESTION (WITH PRE-COMPUTED SENTIMENT) ==========================
def ingest_article(title, summary, published, link, source):
    content = summary or ""
    text = f"{title} {content}".strip()
    lang = detect_language(text)
    embedding = json.dumps(get_embedding(text[:1000])) if text else "null"
    
    safe_execute("""
        INSERT OR IGNORE INTO articles 
        (title, summary, published, link, source, content, embedding, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (title, summary, published, link, source, content, embedding, lang))
    
    rowid = safe_execute("SELECT id FROM articles WHERE link = ?", (link,), fetch=True)
    if rowid:
        rowid = rowid[0][0]
        if lang == "en" and len(text) > 30:
            sents = finbert_sentiment([text[:1000]])
            if sents:
                safe_execute("""
                    UPDATE articles SET sentiment_label=?, sentiment_score=? WHERE id=?
                """, (sents[0]["label"], sents[0]["score"], rowid))

def fetch_rss_articles():
    added = 0
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                title = e.get("title", "")[:500]
                link = e.get("link", "")
                if not link or not title:
                    continue
                summary = e.get("summary", "")[:1000]
                pub = dt.datetime.utcnow().isoformat()
                if hasattr(e, "published_parsed") and e.published_parsed:
                    pub = dt.datetime(*e.published_parsed[:6]).isoformat()
                ingest_article(title, summary, pub, link, url)
                added += 1
        except: pass
    return added

# (GDELT and GNews ingestion simplified — same pattern with ingest_article())

# ========================== QUERY EXPANSION ==========================
def expand_query(query: str) -> List[str]:
    client = get_openai_client()
    prompt = f"""Return 8-12 financial search terms related to: "{query}"
Examples:
"US Tech" → ["US technology stocks", "NASDAQ", "FAANG", "AI companies", "semiconductors"]

Return ONLY a JSON array."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        terms = json.loads(resp.choices[0].message.content.strip()).get("terms", [])
        terms = [t.strip() for t in terms if t.strip()]
        return [query] + list(dict.fromkeys(terms))[:12]
    except:
        return [query]

# ========================== HYBRID SEARCH + WEIGHTED SENTIMENT ==========================
def hybrid_search(query: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    rows = safe_execute("""
        SELECT id, title, summary, published, link, source, embedding, language, sentiment_label, sentiment_score
        FROM articles WHERE date(published) BETWEEN ? AND ?
    """, (start.isoformat(), end.isoformat()), fetch=True)
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows, columns=["id","title","summary","published","link","source","embedding","language","sentiment_label","sentiment_score"])
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
    df["relevance"] = (df["similarity"] * 100).round(1)
    return df.sort_values("similarity", ascending=False)

def calculate_sentiment_index(df: pd.DataFrame) -> float:
    mask = (
        (df["language"] == "en") &
        (df["sentiment_label"].isin(["positive", "negative", "neutral"])) &
        (df["similarity"] >= 0.25)
    )
    scored = df[mask]
    if len(scored) < 8:
        return 0.0
    weights = scored["similarity"]
    values = scored["sentiment_label"].map({"positive": 1, "neutral": 0, "negative": -1})
    index = np.average(values, weights=weights) * 100
    return round(float(index), 1)

# ========================== STREAMLIT APP ==========================
def main():
    st.set_page_config(page_title="Financial News Sentiment", layout="wide")
    st.title("Financial News Sentiment Dashboard")
    st.markdown("**Hybrid Search • Pre-computed FinBERT • Relevance-Weighted Index**")

    with st.sidebar:
        st.header("Update Data")
        if st.button("Fetch Latest News"):
            with st.spinner("Updating..."):
                n = fetch_rss_articles()
            st.success(f"Added {n} articles")

        today = dt.date.today()
        start_date = st.date_input("Start", today - dt.timedelta(days=30))
        end_date = st.date_input("End", today)

    query = st.text_input("Search Topic", placeholder="e.g., US Tech, Nvidia, Germany, Oil")
    if not st.button("Analyze") or not query:
        st.info("Enter a topic and click Analyze")
        return

    with st.spinner("Expanding query..."):
        terms = expand_query(query)
    st.write("**Expanded terms:**", " • ".join(terms))

    with st.spinner("Searching articles..."):
        dfs = [hybrid_search(q, start_date, end_date) for q in terms[:8]]
        df = pd.concat(dfs).drop_duplicates("link").head(1000)

    if df.empty:
        st.error("No relevant articles found.")
        return

    sentiment_index = calculate_sentiment_index(df)
    st.metric("Sentiment Index", f"{sentiment_index:+.1f}", 
              delta="Bullish" if sentiment_index > 15 else "Bearish" if sentiment_index < -15 else "Neutral")

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Articles", "Download"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            counts = df["sentiment_label"].value_counts()
            fig = px.pie(values=counts.values, names=counts.index, hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            top_sources = df["source"].apply(lambda x: x.split("//")[-1].split("/")[0] if isinstance(x,str) else "unknown").value_counts().head(10)
            fig2 = px.bar(y=top_sources.index, x=top_sources.values, orientation='h')
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        for _, r in df.head(100).iterrows():
            icon = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}.get(r["sentiment_label"], "?")
            st.markdown(f"**{icon} {r['title']}**")
            st.caption(f"Source: {r['source'][:50]} • Relevance: {r['relevance']:.1f}% • [{r['link']}]({r['link']})")
            if r.get("summary"):
                st.caption(r["summary"][:300] + "...")
            st.divider()

    with tab3:
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, f"sentiment_{query.replace(' ', '_')}.csv", "text/csv")

if __name__ == "__main__":
    main()
