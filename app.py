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

MIN_SIMILARITY = 0.30
MAX_ARTICLES_DEFAULT = 200
GDELT_MAX_RECORDS = 250
GNEWS_MAX_RESULTS = 200

DetectorFactory.seed = 0

# ========================== DB SETUP ==========================

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

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
            precomputed_sentiment_label TEXT,
            precomputed_sentiment_score REAL,
            embedding_generated INTEGER DEFAULT 0,
            sentiment_computed INTEGER DEFAULT 0
        );
    """)

    return conn

conn = get_connection()

def safe_execute(q, p=()):
    for _ in range(10):
        try:
            cur = conn.cursor()
            cur.execute(q, p)
            conn.commit()
            return cur
        except sqlite3.OperationalError as e:
            if "locked" in str(e):
                time.sleep(0.3)
            else:
                raise
    raise Exception("DB write failed")

# ========================== OPENAI API KEY ==========================

@st.cache_resource
def get_openai_client():
    api_key = None

    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        api_key = st.secrets["openai"]["api_key"]

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("❌ Missing OpenAI API key. Add it to secrets.toml or environment variable.")
        st.stop()

    return OpenAI(api_key=api_key)

# ========================== LANGUAGE DETECTION ==========================

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def ensure_language(df):
    mask = (df["language"] == "unknown") | (df["language"].isna())
    for idx, row in df[mask].iterrows():
        text = f"{row['title']} {row['summary']} {row['content']}"
        lang = detect_language(text)
        df.at[idx, "language"] = lang
        safe_execute("UPDATE articles SET language=? WHERE id=?", (lang, int(row["id"])))
    return df

# ========================== FINBERT ==========================

@st.cache_resource
def load_finbert():
    t = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    m = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)
    m.eval()
    return t, m, device

def finbert_sentiment(texts):
    if not texts:
        return []

    tokenizer, model, device = load_finbert()

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()

    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    results = []

    for p in probs:
        idx = int(np.argmax(p))
        maxp = float(p[idx])
        label = id2label[idx]

        if maxp < 0.6:
            label = "neutral"
            score = 0.5
        else:
            score = min(maxp, 0.95)

        if label == "positive" and p[0] > 0.25:
            label = "neutral"
            score = 0.5

        results.append({"label": label, "score": score})
    return results

# ========================== EMBEDDINGS ==========================

def get_embedding(text: str):
    client = get_openai_client()
    text = (text or "")[:500]
    emb = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text)
    return emb.data[0].embedding

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

# ========================== INGESTION ==========================

def fetch_rss_articles():
    count = 0
    for url in RSS_FEEDS:
        feed = feedparser.parse(url)
        for e in feed.entries:
            try:
                published = dt.datetime(*e.published_parsed[:6]).isoformat()
            except:
                published = dt.datetime.utcnow().isoformat()

            title = e.get("title", "")
            summary = e.get("summary", "")
            link = e.get("link", "")
            content = summary

            safe_execute("""
                INSERT OR IGNORE INTO articles
                (title, summary, published, link, source, content)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (title, summary, published, link, url, content))
            count += 1
    return count

def fetch_gdelt(q, start, end):
    start_s = start.strftime("%Y%m%d000000")
    end_s = end.strftime("%Y%m%d235959")

    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={q}&format=json&mode=artlist&maxrecords={GDELT_MAX_RECORDS}"
        f"&startdatetime={start_s}&enddatetime={end_s}"
    )
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
    except:
        return 0

    count = 0
    for a in data.get("articles", []):
        title = a.get("title", "")
        link = a.get("url", "")
        src = a.get("domain", "gdelt")
        content = a.get("snippet", "")

        try:
            published = dt.datetime.strptime(a["seendate"], "%Y%m%d%H%M%S").isoformat()
        except:
            published = dt.datetime.utcnow().isoformat()

        safe_execute("""
            INSERT OR IGNORE INTO articles
            (title, summary, published, link, source, content)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (title, "", published, link, src, content))
        count += 1
    return count

def fetch_gnews(q, start, end):
    gn = GNews(language="en", max_results=GNEWS_MAX_RESULTS)
    gn.start_date = (start.year, start.month, start.day)
    gn.end_date = (end.year, end.month, end.day)

    try:
        results = gn.get_news(q)
    except:
        return 0

    count = 0
    for r in results:
        title = r.get("title", "")
        summary = r.get("description", "")
        link = r.get("url", "")
        src = r.get("publisher", {}).get("title", "")

        published = dt.datetime.utcnow().isoformat()

        safe_execute("""
            INSERT OR IGNORE INTO articles
            (title, summary, published, link, source, content)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (title, summary, published, link, src, summary))
        count += 1
    return count

# ========================== LOAD ARTICLES ==========================

def load_range(start, end):
    rows = conn.execute("""
        SELECT id, title, summary, published, link, source, content, embedding, language
        FROM articles
        WHERE date(published) BETWEEN ? AND ?
    """, (start, end)).fetchall()

    return pd.DataFrame(rows, columns=[
        "id", "title", "summary", "published", "link", "source",
        "content", "embedding", "language"
    ])

# ========================== EMBEDDING UPDATE ==========================

def ensure_embeddings(df):
    for idx, row in df.iterrows():
        if row["embedding"] in (None, "", "null"):
            txt = f"{row['title']} {row['summary']}"
            emb = get_embedding(txt)
            safe_execute("UPDATE articles SET embedding=? WHERE id=?",
                         (json.dumps(emb), int(row["id"])))

# ========================== HYBRID SEARCH ==========================

def hybrid_search(query, start, end, top_k):
    df = load_range(start, end)
    if df.empty:
        return df

    ensure_embeddings(df)

    df = load_range(start, end)

    q_emb = np.array(get_embedding(query))
    sims = []
    for _, row in df.iterrows():
        try:
            e = np.array(json.loads(row["embedding"]))
            sims.append(cosine(q_emb, e))
        except:
            sims.append(0.0)

    df["similarity"] = sims

    kw_mask = (
        df["title"].str.lower().str.contains(query.lower(), na=False)
        | df["summary"].str.lower().str.contains(query.lower(), na=False)
    )
    sem_mask = df["similarity"] >= MIN_SIMILARITY

    df["match_type"] = np.where(
        kw_mask & sem_mask, "keyword+semantic",
        np.where(kw_mask, "keyword", "semantic")
    )

    df = df[kw_mask | sem_mask].sort_values("similarity", ascending=False).head(top_k)
    return df

# ========================== LLM QUERY EXPANSION ==========================

def expand_query(q):
    client = get_openai_client()

    prompt = f'''
    Expand the financial topic "{q}" into 10 related terms.
    Return ONLY a JSON list of strings.
    '''

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        j = json.loads(r.choices[0].message.content)
        final = [q] + j
        out = []
        seen = set()
        for t in final:
            if t.lower() not in seen:
                out.append(t)
                seen.add(t.lower())
        return out
    except:
        return [q]

# ========================== LLM ARTICLE FILTER ==========================

def llm_filter(query, df):
    client = get_openai_client()
    df_small = df.head(200).copy()

    items = [{
        "id": int(r["id"]),
        "title": r["title"],
        "summary": (r["summary"] or "")[:250]
    } for _, r in df_small.iterrows()]

    prompt = f"""
    Topic: {query}
    Here is a list of articles. Return ONLY a JSON list of IDs that are relevant.

    {json.dumps(items)}
    """

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        ids = json.loads(r.choices[0].message.content)
        return df_small[df_small["id"].isin(ids)]
    except:
        return df_small

# ========================== SENTIMENT INDEX ==========================

def sentiment_index(df):
    df_en = df[df["language"] == "en"]
    df_scored = df_en[df_en["sentiment_label"].isin(["positive", "negative", "neutral"])]

    if df_scored.empty:
        return 0

    df_scored["sent_value"] = df_scored["sentiment_label"].map({
        "positive": 1, "neutral": 0, "negative": -1
    })

    score = df_scored["sent_value"].mean() * 100
    return float(np.clip(score, -95, 95))

# ========================== STREAMLIT UI ==========================

def run_app():
    st.set_page_config(page_title="Financial Dashboard", layout="wide")
    st.title("📊 Financial News Sentiment Dashboard")

    # SIDEBAR
    with st.sidebar:
        st.header("Update Sources")

        if st.button("Fetch RSS"):
            n = fetch_rss_articles()
            st.success(f"Added {n} RSS articles")

        today = dt.date.today()
        start = st.date_input("Start date", today - dt.timedelta(days=30))
        end = st.date_input("End date", today)

        use_gd = st.checkbox("Use GDELT", True)
        use_gn = st.checkbox("Use Google News", True)

        max_k = st.slider("Max articles", 50, 500, 200, 50)

    st.markdown("### 🔍 Search Topic")

    query = st.text_input("Enter topic ('US Tech', 'Germany', 'oil', etc.)")

    if st.button("Analyze"):
        if not query.strip():
            st.error("Enter topic.")
            return

        with st.spinner("Expanding query..."):
            expanded = expand_query(query)
        st.write("Expanded:", ", ".join(expanded))

        if use_gd or use_gn:
            with st.spinner("Fetching extra sources..."):
                added = 0
                for q in expanded[:5]:
                    if use_gd: added += fetch_gdelt(q, start, end)
                    if use_gn: added += fetch_gnews(q, start, end)
                st.success(f"Added {added} extra articles")

        # SEARCH
        dfs = []
        for q in expanded:
            part = hybrid_search(q, start, end, max_k)
            if not part.empty:
                part["query_term"] = q
                dfs.append(part)

        if not dfs:
            st.error("No results.")
            return

        df = pd.concat(dfs).drop_duplicates(subset=["link"])

        with st.spinner("LLM filtering..."):
            df = llm_filter(query, df)

        # DETECT LANGUAGE
        df = ensure_language(df)

        # SENTIMENT
        with st.spinner("Running FinBERT..."):
            df["sentiment_label"] = "not_scored"
            df["sentiment_score"] = np.nan

            mask = df["language"] == "en"
            english_df = df[mask]

            if not english_df.empty:
                texts = [(t or "")[:700] for t in english_df["content"].fillna("")]
                s = finbert_sentiment(texts)
                df.loc[mask, "sentiment_label"] = [x["label"] for x in s]
                df.loc[mask, "sentiment_score"] = [x["score"] for x in s]

        df["relevance"] = (df["similarity"] * 100).round(1)
        df["source_domain"] = df["source"].fillna("unknown")

        # SENTIMENT INDEX
        index_val = sentiment_index(df)

        # TABS
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "📰 Articles", "🔑 Keywords", "📥 Download"])

        # ================= DASHBOARD TAB =================
        with tab1:
            st.subheader("Sentiment Overview")

            pos = (df["sentiment_label"] == "positive").sum()
            neg = (df["sentiment_label"] == "negative").sum()
            neu = (df["sentiment_label"] == "neutral").sum()
            total = len(df)

            colA, colB, colC, colD, colE = st.columns(5)
            colA.metric("Total", total)
            colB.metric("Positive", pos)
            colC.metric("Negative", neg)
            colD.metric("Neutral", neu)

            # FIXED BLOCK
            if index_val > 0:
                direction = "Bullish"
            elif index_val < 0:
                direction = "Bearish"
            else:
                direction = "Neutral"

            colE.metric("Sentiment Index", f"{index_val:.1f}", direction)

            st.markdown("---")

            # PIE
            dist = df[df["sentiment_label"].isin(["positive", "negative", "neutral"])]["sentiment_label"].value_counts()
            if not dist.empty:
                fig = px.pie(names=dist.index, values=dist.values)
                st.plotly_chart(fig, use_container_width=True)

        # ================= ARTICLES TAB =================
        with tab2:
            st.subheader("Articles")
            for _, r in df.sort_values("relevance", ascending=False).iterrows():
                st.markdown(f"### {r['title']}")
                st.caption(f"{r['source_domain']} | {r['published']}")
                st.caption(f"Relevance {r['relevance']}% | Sentiment {r['sentiment_label']}")
                st.write(r['summary'])
                st.markdown(f"[Read more]({r['link']})")
                st.markdown("---")

        # ================= KEYWORDS =================
        with tab3:
            st.subheader("Top Keywords")
            titles = " ".join(df["title"].tolist()).lower()
            words = re.findall(r"[a-zA-Z]{4,}", titles)
            freq = Counter(words).most_common(20)
            kdf = pd.DataFrame(freq, columns=["Keyword", "Count"])
            st.dataframe(kdf)

        # ================= DOWNLOAD =================
        with tab4:
            st.subheader("Download Results")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "results.csv")

            st.dataframe(df)

if __name__ == "__main__":
    run_app()
