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
    """
    Load OpenAI API key from:
      1) Streamlit secrets -> [openai].api_key
      2) Environment variable OPENAI_API_KEY
    """
    api_key = None

    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        api_key = st.secrets["openai"]["api_key"]

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "❌ OpenAI API key missing.\n\n"
            "Add it to .streamlit/secrets.toml under:\n"
            "[openai]\napi_key = \"YOUR_KEY\"\n\n"
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


# ========================== FINBERT ==========================

@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def finbert_sentiment(texts: List[str]) -> List[dict]:
    """FinBERT with calibration to avoid everything being 100% positive."""
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
        label = id2label[idx]

        if max_score < 0.60:
            label = "neutral"
            score = 0.50
        else:
            score = min(max_score, 0.95)

        if label == "positive" and p[0] > 0.25:
            label = "neutral"
            score = 0.50

        results.append({"label": label, "score": score})
    return results


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
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ========================== INGESTION (RSS / GDELT / GNEWS) ==========================

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


# ========================== LOAD + EMBEDDINGS ==========================

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
            emb = get_embedding(text)
            try:
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
            emb_vec = np.array(json.loads(row["embedding"]))
            sims.append(cosine_similarity(q_emb, emb_vec))
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
        df["match_type"] = np.where(
            kw_mask & sem_mask,
            "keyword+semantic",
            np.where(kw_mask, "keyword", np.where(sem_mask, "semantic", "none")),
        )
        filtered = df[kw_mask | sem_mask]
    else:
        return pd.DataFrame()

    filtered = filtered.sort_values("similarity", ascending=False).head(top_k)
    return filtered


# ========================== LLM QUERY EXPANSION ==========================

def expand_query(query: str) -> List[str]:
    client = get_openai_client()
    prompt = f"""
You expand financial news search queries.

Original query: "{query}"

Generate 8–12 related search terms that include:
- synonyms
- related industries/sectors
- broader and narrower concepts
- common phrases used in financial news

Return ONLY a JSON list of strings, e.g.:
["term1", "term2", "term3"]
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        content = resp.choices[0].message.content.strip()

        try:
            expansions = json.loads(content)
        except Exception:
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if not match:
                raise ValueError("No JSON list in response")
            expansions = json.loads(match.group(0))

        if not isinstance(expansions, list):
            raise ValueError("Expansion is not a list")

        expansions = [str(t) for t in expansions if isinstance(t, str)]
        all_terms = [query] + expansions

        seen = set()
        cleaned = []
        for t in all_terms:
            t_clean = t.strip()
            if not t_clean:
                continue
            key = t_clean.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(t_clean)

        return cleaned

    except Exception as e:
        st.warning(f"⚠️ Query expansion failed, using original term only. ({e})")
        return [query]


# ========================== LLM ARTICLE FILTER (ROBUST) ==========================

def llm_select_articles(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust LLM-based relevance selector.
    If LLM returns invalid JSON → recover gracefully.
    """
    if df.empty:
        return df

    client = get_openai_client()
    df_small = df.head(200).copy()

    items = [
        {"id": int(r["id"]), "title": r["title"], "summary": (r["summary"] or "")[:250]}
        for _, r in df_small.iterrows()
    ]

    prompt = f"""
You MUST return JSON ONLY.

Topic: "{query}"

Below is a list of articles (id, title, summary).
Return ONLY a JSON list of IDs that are relevant.

Example:
[1, 5, 7, 10]

Articles:
{json.dumps(items, indent=2)}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        content = resp.choices[0].message.content.strip()

        # 1) Direct JSON parse
        try:
            ids = json.loads(content)
            if isinstance(ids, list):
                ids = [int(x) for x in ids]
                return df_small[df_small["id"].isin(ids)]
        except Exception:
            pass

        # 2) Extract JSON list from text
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            try:
                ids = json.loads(match.group(0))
                if isinstance(ids, list):
                    ids = [int(x) for x in ids]
                    return df_small[df_small["id"].isin(ids)]
            except Exception:
                pass

        # 3) Fallback
        st.warning("⚠️ LLM returned invalid JSON. Using unfiltered top candidate articles.")
        return df_small

    except Exception as e:
        st.warning(f"⚠️ LLM filtering crashed: {e}")
        return df_small


# ========================== SENTIMENT INDEX ==========================

def calculate_sentiment_index(df: pd.DataFrame) -> float:
    if df.empty or "sentiment_label" not in df.columns:
        return 0.0

    df_en = df[
        (df["language"] == "en")
        & df["sentiment_label"].isin(["positive", "neutral", "negative"])
    ].copy()
    if df_en.empty:
        return 0.0

    sent_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    df_en["sent_value"] = df_en["sentiment_label"].map(sent_map).fillna(0.0)
    index = df_en["sent_value"].mean() * 100.0
    index = float(np.clip(index, -95.0, 95.0))
    return round(index, 2)


# ========================== KEYWORDS ==========================

def extract_top_keywords(titles: List[str], n: int = 20) -> List[Tuple[str, int]]:
    text = " ".join(titles).lower()
    words = re.findall(r"[a-zA-Z]{4,}", text)
    stop = {
        "this", "that", "with", "from", "have", "will", "been", "into", "after", "over",
        "under", "they", "them", "your", "their", "about", "which", "there", "where",
        "when", "than", "because", "while", "before", "through", "within", "without",
    }
    words = [w for w in words if w not in stop]
    counter = Counter(words)
    return counter.most_common(n)


# ========================== STREAMLIT APP ==========================

def run_app():
    st.set_page_config(
        page_title="Financial News Sentiment Dashboard",
        layout="wide",
    )

    st.title("📊 Financial News Sentiment Dashboard")

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.header("Update Sources")

        if st.button("Fetch RSS"):
            with st.spinner("Fetching RSS feeds..."):
                n = fetch_rss_articles()
            st.success(f"Added {n} RSS articles.")

        today = dt.date.today()
        start_date = st.date_input("Start date", today - dt.timedelta(days=30))
        end_date = st.date_input("End date", today)

        use_gdelt = st.checkbox("Use GDELT", True)
        use_gnews = st.checkbox("Use Google News", True)

        max_articles = st.slider(
            "Max articles per expanded term",
            50,
            500,
            MAX_ARTICLES_DEFAULT,
            50,
        )

    # ---------- MAIN AREA ----------
    st.markdown("### 🔍 Search Topic")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        query = st.text_input(
            "Enter topic ('US Tech', 'Germany', 'oil', etc.)"
        )

    analyze_clicked = st.button("Analyze")

    if not analyze_clicked:
        st.info("Enter a topic and click **Analyze**.")
        return

    if not query.strip():
        st.warning("Please enter a non-empty topic.")
        return

    # Query expansion
    with st.spinner("Expanding query via LLM..."):
        expanded_terms = expand_query(query)
    st.write("🔍 **Expanded terms used:**")
    st.write(", ".join(expanded_terms))

    # Extra sources
    if use_gdelt or use_gnews:
        with st.spinner("Fetching additional articles from GDELT / Google News..."):
            total_added = 0
            for q in expanded_terms[:5]:
                if use_gdelt:
                    total_added += fetch_gdelt_articles(q, start_date, end_date)
                if use_gnews:
                    total_added += fetch_gnews_articles(q, start_date, end_date)
        st.success(f"Added {total_added} extra articles for this topic.")

    # Hybrid search
    with st.spinner("Running hybrid keyword + semantic search..."):
        dfs = []
        for q in expanded_terms:
            part = hybrid_search(q, start_date, end_date, max_articles)
            if not part.empty:
                part["query_term"] = q
                dfs.append(part)

        if not dfs:
            st.error("No relevant articles found for this topic in the selected date range.")
            return

        df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["link"])

    st.success(f"🔎 Hybrid search found **{len(df)}** candidate articles.")

    # LLM article selection
    with st.spinner("Letting LLM select the most relevant subset..."):
        df = llm_select_articles(query, df)

    if df.empty:
        st.error("No articles remained after LLM filtering.")
        return

    st.success(f"✅ LLM kept **{len(df)}** highly relevant articles.")

    # Ensure language
    df = ensure_language(df)

    # Sentiment
    with st.spinner("Running FinBERT sentiment (English only)..."):
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
    df["source_domain"] = df["source"].apply(
        lambda x: x.split("//")[-1].split("/")[0]
        if isinstance(x, str) and "//" in x
        else (x or "unknown")
    )

    sent_index = calculate_sentiment_index(df)

    # ---------- TABS ----------
    tab_dash, tab_articles, tab_keywords, tab_download = st.tabs(
        ["📈 Dashboard", "📰 Articles", "🔑 Keywords", "📥 Download"]
    )

    # ----- Dashboard tab -----
    with tab_dash:
        st.subheader("Sentiment Overview")

        counts = df["sentiment_label"].value_counts()
        total = int(counts.sum())
        pos = int(counts.get("positive", 0))
        neg = int(counts.get("negative", 0))
        neu = int(counts.get("neutral", 0))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Articles", total)
        c2.metric("Positive", pos)
        c3.metric("Negative", neg)
        c4.metric("Neutral", neu)

        if sent_index > 0:
            direction = "Bullish"
        elif sent_index < 0:
            direction = "Bearish"
        else:
            direction = "Neutral"
        c5.metric("Sentiment Index", f"{sent_index:.1f}", direction)

        st.markdown("---")

        dist = df[
            df["sentiment_label"].isin(["positive", "negative", "neutral"])
        ]["sentiment_label"].value_counts()

        if not dist.empty:
            st.markdown("#### Sentiment Distribution")
            fig_pie = px.pie(
                names=dist.index,
                values=dist.values,
                color=dist.index,
                color_discrete_map={
                    "positive": "#2ecc71",
                    "negative": "#e74c3c",
                    "neutral": "#95a5a6",
                },
                hole=0.3,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No English-scored articles available for sentiment distribution.")

        st.markdown("#### Articles by Source")
        src_counts = df["source_domain"].value_counts()
        if not src_counts.empty:
            fig_src = px.bar(
                x=src_counts.values,
                y=src_counts.index,
                orientation="h",
                labels={"x": "Count", "y": "Source"},
            )
            st.plotly_chart(fig_src, use_container_width=True)

    # ----- Articles tab -----
    with tab_articles:
        st.subheader("Articles")

        sent_filter = st.multiselect(
            "Filter by sentiment",
            options=["positive", "negative", "neutral", "not_scored"],
            default=["positive", "negative", "neutral", "not_scored"],
        )

        df_art = df[df["sentiment_label"].isin(sent_filter)].copy()
        df_art = df_art.sort_values(
            ["relevance", "sentiment_score"], ascending=[False, False]
        )

        for _, row in df_art.iterrows():
            icon_map = {
                "positive": "🟢",
                "negative": "🔴",
                "neutral": "🔵",
                "not_scored": "⚪",
            }
            icon = icon_map.get(row["sentiment_label"], "⚪")

            st.markdown(f"**{icon} {row['title']}**")
            cols = st.columns([3, 2, 2, 1])
            cols[0].caption(
                f"Source: {row['source_domain']} | lang: {row.get('language','unknown')}"
            )
            cols[1].caption(f"Relevance: {row['relevance']:.1f}%")
            if row["sentiment_label"] in ["positive", "negative", "neutral"]:
                cols[2].caption(
                    f"Sentiment conf: {row['sentiment_score']*100:.1f}%"
                )
            else:
                cols[2].caption("Sentiment: not scored")
            cols[3].markdown(f"[Read →]({row['link']})")

            if row.get("summary"):
                st.caption((row["summary"] or "")[:260] + "…")

            st.markdown("---")

    # ----- Keywords tab -----
    with tab_keywords:
        st.subheader("Trending Keywords (titles)")
        keywords = extract_top_keywords(df["title"].tolist(), n=20)
        if keywords:
            kw_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])
            st.dataframe(kw_df, use_container_width=True)
            fig_kw = px.bar(
                kw_df,
                x="Frequency",
                y="Keyword",
                orientation="h",
            )
            fig_kw.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("Not enough data to extract keywords.")

    # ----- Download tab -----
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
        ].copy()

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
