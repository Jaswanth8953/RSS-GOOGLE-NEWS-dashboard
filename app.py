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
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from gnews import GNews
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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

MIN_SIMILARITY = 0.30          # a bit lower → more matches
MAX_ARTICLES_DEFAULT = 200
GDELT_MAX_RECORDS = 250
GNEWS_MAX_RESULTS = 200

# Fix: Initialize langdetect for consistent results
DetectorFactory.seed = 0


# ========================== DB SETUP (LOCK-FREE) ==========================

def get_connection():
    """
    Single shared SQLite connection.
    WAL mode + timeout to avoid 'database is locked'.
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
            precomputed_sentiment_score REAL
        );
        """
    )
    conn.commit()
    return conn


conn = get_connection()


def safe_execute(query: str, params: tuple = ()):
    """
    Safer wrapper for INSERT/UPDATE with retry if DB is locked.
    """
    max_retries = 10
    for attempt in range(max_retries):
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
        raise ValueError(
            "OpenAI API key not found. "
            "Set it in Streamlit secrets or as environment variable OPENAI_API_KEY."
        )

    return OpenAI(api_key=api_key)


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
    Run FinBERT and apply a small calibration:
    - If max probability < 0.55 → treat as neutral.
    This helps avoid "everything is positive".
    Only run on English texts.
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
        # calibration - more aggressive to avoid extremes
        if max_score < 0.60:  # Increased from 0.55
            label = "neutral"
            score = max_score * 0.8  # Reduce confidence for neutral
        else:
            label = base_label
            # Scale down extreme confidences
            if max_score > 0.90:
                score = max_score * 0.9
            else:
                score = max_score
        results.append({"label": label, "score": score})
    return results


# ========================== LANGUAGE DETECTION ==========================

def detect_article_language(title: str, summary: str) -> str:
    """
    Detect language of article content.
    Returns language code like 'en', 'es', 'fr', etc.
    """
    text = f"{title} {summary}"[:500]  # Limit text for speed
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"
    except Exception:
        return "unknown"


def filter_english_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to keep only English articles for reliable FinBERT analysis.
    """
    if df.empty:
        return df
    
    english_rows = []
    for _, row in df.iterrows():
        # Check if language already detected
        if row.get('language') == 'en':
            english_rows.append(row)
        else:
            # Detect language
            lang = detect_article_language(str(row.get('title', '')), 
                                          str(row.get('summary', '')))
            if lang == 'en':
                # Update DB with detected language
                try:
                    safe_execute(
                        "UPDATE articles SET language = ? WHERE id = ?",
                        (lang, int(row['id']))
                    )
                except:
                    pass
                english_rows.append(row)
    
    if not english_rows:
        return pd.DataFrame()
    
    return pd.DataFrame(english_rows)


# ========================== EMBEDDINGS ==========================

def get_embedding(text: str) -> List[float]:
    client = get_openai_client()
    # short, cleaned text for cheaper, faster embeddings
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
                # Detect language during ingestion
                lang = detect_article_language(art["title"], art["summary"])
                
                safe_execute(
                    """
                    INSERT OR IGNORE INTO articles
                    (title, summary, published, link, source, content, embedding, language)
                    VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                    """,
                    (
                        art["title"],
                        art["summary"],
                        art["published"],
                        art["link"],
                        art["source"],
                        art["content"],
                        lang,
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
        
        # Detect language
        lang = detect_article_language(title, content)

        try:
            safe_execute(
                """
                INSERT OR IGNORE INTO articles
                (title, summary, published, link, source, content, embedding, language)
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (title, summary, published, link, src, content, lang),
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
        
        # Detect language (mostly English from GNews)
        lang = detect_article_language(title, desc)

        try:
            safe_execute(
                """
                INSERT OR IGNORE INTO articles
                (title, summary, published, link, source, content, embedding, language)
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (title, desc, published, link, source, content, lang),
            )
            new_count += 1
        except Exception:
            pass

    return new_count


# ========================== PRE-COMPUTATION FUNCTIONS ==========================

def precompute_sentiment_for_new_articles():
    """
    Background function to precompute sentiment for new articles.
    This should be called periodically, not at query time.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary, content 
        FROM articles 
        WHERE precomputed_sentiment_label IS NULL 
        AND language = 'en'
        LIMIT 100
        """
    )
    rows = cur.fetchall()
    
    if not rows:
        return 0
    
    articles_data = []
    for row in rows:
        articles_data.append({
            'id': row[0],
            'text': f"{row[1]} {row[2]}"[:1000]  # Combine title + summary
        })
    
    texts = [a['text'] for a in articles_data]
    if not texts:
        return 0
    
    # Run FinBERT in batch
    sentiments = finbert_sentiment(texts)
    
    # Update database
    updated = 0
    for i, sentiment in enumerate(sentiments):
        try:
            safe_execute(
                """
                UPDATE articles 
                SET precomputed_sentiment_label = ?, precomputed_sentiment_score = ?
                WHERE id = ?
                """,
                (sentiment['label'], sentiment['score'], articles_data[i]['id'])
            )
            updated += 1
        except:
            continue
    
    return updated


def precompute_embeddings_for_new_articles():
    """
    Background function to precompute embeddings for new articles.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary 
        FROM articles 
        WHERE embedding IS NULL 
        LIMIT 50
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
                "UPDATE articles SET embedding = ? WHERE id = ?",
                (json.dumps(emb), row[0])
            )
            updated += 1
        except Exception as e:
            continue
    
    return updated


# ========================== EMBEDDING MANAGEMENT ==========================

def load_articles_for_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary, published, link, source, content, 
               embedding, language, precomputed_sentiment_label, 
               precomputed_sentiment_score
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
        if row["embedding"] is None:
            text = (row["title"] or "") + " " + (row["summary"] or "")
            if not text.strip():
                continue
            emb = get_embedding(text)
            safe_execute(
                "UPDATE articles SET embedding = ? WHERE id = ?",
                (json.dumps(emb), int(row["id"])),
            )


def hybrid_search(
    query: str,
    start: dt.date,
    end: dt.date,
    top_k: int,
    min_sim: float = MIN_SIMILARITY,
) -> pd.DataFrame:
    """
    Hybrid keyword + semantic search within date range.
    Also returns 'match_type' = keyword / semantic / keyword+semantic.
    """
    df = load_articles_for_range(start, end)
    if df.empty:
        return df

    df["embedding"] = df["embedding"].apply(
        lambda x: None if x in (None, "", "NULL") else x
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
        emb_vec = np.array(json.loads(row["embedding"]))
        sims.append(cosine_similarity(q_emb, emb_vec))
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
        mask = kw_mask & sem_mask
        df["match_type"] = np.where(
            mask,
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
        )
        content = resp.choices[0].message.content.strip()
        expansions = json.loads(content)
        expansions = [str(t) for t in expansions if isinstance(t, str)]
        # always include original query as first term
        all_terms = [query] + expansions
        # deduplicate while preserving order
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
        return cleaned
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

    # cap to avoid huge prompts (AWS cost + speed)
    df_small = df.head(200).copy()

    # Prepare lightweight list
    items = [
        {
            "id": int(row["id"]),
            "title": row["title"],
            "summary": (row["summary"] or "")[:250],
            "similarity": float(row.get("similarity", 0)),
        }
        for _, row in df_small.iterrows()
    ]

    prompt = f"""
You are an article relevance filter for a financial news dashboard.

The user topic is: "{query}"

Below is a list of articles with IDs, titles, summaries, and similarity scores (0-1).

Select articles that are TRULY RELEVANT to this topic.
Consider both semantic similarity and topical relevance.

Return ONLY a JSON list of the IDs to KEEP, sorted by relevance.
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
            max_tokens=500,
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
    FIXED: Calculate sentiment index without always hitting ±100%
    Uses weighted average approach that includes neutrals
    """
    if df.empty:
        return 0.0
    
    # Use precomputed sentiment if available
    if 'precomputed_sentiment_label' in df.columns and not df['precomputed_sentiment_label'].isna().all():
        pos = len(df[df["precomputed_sentiment_label"] == "positive"])
        neg = len(df[df["precomputed_sentiment_label"] == "negative"])
        neu = len(df[df["precomputed_sentiment_label"] == "neutral"])
    else:
        pos = len(df[df["sentiment_label"] == "positive"])
        neg = len(df[df["sentiment_label"] == "negative"])
        neu = len(df[df["sentiment_label"] == "neutral"])
    
    total = len(df)
    
    if total == 0:
        return 0.0
    
    # METHOD 1: Weighted average (prevents extremes)
    # positive = +1, negative = -1, neutral = 0
    weighted_sum = (pos * 1.0) + (neg * -1.0) + (neu * 0.0)
    index = (weighted_sum / total) * 100
    
    # METHOD 2: Alternative - only positive vs negative (ignores neutrals)
    # if (pos + neg) > 0:
    #     index = ((pos - neg) / (pos + neg)) * 100
    # else:
    #     index = 0.0
    
    return round(index, 2)


def calculate_composite_score(df: pd.DataFrame, query: str) -> dict:
    """
    Calculate comprehensive composite score with weighting.
    Addresses Professor's concern about different regions/countries.
    """
    if df.empty:
        return {
            "sentiment_index": 0.0,
            "confidence": 0.0,
            "source_diversity": 0.0,
            "regional_bias": 0.0
        }
    
    # 1. Weight by similarity/relevance
    if 'similarity' in df.columns:
        weights = df['similarity'].clip(lower=0.1).values
    else:
        weights = np.ones(len(df))
    
    # 2. Weight by source reliability (simple heuristic)
    source_weights = {
        'reuters': 1.2,
        'bbc': 1.2,
        'cnbc': 1.1,
        'marketwatch': 1.0,
        'economictimes': 1.0,
        'aljazeera': 0.9,
        'gdelt': 0.8,
        'google-news': 1.0
    }
    
    df['source_weight'] = df['source'].apply(
        lambda x: next((v for k, v in source_weights.items() if k in str(x).lower()), 1.0)
    )
    weights = weights * df['source_weight'].values
    
    # 3. Calculate weighted sentiment
    sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    
    if 'precomputed_sentiment_label' in df.columns and not df['precomputed_sentiment_label'].isna().all():
        sentiment_values = df['precomputed_sentiment_label'].map(sentiment_map).fillna(0).values
        sentiment_scores = df['precomputed_sentiment_score'].fillna(0.5).values
    else:
        sentiment_values = df['sentiment_label'].map(sentiment_map).fillna(0).values
        sentiment_scores = df['sentiment_score'].fillna(0.5).values
    
    # Apply confidence weighting
    weighted_sentiments = sentiment_values * sentiment_scores * weights
    
    if weights.sum() > 0:
        composite = (weighted_sentiments.sum() / weights.sum()) * 100
    else:
        composite = 0.0
    
    # 4. Calculate source diversity
    unique_sources = df['source'].nunique()
    source_diversity = min(100, (unique_sources / len(df)) * 100) if len(df) > 0 else 0
    
    # 5. Detect regional bias
    query_lower = query.lower()
    us_keywords = ['us', 'usa', 'united states', 'american', 'wall street']
    eu_keywords = ['eu', 'europe', 'european', 'germany', 'france', 'uk', 'british']
    asia_keywords = ['china', 'asian', 'japan', 'india', 'asia']
    
    region_counts = {
        'us': sum(any(kw in str(text).lower() for kw in us_keywords) 
                 for text in df['title'].fillna('') + ' ' + df['summary'].fillna('')),
        'eu': sum(any(kw in str(text).lower() for kw in eu_keywords) 
                 for text in df['title'].fillna('') + ' ' + df['summary'].fillna('')),
        'asia': sum(any(kw in str(text).lower() for kw in asia_keywords) 
                   for text in df['title'].fillna('') + ' ' + df['summary'].fillna(''))
    }
    
    total_region_mentions = sum(region_counts.values())
    regional_bias = 0.0
    if total_region_mentions > 0:
        max_region = max(region_counts.values())
        regional_bias = (max_region / total_region_mentions) * 100
    
    return {
        "sentiment_index": round(composite, 2),
        "confidence": round(sentiment_scores.mean() * 100, 1),
        "source_diversity": round(source_diversity, 1),
        "regional_bias": round(regional_bias, 1)
    }


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

        # Background precomputation button
        if st.button("Run background precomputation"):
            with st.spinner("Precomputing embeddings and sentiment…"):
                emb_count = precompute_embeddings_for_new_articles()
                sent_count = precompute_sentiment_for_new_articles()
            st.success(f"Precomputed: {emb_count} embeddings, {sent_count} sentiments")

        today = dt.date.today()
        default_start = today - dt.timedelta(days=30)
        start_date = st.date_input("Start date", default_start)
        end_date = st.date_input("End date", today)

        st.subheader("Extra sources for this topic")
        use_gdelt = st.checkbox("Include GDELT", True)
        use_gnews = st.checkbox("Include Google News", True)
        
        st.subheader("Analysis Options")
        use_precomputed = st.checkbox("Use precomputed sentiment (faster)", True)
        filter_english = st.checkbox("Filter English articles only", True)

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

    # Filter English articles if selected
    if filter_english:
        with st.spinner("Filtering English articles for accurate sentiment analysis…"):
            df_before = len(df)
            df = filter_english_articles(df)
            df_after = len(df)
            if df_after < df_before:
                st.info(f"Filtered out {df_before - df_after} non-English articles.")

    # Optional LLM article filter / selector
    with st.spinner("Letting LLM choose the most relevant subset of articles…"):
        df = llm_select_articles(query, df)

    if df.empty:
        st.error("After LLM filtering, no strongly relevant articles remained.")
        return

    st.success(f"✅ LLM selected **{len(df)}** final articles for analysis.")

    # SENTIMENT ANALYSIS
    with st.spinner("Running sentiment analysis…"):
        # Use precomputed sentiment if available and selected
        if (use_precomputed and 'precomputed_sentiment_label' in df.columns 
            and not df['precomputed_sentiment_label'].isna().all()):
            
            df["sentiment_label"] = df["precomputed_sentiment_label"]
            df["sentiment_score"] = df["precomputed_sentiment_score"]
            st.info("Using precomputed sentiment scores (faster)")
            
        else:
            # Run FinBERT on the fly
            texts = df["content"].fillna(df["summary"]).tolist()
            sents = finbert_sentiment(texts)
            df["sentiment_label"] = [s["label"] for s in sents]
            df["sentiment_score"] = [s["score"] for s in sents]
            st.info("Computed sentiment in real-time")

    df["relevance"] = (df["similarity"] * 100).round(1)
    df["source_domain"] = df["source"].fillna("").replace("", "unknown")

    # Calculate multiple sentiment metrics
    sentiment_index = calculate_sentiment_index(df)
    composite_scores = calculate_composite_score(df, query)

    # TABS
    tab_dash, tab_articles, tab_keywords, tab_download, tab_advanced = st.tabs(
        ["📊 Dashboard", "📰 Articles", "🔑 Keywords", "📥 Download", "📈 Advanced Analytics"]
    )

    # DASHBOARD TAB
    with tab_dash:
        st.subheader("Executive Sentiment Overview")

        counts = df["sentiment_label"].value_counts()
        total = int(counts.sum())
        pos = int(counts.get("positive", 0))
        neg = int(counts.get("negative", 0))
        neu = int(counts.get("neutral", 0))

        c_a, c_b, c_c, c_d, c_e, c_f = st.columns(6)
        c_a.metric("Total Articles", total)
        c_b.metric("Positive", pos, f"{(pos/total*100):.1f}%" if total else "0.0%")
        c_c.metric("Negative", neg, f"{(neg/total*100):.1f}%" if total else "0.0%")
        c_d.metric("Neutral", neu, f"{(neu/total*100):.1f}%" if total else "0.0%")

        direction = (
            "Bullish" if sentiment_index > 0 else "Bearish" if sentiment_index < 0 else "Neutral"
        )
        c_e.metric("Sentiment Index", f"{sentiment_index:.1f}", direction)
        
        # Show composite confidence
        confidence_level = "High" if composite_scores["confidence"] > 70 else "Medium" if composite_scores["confidence"] > 50 else "Low"
        c_f.metric("Analysis Confidence", f"{composite_scores['confidence']:.1f}%", confidence_level)

        st.markdown("---")

        # Sentiment distribution with advanced metrics
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown("#### Sentiment Distribution")
            sentiment_count = df["sentiment_label"].value_counts()
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

        # Composite gauge chart
        with r1c2:
            st.markdown("#### Composite Sentiment Gauge")
            gauge_color = "#2ecc71" if composite_scores["sentiment_index"] > 0 else "#e74c3c"
            fig_g = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=composite_scores["sentiment_index"],
                    title={"text": "Weighted Sentiment Index"},
                    delta={"reference": 0},
                    gauge={
                        "axis": {"range": [-100, 100]},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [-100, -30], "color": "rgba(231, 76, 60, 0.4)"},
                            {"range": [-30, 0], "color": "rgba(231, 76, 60, 0.2)"},
                            {"range": [0, 30], "color": "rgba(46, 204, 113, 0.2)"},
                            {"range": [30, 100], "color": "rgba(46, 204, 113, 0.4)"},
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

        # Source diversity and language info
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("#### Articles by Source")
            src_counts = df["source_domain"].value_counts().head(10)
            fig_src = px.bar(
                x=src_counts.values,
                y=src_counts.index,
                orientation="h",
                labels={"x": "Count", "y": "Source"},
                color=src_counts.values,
                color_continuous_scale="Viridis",
                title=f"Source Diversity: {composite_scores['source_diversity']:.1f}%"
            )
            st.plotly_chart(fig_src, use_container_width=True)

        with r2c2:
            st.markdown("#### Average Confidence by Sentiment")
            avg_conf = (df.groupby("sentiment_label")["sentiment_score"].mean() * 100).round(1)
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
        
        # Advanced filtering
        col1, col2, col3 = st.columns(3)
        with col1:
            sent_filter = st.multiselect(
                "Filter by sentiment",
                options=["positive", "negative", "neutral"],
                default=["positive", "negative", "neutral"],
            )
        with col2:
            min_relevance = st.slider("Min relevance (%)", 0, 100, 0)
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                options=["Relevance", "Sentiment Confidence", "Date"],
                index=0
            )
        
        filtered = df[df["sentiment_label"].isin(sent_filter)].copy()
        filtered = filtered[filtered["relevance"] >= min_relevance]
        
        if sort_by == "Relevance":
            filtered = filtered.sort_values("relevance", ascending=False)
        elif sort_by == "Sentiment Confidence":
            filtered = filtered.sort_values("sentiment_score", ascending=False)
        else:  # Date
            filtered = filtered.sort_values("published", ascending=False)
        
        for _, row in filtered.iterrows():
            icon = {"positive": "🟢", "negative": "🔴", "neutral": "🔵"}[row["sentiment_label"]]
            st.markdown(f"**{icon} {row['title']}**")
            c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
            c1.caption(f"📰 {row['source_domain']}")
            c2.caption(f"🎯 Relevance: {row['relevance']:.1f}%")
            c3.caption(f"😊 Sentiment: {row['sentiment_label']}")
            c4.caption(f"⚡ Confidence: {row['sentiment_score']*100:.1f}%")
            c5.markdown(f"[Read →]({row['link']})")
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
                    title="Top Keywords"
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
                    title="Keyword Treemap"
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
                "sentiment_label",
                "sentiment_score",
                "relevance",
                "match_type",
                "query_term",
                "language",
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
    
    # ADVANCED ANALYTICS TAB
    with tab_advanced:
        st.subheader("Advanced Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Composite Sentiment", f"{composite_scores['sentiment_index']:.1f}")
        with col2:
            st.metric("Analysis Confidence", f"{composite_scores['confidence']:.1f}%")
        with col3:
            st.metric("Source Diversity", f"{composite_scores['source_diversity']:.1f}%")
        with col4:
            st.metric("Regional Bias", f"{composite_scores['regional_bias']:.1f}%")
        
        st.markdown("---")
        
        # Time series sentiment
        if len(df) > 1:
            st.markdown("#### Sentiment Over Time")
            df['published_date'] = pd.to_datetime(df['published']).dt.date
            time_series = df.groupby('published_date').apply(
                lambda x: calculate_sentiment_index(x)
            ).reset_index(name='sentiment_index')
            
            fig_time = px.line(
                time_series,
                x='published_date',
                y='sentiment_index',
                title='Daily Sentiment Trend',
                markers=True
            )
            fig_time.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Source reliability analysis
        st.markdown("#### Source Reliability Analysis")
        source_stats = df.groupby('source_domain').agg({
            'sentiment_score': 'mean',
            'id': 'count',
            'relevance': 'mean'
        }).round(2).reset_index()
        source_stats.columns = ['Source', 'Avg Confidence', 'Article Count', 'Avg Relevance']
        source_stats = source_stats.sort_values('Avg Confidence', ascending=False)
        st.dataframe(source_stats, use_container_width=True)


if __name__ == "__main__":
    run_app()
