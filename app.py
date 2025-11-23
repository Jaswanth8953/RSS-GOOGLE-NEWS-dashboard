import os
import json
import sqlite3
import time
import datetime as dt
from typing import List, Tuple, Dict, Any
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

# ========================== CONFIGURATION ==========================

DB_PATH = "news_articles.db"

# Comprehensive RSS feeds for better coverage
RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "https://www.marketwatch.com/marketwatch/rss/topstories",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://feeds.reuters.com/reuters/companiesNews",
    "https://rss.bloomberg.com/markets/news.rss",
    "https://www.wsj.com/news/markets?mod=rss_markets_main",
]

OPENAI_EMBED_MODEL = "text-embedding-3-small"
FINBERT_MODEL = "ProsusAI/finbert"

# Tuned parameters
MIN_SIMILARITY = 0.18  # Lower threshold for broader matching
MAX_ARTICLES_DEFAULT = 500
GDELT_MAX_RECORDS = 150
GNEWS_MAX_RESULTS = 150

# ========================== PREMIUM STYLING ==========================

def apply_premium_styling():
    """Apply luxury styling to the dashboard"""
    st.markdown("""
    <style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Premium headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #FFD700, #FFA500, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    /* Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 215, 0, 0.2);
        box-shadow: 0 8px 32px rgba(255, 215, 0, 0.1);
        margin: 10px 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #000000 !important;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 215, 0, 0.3);
        border-radius: 15px;
        color: white;
        padding: 12px 20px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        border: 1px solid rgba(255, 215, 0, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%) !important;
        color: #000000 !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #FFD700, #FFA500);
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ========================== DATABASE SETUP ==========================

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
            embedding TEXT
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

# ========================== OPENAI SETUP ==========================

@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client"""
    api_key = st.secrets["openai"]["api_key"]
    return OpenAI(api_key=api_key)

# ========================== FIXED FINBERT SENTIMENT ==========================

@st.cache_resource
def load_finbert():
    try:
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Failed to load FinBERT: {e}")
        return None, None, None

def finbert_sentiment(texts: List[str]) -> List[dict]:
    if not texts:
        return []
    
    tokenizer, model, device = load_finbert()
    
    # Fallback if model didn't load properly
    if tokenizer is None or model is None:
        return [{"label": "neutral", "score": 1.0}] * len(texts)
    
    # Better text preprocessing for financial news
    processed_texts = []
    for text in texts:
        if not text or pd.isna(text):
            processed_texts.append("")
            continue
        # Clean the text - remove extra spaces, truncate properly
        clean_text = ' '.join(str(text).split())[:512]  # Limit length
        processed_texts.append(clean_text)
    
    # Remove empty texts
    valid_texts = [t for t in processed_texts if t.strip()]
    if not valid_texts:
        return [{"label": "neutral", "score": 1.0}] * len(texts)
    
    try:
        enc = tokenizer(
            valid_texts,
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
            results.append({"label": id2label[idx], "score": float(p[idx])})
        
        # Handle any texts that were filtered out
        final_results = []
        result_idx = 0
        for i, text in enumerate(processed_texts):
            if text.strip():  # Valid text
                final_results.append(results[result_idx])
                result_idx += 1
            else:  # Empty text - default to neutral
                final_results.append({"label": "neutral", "score": 1.0})
                
        return final_results
        
    except Exception as e:
        # Fallback: return all neutral
        return [{"label": "neutral", "score": 1.0}] * len(texts)

# ========================== EMBEDDINGS ==========================

def get_embedding(text: str) -> List[float]:
    client = get_openai_client()
    text = text.replace("\n", " ")
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
        title = entry.get("title", "").strip()
        summary = entry.get("summary", "").strip()
        link = entry.get("link", "")

        if hasattr(entry, "published_parsed") and entry.published_parsed:
            pub = dt.datetime(*entry.published_parsed[:6])
        else:
            pub = dt.datetime.utcnow()
        published = pub.isoformat()

        # Better content extraction
        content = ""
        if summary and len(summary) > 50:  # Only use substantial summaries
            content = summary
        elif "content" in entry and entry.content:
            # Try to get the longest content available
            contents = [c.value for c in entry.content if hasattr(c, 'value')]
            if contents:
                content = max(contents, key=len)
        
        # If still no good content, use title
        if not content or len(content) < 20:
            content = title

        articles.append({
            "title": title,
            "summary": summary,
            "published": published,
            "link": link,
            "source": url,
            "content": content,
        })
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
                safe_execute(
                    """
                    INSERT OR IGNORE INTO articles
                    (title, summary, published, link, source, content, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        art["title"],
                        art["summary"],
                        art["published"],
                        art["link"],
                        art["source"],
                        art["content"],
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
        r = requests.get(base_url, timeout=15)
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

        try:
            safe_execute(
                """
                INSERT OR IGNORE INTO articles
                (title, summary, published, link, source, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (title, summary, published, link, src, content),
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

        try:
            safe_execute(
                """
                INSERT OR IGNORE INTO articles
                (title, summary, published, link, source, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (title, desc, published, link, source, content),
            )
            new_count += 1
        except Exception:
            pass

    return new_count

# ========================== EMBEDDING MANAGEMENT ==========================

def load_articles_for_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary, published, link, source, content, embedding
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

# ========================== SEMANTIC SEARCH MULTI ==========================

def semantic_search_multi(
    queries: List[str],
    start: dt.date,
    end: dt.date,
    top_k: int,
    min_sim: float = MIN_SIMILARITY,
) -> pd.DataFrame:
    """
    Semantic-first hybrid search using ALL expanded queries together.
    """
    df = load_articles_for_range(start, end)
    if df.empty:
        return df

    # Normalise embedding column
    df["embedding"] = df["embedding"].apply(
        lambda x: None if x in (None, "", "NULL") else x
    )

    # Ensure every article has an embedding
    ensure_embeddings(df[["id", "title", "summary", "embedding"]])

    # Reload to get fresh embeddings
    df = load_articles_for_range(start, end)
    if df.empty:
        return df

    # Build article embedding matrix
    emb_list = []
    valid_idx = []
    for idx, row in df.iterrows():
        try:
            vec = np.array(json.loads(row["embedding"]), dtype=float)
            emb_list.append(vec)
            valid_idx.append(idx)
        except Exception:
            emb_list.append(None)

    if not emb_list:
        return pd.DataFrame()

    emb_mat = np.vstack([e for e in emb_list if e is not None])
    article_norms = np.linalg.norm(emb_mat, axis=1)
    article_norms[article_norms == 0] = 1e-8

    # Build combined query embedding (average of all expanded queries)
    clean_queries = [q.strip() for q in queries if q and q.strip()]
    if not clean_queries:
        return pd.DataFrame()

    q_vecs = []
    for q in clean_queries:
        q_vecs.append(np.array(get_embedding(q), dtype=float))
    q_mat = np.vstack(q_vecs)
    q_mean = q_mat.mean(axis=0)
    q_norm = np.linalg.norm(q_mean)
    if q_norm == 0:
        q_norm = 1e-8

    # Cosine similarities
    sims = (emb_mat @ q_mean) / (article_norms * q_norm)

    # Put similarities back into df
    df = df.iloc[valid_idx].copy()
    df["similarity"] = sims

    # Keyword mask over ALL queries
    q_terms = [q.lower() for q in clean_queries]

    def contains_any(text: str) -> bool:
        t = (text or "").lower()
        return any(term in t for term in q_terms)

    kw_mask = (
        df["title"].apply(contains_any)
        | df["summary"].apply(contains_any)
        | df["content"].apply(contains_any)
    )

    # Semantic mask
    sem_mask = df["similarity"] >= min_sim

    # Give a small boost to articles that also match keywords
    df.loc[kw_mask, "similarity"] = df.loc[kw_mask, "similarity"] + 0.05

    # Decide which rows to keep
    base_mask = sem_mask | kw_mask
    if base_mask.any():
        filtered = df[base_mask].copy()
        filtered["match_type"] = np.where(
            kw_mask[base_mask] & sem_mask[base_mask],
            "keyword+semantic",
            np.where(kw_mask[base_mask], "keyword", "semantic"),
        )
    else:
        # Fallback: just take best semantic matches
        filtered = df.copy()
        filtered["match_type"] = "semantic"

    filtered = filtered.sort_values("similarity", ascending=False).head(top_k)
    return filtered

# ========================== LLM QUERY EXPANSION ==========================

def expand_query_with_llm(query: str) -> List[str]:
    """
    Use LLM to generate related terms for the user's query.
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
        return [query] + expansions
    except Exception:
        return [query]

# ========================== LLM ARTICLE SELECTION ==========================

def llm_select_articles(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Let the LLM decide which of the already-matched articles
    are truly relevant to the user's topic.
    """
    if df.empty:
        return df

    client = get_openai_client()

    # Prepare lightweight list
    items = [
        {
            "id": int(row["id"]),
            "title": row["title"],
            "summary": (row["summary"] or "")[:250],
        }
        for _, row in df.iterrows()
    ]

    prompt = f"""
You are an article relevance filter for a financial news dashboard.

The user topic is: "{query}"

Below is a list of articles with IDs, titles, and summaries.

Decide which articles are truly relevant to this topic.
Return ONLY a JSON list of the IDs to KEEP.
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
        )
        content = resp.choices[0].message.content.strip()
        selected_ids = json.loads(content)
        if not isinstance(selected_ids, list):
            return df
        selected_ids = set(int(i) for i in selected_ids)
        return df[df["id"].isin(selected_ids)]
    except Exception:
        # if anything goes wrong, just return original df
        return df

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
    pos = len(df[df["sentiment_label"] == "positive"])
    neg = len(df[df["sentiment_label"] == "negative"])
    total = len(df)
    if total == 0:
        return 0.0
    index = ((pos - neg) / total) * 100
    return round(index, 2)

# ========================== PREMIUM VISUALIZATIONS ==========================

def create_luxury_gauge(value: float, title: str):
    """Create premium gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': 'white', 'family': "Arial Black"}},
        delta={'reference': 0, 'increasing': {'color': "#00FF87"}, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "gold", 'tickfont': {'color': 'white', 'size': 12}},
            'bar': {'color': "linear-gradient(90deg, #FF6B6B, #FFD93D, #00FF87)"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gold",
            'steps': [
                {'range': [-100, -33], 'color': 'rgba(255, 107, 107, 0.3)'},
                {'range': [-33, 33], 'color': 'rgba(255, 217, 61, 0.3)'},
                {'range': [33, 100], 'color': 'rgba(0, 255, 135, 0.3)'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value}}))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"},
        margin=dict(t=60, b=10, l=10, r=10)
    )
    return fig

def create_premium_pie_chart(positive: int, negative: int, neutral: int):
    """Create luxury pie chart"""
    colors = ['#00FF87', '#FF6B6B', '#FFD93D']
    fig = px.pie(
        values=[positive, negative, neutral],
        names=['Positive', 'Negative', 'Neutral'],
        color=['Positive', 'Negative', 'Neutral'],
        color_discrete_map={'Positive': colors[0], 'Negative': colors[1], 'Neutral': colors[2]},
        hole=0.6
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='gold', width=2)),
        hoverinfo='label+percent+value'
    )
    
    fig.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'size': 14, 'family': "Arial"},
        margin=dict(t=40, b=40, l=40, r=40),
        height=350
    )
    
    return fig

# ========================== PREMIUM DASHBOARD COMPONENTS ==========================

def create_metric_card(value, label, delta=None, delta_color="normal"):
    """Create premium metric card"""
    delta_color_map = {
        "normal": "",
        "inverse": "color: #FF6B6B" if float(delta or 0) < 0 else "color: #00FF87"
    }
    
    card_html = f"""
    <div class="metric-card">
        <div style="font-size: 2.5rem; font-weight: 800; background: linear-gradient(90deg, #FFD700, #FFA500); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            {value}
        </div>
        <div style="font-size: 1rem; color: #CCCCCC; margin-bottom: 8px;">{label}</div>
        {f'<div style="font-size: 0.9rem; {delta_color_map[delta_color]}">{"+" if float(delta or 0) > 0 else ""}{delta}</div>' if delta else ''}
    </div>
    """
    return card_html

# ========================== STREAMLIT APP ==========================

def main():
    st.set_page_config(
        page_title="Quantum Sentiment Pro",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🚀"
    )
    
    apply_premium_styling()
    
    # Premium Header
    col1, col2, col3 = st.columns([2, 3, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem; background: linear-gradient(90deg, #FFD700, #FFA500, #FF6B6B); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                QUANTUM SENTIMENT PRO
            </h1>
            <p style="font-size: 1.2rem; color: #CCCCCC; font-style: italic;">
                Enterprise-Grade Financial Intelligence Platform
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar - Luxury Design
    with st.sidebar:
        st.markdown("""
        <div style="padding: 2rem 1rem; text-align: center;">
            <h2 style="background: linear-gradient(90deg, #FFD700, #FFA500); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                QUANTUM CONTROL
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Data Management Section
        st.markdown("### 📊 DATA MANAGEMENT")
        if st.button("🔄 SYNC MARKET DATA", use_container_width=True, type="primary"):
            with st.spinner("🔄 Synchronizing global market feeds..."):
                new_count = fetch_rss_articles()
                if new_count > 0:
                    st.success(f"✅ Synced {new_count} premium articles")
                else:
                    st.info("💎 Database already synchronized")
        
        st.markdown("---")
        
        # Search Configuration
        st.markdown("### 🎯 SEARCH CONFIGURATION")
        today = dt.date.today()
        start_date = st.date_input("📅 ANALYSIS PERIOD START", today - dt.timedelta(days=30))
        end_date = st.date_input("📅 ANALYSIS PERIOD END", today)
        
        st.markdown("### ⚙️ ADVANCED SETTINGS")
        use_external_sources = st.toggle("🌐 ENABLE GLOBAL SOURCES", value=True)
        max_articles = st.slider("📈 MAX ARTICLES ANALYSIS", 100, 1000, 500, 50)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #888; font-size: 0.8rem;">
            <p>QUANTUM SENTIMENT PRO™</p>
            <p>Enterprise Edition</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Search Interface - Luxury Design
    st.markdown("### 🔍 QUANTUM SEARCH ENGINE")
    
    search_col1, search_col2, search_col3 = st.columns([3, 1, 1])
    
    with search_col1:
        query = st.text_input(
            " ",
            placeholder="🔮 Enter market topic (e.g., US Tech, AI Revolution, Quantum Computing...)",
            key="search_query",
            label_visibility="collapsed"
        )
    
    with search_col2:
        search_clicked = st.button("🚀 LAUNCH ANALYSIS", type="primary", use_container_width=True)
    
    with search_col3:
        clear_clicked = st.button("🔄 RESET", use_container_width=True)
    
    if clear_clicked:
        st.rerun()
    
    if not search_clicked or not query.strip():
        st.markdown("""
        <div style="text-align: center; padding: 4rem; background: rgba(255,255,255,0.05); border-radius: 15px; border: 1px solid rgba(255,215,0,0.2);">
            <h3 style="color: #FFD700; margin-bottom: 1rem;">💎 WELCOME TO QUANTUM SENTIMENT PRO</h3>
            <p style="color: #CCCCCC;">Enter a market topic above to unleash the power of AI-driven sentiment intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Analysis Process
    with st.status("🚀 INITIATING QUANTUM ANALYSIS...", expanded=True) as status:
        status.update(label="🔄 Expanding search universe...")
        expanded_queries = expand_query_with_llm(query)
        
        status.update(label="🌐 Accessing global data streams...")
        external_count = 0
        if use_external_sources:
            for search_term in expanded_queries[:8]:
                external_count += fetch_gdelt_articles(search_term, start_date, end_date)
                external_count += fetch_gnews_articles(search_term, start_date, end_date)
        
        status.update(label="🎯 Executing semantic intelligence...")
        results_df = semantic_search_multi(expanded_queries, start_date, end_date, max_articles)
        
        if results_df.empty:
            status.update(label="❌ Analysis complete - No significant data found", state="error")
            st.error("No relevant articles found. Try expanding your search criteria.")
            return
        
        status.update(label="🔍 Applying precision filtering...")
        results_df = llm_select_articles(query, results_df)
        
        status.update(label="😊 Deploying sentiment analysis matrix...")
        # Use both title and content for better sentiment analysis
        texts = []
        for _, row in results_df.iterrows():
            content = row["content"] if pd.notna(row["content"]) else ""
            title = row["title"] if pd.notna(row["title"]) else ""
            # Combine title and first 200 chars of content
            combined_text = f"{title}. {content}"[:300]
            texts.append(combined_text)
        
        sentiment_results = finbert_sentiment(texts)
        
        results_df["sentiment_label"] = [s["label"] for s in sentiment_results]
        results_df["sentiment_score"] = [s["score"] for s in sentiment_results]
        results_df["relevance"] = (results_df["similarity"] * 100).round(1)
        results_df["source_domain"] = results_df["source"].fillna("").replace("", "unknown")
        
        status.update(label=f"✅ QUANTUM ANALYSIS COMPLETE - {len(results_df)} assets processed", state="complete")
    
    # Results Header
    st.success(f"**💎 QUANTUM INTELLIGENCE REPORT: {len(results_df)} Market Signals Processed**")
    
    # Premium Metrics Row
    total = len(results_df)
    sentiment_counts = results_df["sentiment_label"].value_counts()
    positive = sentiment_counts.get("positive", 0)
    negative = sentiment_counts.get("negative", 0)
    neutral = sentiment_counts.get("neutral", 0)
    sentiment_index = calculate_sentiment_index(results_df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(create_metric_card(total, "TOTAL SIGNALS"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card(positive, "BULLISH", f"+{positive/total*100:.1f}%"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card(negative, "BEARISH", f"{negative/total*100:.1f}%", "inverse"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card(neutral, "NEUTRAL", f"{neutral/total*100:.1f}%"), unsafe_allow_html=True)
    with col5:
        direction = "🚀" if sentiment_index > 0 else "📉"
        st.markdown(create_metric_card(f"{sentiment_index:.1f}", f"SENTIMENT INDEX {direction}"), unsafe_allow_html=True)
    
    # Premium Visualization Section
    st.markdown("---")
    st.markdown("### 📊 QUANTUM VISUALIZATION MATRIX")
    
    viz_col1, viz_col2 = st.columns([1, 1])
    
    with viz_col1:
        st.plotly_chart(create_premium_pie_chart(positive, negative, neutral), use_container_width=True)
    
    with viz_col2:
        st.plotly_chart(create_luxury_gauge(sentiment_index, "MARKET SENTIMENT GAUGE"), use_container_width=True)
    
    # Advanced Analytics
    st.markdown("### 🔬 ADVANCED ANALYTICS DASHBOARD")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💎 MARKET INTELLIGENCE", "📈 SIGNAL STREAM", "🔍 DEEP ANALYSIS", "📊 EXPORT HUB"])
    
    with tab1:
        st.subheader("💎 Market Intelligence Overview")
        
        # Create advanced charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution by source
            source_sentiment = results_df.groupby(['source_domain', 'sentiment_label']).size().unstack(fill_value=0)
            if not source_sentiment.empty:
                fig_source = px.bar(
                    source_sentiment,
                    barmode='stack',
                    color_discrete_map={'positive': '#00FF87', 'negative': '#FF6B6B', 'neutral': '#FFD93D'},
                    title="Sentiment Distribution by Source"
                )
                fig_source.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'}
                )
                st.plotly_chart(fig_source, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig_conf = px.histogram(
                results_df,
                x='sentiment_score',
                color='sentiment_label',
                color_discrete_map={'positive': '#00FF87', 'negative': '#FF6B6B', 'neutral': '#FFD93D'},
                title="Sentiment Confidence Distribution"
            )
            fig_conf.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(fig_conf, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Real-time Signal Stream")
        
        # Filter options
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            sentiment_filter = st.multiselect(
                "Filter Signals:",
                options=["positive", "negative", "neutral"],
                default=["positive", "negative", "neutral"],
                key="signal_filter"
            )
        with filter_col2:
            sort_option = st.selectbox(
                "Sort By:",
                options=["Highest Relevance", "Latest", "Strongest Sentiment"],
                key="signal_sort"
            )
        
        # Display articles with premium styling
        filtered_df = results_df[results_df["sentiment_label"].isin(sentiment_filter)].copy()
        
        if sort_option == "Highest Relevance":
            filtered_df = filtered_df.sort_values("relevance", ascending=False)
        elif sort_option == "Latest":
            filtered_df = filtered_df.sort_values("published", ascending=False)
        else:
            filtered_df = filtered_df.sort_values("sentiment_score", ascending=False)
        
        for idx, article in filtered_df.iterrows():
            sentiment_color = {
                "positive": "#00FF87",
                "negative": "#FF6B6B", 
                "neutral": "#FFD93D"
            }[article["sentiment_label"]]
            
            with st.container():
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-left: 4px solid {sentiment_color}; padding: 1.5rem; border-radius: 10px; margin: 10px 0; border: 1px solid rgba(255,215,0,0.1);">
                    <h4 style="color: white; margin-bottom: 0.5rem;">{article['title']}</h4>
                    <div style="display: flex; justify-content: space-between; color: #CCCCCC; font-size: 0.9rem;">
                        <span>🔗 {article['source_domain']}</span>
                        <span>🎯 Relevance: {article['relevance']}%</span>
                        <span>😊 Sentiment: <span style="color: {sentiment_color}">{article['sentiment_label'].upper()}</span></span>
                        <span>📊 Confidence: {article['sentiment_score']*100:.1f}%</span>
                    </div>
                    {f'<p style="color: #AAAAAA; margin-top: 0.5rem;">{article["summary"][:200]}...</p>' if article["summary"] else ''}
                    <div style="margin-top: 0.5rem;">
                        <a href="{article['link']}" target="_blank" style="color: #FFD700; text-decoration: none;">📖 Read Full Analysis →</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("🔍 Deep Market Analysis")
        
        # Keyword analysis
        st.markdown("#### 📊 Trending Market Terminology")
        keywords = extract_top_keywords(results_df["title"].tolist(), n=20)
        if keywords:
            kw_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])
            fig_keywords = px.treemap(
                kw_df,
                path=['Keyword'],
                values='Frequency',
                title="Market Terminology Cloud"
            )
            fig_keywords.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(fig_keywords, use_container_width=True)
        else:
            st.info("No significant keywords found")
    
    with tab4:
        st.subheader("📊 Quantum Export Hub")
        
        # Export data
        export_df = results_df[[
            "title", "summary", "published", "link", "source_domain", 
            "sentiment_label", "sentiment_score", "relevance", "match_type"
        ]].copy()
        
        st.markdown("#### 💾 Export Market Intelligence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                "📥 EXPORT CSV",
                csv_data,
                file_name=f"quantum_sentiment_{query.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = export_df.to_json(orient="records", indent=2)
            st.download_button(
                "📊 EXPORT JSON",
                json_data,
                file_name=f"quantum_sentiment_{query.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown("#### 📈 Data Preview")
        st.dataframe(export_df.head(10), use_container_width=True)

if __name__ == "__main__":
    main()
