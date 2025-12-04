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

# 25+ RSS FEEDS FOR MORE ARTICLES
RSS_FEEDS = [
    # Business & Finance
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "https://www.cnbc.com/id/19854910/device/rss/rss.html",  # Technology
    "https://feeds.reuters.com/reuters/topNews",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://www.marketwatch.com/marketwatch/rss/topstories",
    "https://www.marketwatch.com/marketwatch/rss/tech",
    
    # Technology
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.wired.com/feed/rss",
    "https://arstechnica.com/feed/",
    
    # International
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms",
    
    # US News
    "https://www.npr.org/rss/rss.php?id=1001",  # Business
    "https://www.npr.org/rss/rss.php?id=1019",  # Technology
    
    # European
    "https://www.dw.com/rss/rdf-en-business",
    "https://www.dw.com/rss/rdf-en-science",
    
    # More sources
    "https://www.engadget.com/rss.xml",
    "https://www.zdnet.com/news/rss.xml",
    "https://feeds.feedburner.com/TechCrunch/",
    "https://www.bloomberg.com/technology/rss",
]

OPENAI_EMBED_MODEL = "text-embedding-3-small"
FINBERT_MODEL = "ProsusAI/finbert"

# CONFIG
MIN_SIMILARITY = 0.25
MAX_ARTICLES_DEFAULT = 500  # Increased from 200
GDELT_MAX_RECORDS = 200
GNEWS_MAX_RESULTS = 200

# Fix: Initialize langdetect for consistent results
DetectorFactory.seed = 0


# ========================== DB SETUP ==========================

def get_connection():
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
        timeout=30
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size = -50000;")  # Increased cache

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
            processed INTEGER DEFAULT 0
        );
        """
    )
    
    # Create index for faster searches
    conn.execute("CREATE INDEX IF NOT EXISTS idx_published ON articles(published);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_language ON articles(language);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_processed ON articles(processed);")
    
    conn.commit()
    return conn

conn = get_connection()


def safe_execute(query: str, params: tuple = ()):
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
    api_key = None
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        api_key = st.secrets["openai"]["api_key"]

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OpenAI API key not found. "
            "Set it in Streamlit secrets or as environment variable OPENAI_API_KEY."
        )

    return OpenAI(api_key=api_key)


# ========================== SENTIMENT ANALYSIS ==========================

@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def finbert_sentiment(texts: List[str]) -> List[dict]:
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
        
        # Balanced calibration
        if max_score < 0.55:
            label = "neutral"
            score = 0.5
        else:
            label = base_label
            # Scale down extreme confidences
            if max_score > 0.85:
                score = max_score * 0.8
            elif max_score > 0.75:
                score = max_score * 0.85
            else:
                score = max_score
        
        results.append({"label": label, "score": score})
    
    return results


# ========================== BULK RSS FETCHER ==========================

def fetch_all_rss_feeds():
    """Fetch ALL RSS feeds to build large database"""
    total_added = 0
    
    with st.spinner(f"Fetching from {len(RSS_FEEDS)} sources... This may take 1-2 minutes"):
        progress_bar = st.progress(0)
        
        for i, url in enumerate(RSS_FEEDS):
            try:
                feed = feedparser.parse(url)
                articles_added = 0
                
                for entry in feed.entries[:50]:  # Get up to 50 per feed
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
                    
                    # Detect language
                    lang = detect_article_language(title, summary)
                    
                    try:
                        safe_execute(
                            """
                            INSERT OR IGNORE INTO articles
                            (title, summary, published, link, source, content, embedding, language, processed)
                            VALUES (?, ?, ?, ?, ?, ?, NULL, ?, 0)
                            """,
                            (title, summary, published, link, url, content, lang),
                        )
                        articles_added += 1
                        total_added += 1
                    except Exception:
                        continue
                
                progress_bar.progress((i + 1) / len(RSS_FEEDS))
                
            except Exception as e:
                continue
    
    return total_added


def detect_article_language(title: str, summary: str) -> str:
    text = f"{title} {summary}"[:500]
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"
    except Exception:
        return "unknown"


def filter_english_articles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    english_rows = []
    for _, row in df.iterrows():
        if row.get('language') == 'en':
            english_rows.append(row)
        else:
            lang = detect_article_language(str(row.get('title', '')), 
                                          str(row.get('summary', '')))
            if lang == 'en':
                english_rows.append(row)
    
    if not english_rows:
        return pd.DataFrame()
    
    return pd.DataFrame(english_rows)


# ========================== EMBEDDINGS ==========================

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


# ========================== HYBRID SEARCH WITH PRECISION MODE ==========================

def load_articles_for_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, title, summary, published, link, source, content, embedding, language
        FROM articles
        WHERE date(published) BETWEEN ? AND ?
        AND language = 'en'
        """,
        (start.isoformat(), end.isoformat()),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "id", "title", "summary", "published", "link", 
            "source", "content", "embedding", "language"
        ],
    )
    return df


def ensure_embeddings(df: pd.DataFrame):
    """Generate embeddings for articles that don't have them"""
    to_process = df[df["embedding"].isna()]
    
    if not to_process.empty:
        with st.spinner(f"Generating embeddings for {len(to_process)} articles..."):
            for _, row in to_process.iterrows():
                text = (row["title"] or "") + " " + (row["summary"] or "")
                if not text.strip():
                    continue
                try:
                    emb = get_embedding(text)
                    safe_execute(
                        "UPDATE articles SET embedding = ? WHERE id = ?",
                        (json.dumps(emb), int(row["id"])),
                    )
                except Exception:
                    continue


def hybrid_search(
    query: str,
    start: dt.date,
    end: dt.date,
    top_k: int,
    precision_mode: bool = False,  # NEW: Precision mode toggle
    min_sim: float = MIN_SIMILARITY,
) -> pd.DataFrame:
    """
    Hybrid search with PRECISION MODE option
    - precision_mode=True: Only exact/similar matches
    - precision_mode=False: Broader semantic search
    """
    df = load_articles_for_range(start, end)
    if df.empty:
        return df

    # Generate missing embeddings
    ensure_embeddings(df)

    # Reload with embeddings
    df = load_articles_for_range(start, end)
    if df.empty:
        return df

    # Get query embedding
    q_emb = np.array(get_embedding(query))
    
    # Calculate similarities
    sims = []
    for _, row in df.iterrows():
        if row["embedding"] and row["embedding"] != "null":
            try:
                emb_vec = np.array(json.loads(row["embedding"]))
                sim = cosine_similarity(q_emb, emb_vec)
            except:
                sim = 0.0
        else:
            sim = 0.0
        sims.append(sim)
    
    df["similarity"] = sims
    
    # PRECISION MODE LOGIC
    if precision_mode:
        # STRICT FILTERING: Only high similarity OR exact keyword matches
        q_lower = query.lower()
        
        # 1. High similarity articles
        high_sim_mask = df["similarity"] >= 0.4  # Higher threshold
        
        # 2. Exact keyword matches
        exact_mask = (
            df["title"].str.lower().str.contains(q_lower, na=False) |
            df["summary"].str.lower().str.contains(q_lower, na=False)
        )
        
        # Combine: High similarity OR exact match
        mask = high_sim_mask | exact_mask
        
        if mask.any():
            filtered = df[mask].copy()
            filtered["match_type"] = np.where(
                exact_mask[mask], "exact", "high_similarity"
            )
        else:
            return pd.DataFrame()
    
    else:
        # NORMAL MODE: Broader semantic search (Professor's preference)
        mask = df["similarity"] >= min_sim
        
        if mask.any():
            filtered = df[mask].copy()
            filtered["match_type"] = "semantic"
        else:
            return pd.DataFrame()
    
    # Sort and limit
    filtered = filtered.sort_values("similarity", ascending=False).head(top_k)
    return filtered


# ========================== QUERY EXPANSION ==========================

def expand_query_with_llm(query: str, enable_expansion: bool = True) -> List[str]:
    """
    Query expansion with toggle
    """
    if not enable_expansion:
        return [query]
    
    client = get_openai_client()

    prompt = f"""
Generate specific search terms for financial news about: "{query}"

Focus on exact terms, companies, and specific contexts.
Return JSON list: ["term1", "term2", "term3"]
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Generate specific financial news search terms."},
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
        
        # Add original query
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
        
        return cleaned[:10]
        
    except Exception:
        return [query]


# ========================== EXTERNAL SOURCES ==========================

def fetch_gdelt_articles(query: str, start: dt.date, end: dt.date) -> int:
    """Fetch from GDELT for more articles"""
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
        r = requests.get(base_url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            articles = data.get("articles", [])
            
            for a in articles[:50]:  # Limit to 50
                title = a.get("title", "")
                summary = a.get("seendate", "")
                link = a.get("url", "")
                src = a.get("domain", "gdelt")
                
                try:
                    safe_execute(
                        """
                        INSERT OR IGNORE INTO articles
                        (title, summary, published, link, source, content, embedding, language, processed)
                        VALUES (?, ?, ?, ?, ?, ?, NULL, 'en', 0)
                        """,
                        (title, summary, dt.datetime.utcnow().isoformat(), link, src, title + " " + summary),
                    )
                    new_count += 1
                except:
                    continue
    except:
        pass
    
    return new_count


def fetch_gnews_articles(query: str, start: dt.date, end: dt.date) -> int:
    """Fetch from Google News"""
    new_count = 0
    
    try:
        gn = GNews(language="en", max_results=GNEWS_MAX_RESULTS)
        gn.start_date = (start.year, start.month, start.day)
        gn.end_date = (end.year, end.month, end.day)
        results = gn.get_news(query)
        
        for r in results[:50]:  # Limit to 50
            title = r.get("title", "")
            link = r.get("url") or r.get("link", "")
            desc = r.get("description", "")
            source = r.get("publisher", {}).get("title", "google-news")
            
            try:
                safe_execute(
                    """
                    INSERT OR IGNORE INTO articles
                    (title, summary, published, link, source, content, embedding, language, processed)
                    VALUES (?, ?, ?, ?, ?, ?, NULL, 'en', 0)
                    """,
                    (title, desc, dt.datetime.utcnow().isoformat(), link, source, title + " " + desc),
                )
                new_count += 1
            except:
                continue
    except:
        pass
    
    return new_count


# ========================== UTILITY FUNCTIONS ==========================

def extract_top_keywords(titles: List[str], n: int = 20) -> List[Tuple[str, int]]:
    text = " ".join(titles).lower()
    words = re.findall(r"[a-zA-Z]{4,}", text)
    stop = {
        "this", "that", "with", "from", "will", "have", "been", "into",
        "after", "over", "under", "they", "them", "your", "their",
        "about", "which", "there", "where", "when", "than", "because",
        "while", "before", "through", "within", "without",
    }
    words = [w for w in words if w not in stop]
    counter = Counter(words)
    return counter.most_common(n)


def calculate_sentiment_index(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    
    pos = len(df[df["sentiment_label"] == "positive"])
    neg = len(df[df["sentiment_label"] == "negative"])
    neu = len(df[df["sentiment_label"] == "neutral"])
    total = len(df)
    
    if total == 0:
        return 0.0
    
    weighted_sum = (pos * 1.0) + (neg * -1.0) + (neu * 0.0)
    index = (weighted_sum / total) * 100
    
    return round(index, 2)


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
        "**25+ RSS Feeds → Smart Search → Balanced Sentiment Analysis**"
    )

    # SIDEBAR
    with st.sidebar:
        st.header("📊 Database Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Fetch ALL RSS Feeds", type="primary"):
                total = fetch_all_rss_feeds()
                st.success(f"✅ Added {total} new articles!")
        
        with col2:
            if st.button("🗑️ Clear Database"):
                conn.execute("DELETE FROM articles")
                conn.commit()
                st.success("Database cleared!")
        
        # Show database stats
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM articles")
        total_articles = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM articles WHERE embedding IS NOT NULL")
        embedded_articles = cur.fetchone()[0]
        
        st.metric("📂 Total Articles", f"{total_articles:,}")
        st.metric("🧠 With Embeddings", f"{embedded_articles:,}")
        
        st.markdown("---")
        st.header("🔍 Search Settings")
        
        today = dt.date.today()
        default_start = today - dt.timedelta(days=90)  # 3 months for more articles
        start_date = st.date_input("Start date", default_start)
        end_date = st.date_input("End date", today)
        
        st.subheader("Search Mode")
        search_mode = st.radio(
            "Select search mode:",
            ["🔍 Precision Mode (Exact matches)", "🌐 Broad Mode (Semantic + Related)"],
            index=1,
            help="Precision: Only exact/similar articles | Broad: Includes related articles"
        )
        
        precision_mode = (search_mode == "🔍 Precision Mode (Exact matches)")
        
        st.subheader("Additional Sources")
        use_gdelt = st.checkbox("Include GDELT", True)
        use_gnews = st.checkbox("Include Google News", True)
        use_query_expansion = st.checkbox("Use Query Expansion", True)
        filter_english = st.checkbox("Filter English articles only", True)
        
        max_articles = st.slider(
            "Maximum articles",
            min_value=100,
            max_value=1000,
            value=500,
            step=100,
        )

    # MAIN QUERY
    st.markdown("### 🔍 Search Topic")
    query = st.text_input(
        "Enter topic (e.g. 'Germany', 'US Tech', 'Nvidia', 'Oil Prices')",
        key="main_query",
        placeholder="Type your topic here..."
    )

    analyze_clicked = st.button("🚀 Analyze", type="primary")

    if not analyze_clicked:
        st.info("Enter a topic and click **Analyze** to run the analysis.")
        
        # Show recent articles
        st.markdown("### 📰 Recent Articles in Database")
        cur = conn.cursor()
        cur.execute("""
            SELECT title, source, published 
            FROM articles 
            WHERE language = 'en'
            ORDER BY published DESC 
            LIMIT 10
        """)
        recent = cur.fetchall()
        
        for title, source, published in recent:
            st.caption(f"**{title}**")
            st.caption(f"📰 {source} | 📅 {published[:10]}")
            st.markdown("---")
        
        return

    if not query.strip():
        st.warning("Please enter a non-empty topic.")
        return

    # QUERY EXPANSION
    if use_query_expansion and not precision_mode:
        with st.spinner("Expanding search terms..."):
            expanded_queries = expand_query_with_llm(query, use_query_expansion)
        
        st.write("**Expanded search terms:**")
        st.write(", ".join(expanded_queries[:8]) + ("..." if len(expanded_queries) > 8 else ""))
        search_queries = expanded_queries
    else:
        search_queries = [query]
        st.info(f"Using exact query: **{query}**")

    # EXTERNAL SOURCES
    external_added = 0
    if use_gdelt or use_gnews:
        with st.spinner("Fetching from external sources..."):
            for q in search_queries[:3]:
                if use_gdelt:
                    external_added += fetch_gdelt_articles(q, start_date, end_date)
                if use_gnews:
                    external_added += fetch_gnews_articles(q, start_date, end_date)
        
        if external_added > 0:
            st.success(f"Added {external_added} articles from external sources")

    # HYBRID SEARCH
    with st.spinner(f"Searching {len(search_queries)} queries..."):
        dfs = []
        total_found = 0
        
        for q in search_queries:
            part = hybrid_search(
                query=q,
                start=start_date,
                end=end_date,
                top_k=max_articles,
                precision_mode=precision_mode,
                min_sim=0.4 if precision_mode else MIN_SIMILARITY
            )
            
            if not part.empty:
                part["query_term"] = q
                dfs.append(part)
                total_found += len(part)
        
        if not dfs:
            st.error(f"No articles found for '{query}'. Try:")
            st.info("1. Broaden date range (currently {start_date} to {end_date})")
            st.info("2. Use 'Broad Mode' instead of 'Precision Mode'")
            st.info("3. Click 'Fetch ALL RSS Feeds' to add more articles")
            return

        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["link"])

    st.success(f"🔎 Found **{len(df):,}** relevant articles.")

    # FILTER ENGLISH
    if filter_english:
        with st.spinner("Filtering English articles..."):
            df_before = len(df)
            df = filter_english_articles(df)
            df_after = len(df)
            if df_after < df_before:
                st.info(f"Filtered out {df_before - df_after} non-English articles.")

    # LIMIT ARTICLES
    if len(df) > max_articles:
        df = df.head(max_articles)
        st.info(f"Limited to {max_articles} articles for analysis")

    if df.empty:
        st.error("No articles remained after filtering.")
        return

    st.success(f"✅ Analyzing **{len(df)}** articles")

    # SENTIMENT ANALYSIS
    with st.spinner(f"Analyzing sentiment..."):
        texts = df["content"].fillna(df["summary"]).tolist()
        sents = finbert_sentiment(texts)
        df["sentiment_label"] = [s["label"] for s in sents]
        df["sentiment_score"] = [s["score"] for s in sents]
    
    df["relevance"] = (df["similarity"] * 100).round(1)
    df["source_domain"] = df["source"].apply(
        lambda x: x.split("//")[-1].split("/")[0] if "//" in str(x) else str(x)
    )

    # CALCULATE METRICS
    sentiment_index = calculate_sentiment_index(df)
    
    counts = df["sentiment_label"].value_counts()
    total = len(df)
    pos = counts.get("positive", 0)
    neg = counts.get("negative", 0)
    neu = counts.get("neutral", 0)
    
    avg_confidence = (df["sentiment_score"].mean() * 100)

    # TABS
    tab_dash, tab_articles, tab_keywords, tab_download = st.tabs(
        ["📊 Dashboard", "📰 Articles", "🔑 Keywords", "📥 Download"]
    )

    # DASHBOARD TAB
    with tab_dash:
        st.subheader("Executive Sentiment Overview")
        
        # Show search mode info
        mode_info = "🔍 **Precision Mode**: Showing exact/similar matches" if precision_mode else "🌐 **Broad Mode**: Including related articles"
        st.info(mode_info)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Articles", f"{total:,}")
        c2.metric("Positive", f"{pos:,}", f"{(pos/total*100):.1f}%" if total else "0.0%")
        c3.metric("Negative", f"{neg:,}", f"{(neg/total*100):.1f}%" if total else "0.0%")
        c4.metric("Neutral", f"{neu:,}", f"{(neu/total*100):.1f}%" if total else "0.0%")
        
        direction = "Bullish" if sentiment_index > 0 else "Bearish" if sentiment_index < 0 else "Neutral"
        c5.metric("Sentiment Index", f"{sentiment_index:.1f}", direction)

        st.markdown("---")

        # CHARTS
        col1, col2 = st.columns(2)
        
        with col1:
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
        
        with col2:
            st.markdown("#### Articles by Source")
            src_counts = df["source_domain"].value_counts().head(15)
            fig_src = px.bar(
                x=src_counts.values,
                y=src_counts.index,
                orientation="h",
                labels={"x": "Count", "y": "Source"},
                color=src_counts.values,
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_src, use_container_width=True)

    # ARTICLES TAB
    with tab_articles:
        st.subheader(f"📰 Articles ({len(df)} total)")
        
        # Show match type distribution
        if "match_type" in df.columns:
            match_counts = df["match_type"].value_counts()
            st.caption(f"Match types: {', '.join([f'{k}: {v}' for k, v in match_counts.items()])}")
        
        sent_filter = st.multiselect(
            "Filter by sentiment",
            options=["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"],
        )
        
        filtered = df[df["sentiment_label"].isin(sent_filter)].copy()
        filtered = filtered.sort_values("relevance", ascending=False)
        
        for _, row in filtered.iterrows():
            icon = {"positive": "🟢", "negative": "🔴", "neutral": "🔵"}[row["sentiment_label"]]
            
            # Show match type badge
            match_badge = ""
            if "match_type" in row and row["match_type"] == "exact":
                match_badge = "🎯 "
            elif "match_type" in row and row["match_type"] == "high_similarity":
                match_badge = "⚡ "
            
            st.markdown(f"**{match_badge}{icon} {row['title']}**")
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            c1.caption(f"📰 {row['source_domain']}")
            c2.caption(f"🎯 Relevance: {row['relevance']:.1f}%")
            c3.caption(f"😊 {row['sentiment_label']} ({row['sentiment_score']*100:.1f}%)")
            c4.markdown(f"[Read →]({row['link']})")
            if row["summary"]:
                st.caption(row["summary"][:300] + "…")
            st.markdown("---")

    # KEYWORDS TAB
    with tab_keywords:
        st.subheader("Trending Keywords")
        keywords = extract_top_keywords(df["title"].tolist(), n=25)
        if keywords:
            kw_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])
            fig_kw = px.bar(
                kw_df,
                x="Frequency",
                y="Keyword",
                orientation="h",
                color="Frequency",
                color_continuous_scale="Blues",
            )
            fig_kw.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_kw, use_container_width=True)
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
                "match_type" if "match_type" in df.columns else "",
                "query_term",
            ]
        ]
        csv = dl_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name=f"news_sentiment_{query.replace(' ', '_')}.csv",
            mime="text/csv",
        )
        st.write("### Data Preview")
        st.dataframe(dl_df.head(20), use_container_width=True)


if __name__ == "__main__":
    run_app()
