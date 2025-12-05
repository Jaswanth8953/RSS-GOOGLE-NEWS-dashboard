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


# ========================== FIXED FINBERT SENTIMENT ==========================

@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def finbert_sentiment(texts: List[str]) -> List[dict]:
    """COMPLETELY REVISED FinBERT sentiment - fixes the bug"""
    if not texts:
        return []
    
    tokenizer, model, device = load_finbert()
    
    # Process in batches to avoid memory issues
    batch_size = 32
    all_results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Clean texts
        batch_texts = [str(t)[:1000] if t else "" for t in batch_texts]
        
        try:
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
                
                # SIMPLE LOGIC - NO OVERLY AGGRESSIVE NEUTRAL CONVERSION
                # Only force neutral if confidence is very low
                if max_score < 0.50:  # Reduced threshold
                    label = "neutral"
                    score = 0.5
                else:
                    score = max_score
                
                all_results.append({"label": label, "score": score})
                
        except Exception as e:
            # If batch fails, assign neutral
            for _ in batch_texts:
                all_results.append({"label": "neutral", "score": 0.5})
    
    return all_results


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
    prompt = f"""Generate financial news search terms for: "{query}"

Return JSON array of 8-12 terms.

Example: ["term1", "term2", "term3"]

Your JSON:"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return JSON arrays only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300,
        )
        content = resp.choices[0].message.content.strip()
        
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
        
        expansions = json.loads(content)
        
        if not isinstance(expansions, list):
            raise ValueError("Not a list")
        
        expansions = [str(t).strip() for t in expansions if t and str(t).strip()]
        
        all_terms = [query.strip()] + expansions
        seen = set()
        cleaned = []
        for t in all_terms:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(t)
        
        return cleaned[:12]
        
    except Exception:
        return [query]


# ========================== SIMPLE LLM FILTER ==========================

def llm_select_articles(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """Simple filter - returns reasonable number of articles"""
    if len(df) <= 50:
        return df
    
    # Just use similarity ranking
    filtered = df.head(150).copy()
    return filtered


# ========================== FIXED SENTIMENT INDEX ==========================

def calculate_sentiment_index(df: pd.DataFrame) -> Tuple[float, dict]:
    """Calculate sentiment index and return with counts"""
    if df.empty:
        return 0.0, {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
    
    # Only scored English articles
    scored_df = df[
        (df["language"] == "en") & 
        df["sentiment_label"].isin(["positive", "neutral", "negative"])
    ].copy()
    
    if scored_df.empty:
        return 0.0, {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
    
    # Counts
    counts = scored_df["sentiment_label"].value_counts()
    pos = int(counts.get("positive", 0))
    neg = int(counts.get("negative", 0))
    neu = int(counts.get("neutral", 0))
    total_scored = pos + neg + neu
    
    # Sentiment index (only if we have scored articles)
    if total_scored > 0:
        sent_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        scored_df["sent_value"] = scored_df["sentiment_label"].map(sent_map).fillna(0.0)
        
        # Weight by confidence if available
        if "sentiment_score" in scored_df.columns:
            weights = scored_df["sentiment_score"].fillna(0.5)
            index = (scored_df["sent_value"] * weights).sum() / weights.sum() * 100
        else:
            index = scored_df["sent_value"].mean() * 100
        
        index = float(np.clip(index, -100.0, 100.0))
        index = round(index, 1)
    else:
        index = 0.0
    
    return index, {"positive": pos, "negative": neg, "neutral": neu, "total": total_scored}


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
        "Enter topic (e.g., 'US Tech', 'Germany', 'oil')",
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
        expanded_terms = expand_query(query)
    
    if len(expanded_terms) > 1:
        st.markdown("### 🔍 Expanded terms used:")
        st.write(", ".join(expanded_terms[:8]) + ("..." if len(expanded_terms) > 8 else ""))

    # Fetch additional articles
    total_added = 0
    if use_gdelt or use_gnews:
        with st.spinner("🌐 Fetching additional articles..."):
            for q in expanded_terms[:5]:
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
            st.error("❌ No articles found.")
            return
    
    st.success(f"✅ Found **{len(df)}** candidate articles.")

    # Article Selection
    if len(df) > 50:
        with st.spinner("🤖 Selecting relevant articles..."):
            df = llm_select_articles(query, df)
        st.info(f"📊 Using {len(df)} most relevant articles.")
    else:
        st.success(f"✅ Using all {len(df)} articles.")

    # Language detection
    df = ensure_language(df)

    # FIXED: Sentiment Analysis
    with st.spinner("😊 Analyzing sentiment..."):
        # Initialize columns
        df["sentiment_label"] = "not_scored"
        df["sentiment_score"] = 0.5
        
        # Only analyze English articles
        mask_en = df["language"] == "en"
        df_en = df[mask_en].copy()
        
        if not df_en.empty:
            # Prepare texts
            texts = []
            indices = []
            for idx, row in df_en.iterrows():
                # Use title + summary
                text = f"{row.get('title', '')}. {row.get('summary', '')}"
                if text.strip():
                    texts.append(text[:800])  # Limit length
                    indices.append(idx)
            
            if texts:
                # Run sentiment analysis
                try:
                    sents = finbert_sentiment(texts)
                    
                    # Apply results
                    for i, idx in enumerate(indices):
                        if i < len(sents):
                            df.at[idx, "sentiment_label"] = sents[i]["label"]
                            df.at[idx, "sentiment_score"] = sents[i]["score"]
                except Exception as e:
                    st.warning(f"⚠️ Sentiment analysis had issues. Using neutral for all.")

    # Calculate relevance and source
    df["relevance"] = (df["similarity"] * 100).round(1)
    df["source_domain"] = df["source"].apply(
        lambda x: x.split("//")[-1].split("/")[0]
        if isinstance(x, str) and "//" in x
        else (str(x)[:30] if x else "unknown")
    )

    # Calculate sentiment index and counts
    sent_index, sent_counts = calculate_sentiment_index(df)
    total_articles = len(df)
    pos = sent_counts["positive"]
    neg = sent_counts["negative"]
    neu = sent_counts["neutral"]
    total_scored = sent_counts["total"]

    # ================== TABS ==================
    
    tab_dash, tab_articles, tab_keywords, tab_download = st.tabs(
        ["📈 Dashboard", "📰 Articles", "🔑 Keywords", "📥 Download"]
    )

    # ----- Dashboard tab -----
    with tab_dash:
        st.subheader("📊 Sentiment Overview")
        
        # Display metrics - FIXED
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📄 Total Articles", total_articles)
        
        with col2:
            st.metric("🟢 Positive", pos)
        
        with col3:
            st.metric("🔴 Negative", neg)
        
        with col4:
            st.metric("🔵 Neutral", neu)
        
        with col5:
            # Determine direction
            if sent_index > 15:
                direction = "📈 Bullish"
                delta_color = "normal"
            elif sent_index < -15:
                direction = "📉 Bearish"
                delta_color = "inverse"
            elif abs(sent_index) < 5:
                direction = "➡️ Neutral"
                delta_color = "off"
            else:
                direction = "Slightly Bullish" if sent_index > 0 else "Slightly Bearish"
                delta_color = "normal" if sent_index > 0 else "inverse"
            
            st.metric(
                "📊 Sentiment Index", 
                f"{sent_index:.1f}", 
                direction,
                delta_color=delta_color
            )
        
        st.markdown("---")
        
        # Charts - FIXED
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### 📊 Sentiment Distribution")
            # Only show scored sentiments
            scored_sentiments = df[
                df["sentiment_label"].isin(["positive", "negative", "neutral"])
            ]["sentiment_label"]
            
            if not scored_sentiments.empty:
                dist = scored_sentiments.value_counts()
                
                # Create pie chart
                fig_pie = px.pie(
                    names=dist.index,
                    values=dist.values,
                    color=dist.index,
                    color_discrete_map={
                        "positive": "#2ecc71",
                        "negative": "#e74c3c",
                        "neutral": "#3498db",
                    },
                    hole=0.4,
                )
                fig_pie.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                    hovertemplate="<b>%{label}</b><br>%{value} articles<br>%{percent}"
                )
                fig_pie.update_layout(
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.1,
                        xanchor="center",
                        x=0.5
                    )
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No sentiment-scored articles available.")
        
        with col_chart2:
            st.markdown("#### 📰 Articles by Source")
            src_counts = df["source_domain"].value_counts().head(10)
            if not src_counts.empty:
                fig_src = px.bar(
                    x=src_counts.values,
                    y=src_counts.index,
                    orientation="h",
                    labels={"x": "Count", "y": "Source"},
                    color=src_counts.values,
                    color_continuous_scale="Blues",
                )
                fig_src.update_layout(showlegend=False)
                st.plotly_chart(fig_src, use_container_width=True)
            else:
                st.info("No source data available.")

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
                ["Relevance", "Date", "Sentiment"],
                index=0
            )
        
        # Apply filters
        df_art = df[df["sentiment_label"].isin(sent_filter)].copy()
        
        if sort_by == "Relevance":
            df_art = df_art.sort_values("relevance", ascending=False)
        elif sort_by == "Date":
            df_art = df_art.sort_values("published", ascending=False)
        elif sort_by == "Sentiment":
            df_art = df_art.sort_values("sentiment_score", ascending=False)
        
        # Display
        for idx, row in df_art.iterrows():
            icon_map = {
                "positive": "🟢",
                "negative": "🔴",
                "neutral": "🔵",
                "not_scored": "⚪",
            }
            icon = icon_map.get(row["sentiment_label"], "⚪")
            
            with st.container():
                st.markdown(f"**{icon} {row['title']}**")
                
                cols = st.columns([3, 2, 2, 1])
                cols[0].caption(f"📰 {row['source_domain']}")
                cols[1].caption(f"🎯 {row['relevance']:.1f}% relevant")
                if row["sentiment_label"] in ["positive", "negative", "neutral"]:
                    score_pct = row.get('sentiment_score', 0.5) * 100
                    cols[2].caption(f"{row['sentiment_label'].title()} ({score_pct:.0f}%)")
                else:
                    cols[2].caption("Not scored")
                cols[3].markdown(f"[📖 Read]({row['link']})")
                
                if row.get("summary"):
                    summary = str(row["summary"])
                    st.caption(summary[:250] + ("..." if len(summary) > 250 else ""))
                
                st.markdown("---")

    # ----- Keywords tab -----
    with tab_keywords:
        st.subheader("🔑 Trending Keywords")
        
        keywords = extract_top_keywords(df["title"].tolist(), n=20)
        
        if keywords:
            kw_df = pd.DataFrame(keywords[:15], columns=["Keyword", "Frequency"])
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
            st.info("Not enough data for keywords.")

    # ----- Download tab -----
    with tab_download:
        st.subheader("📥 Download Results")
        
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
                "similarity",
                "query_term",
            ]
        ].copy()
        
        csv = dl_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name=f"sentiment_{query.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("#### Preview")
        st.dataframe(dl_df.head(10), use_container_width=True)


if __name__ == "__main__":
    run_app()
