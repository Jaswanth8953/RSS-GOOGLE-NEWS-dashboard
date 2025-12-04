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

# OPTIMIZED CONFIG
MIN_SIMILARITY = 0.25
MAX_ARTICLES_DEFAULT = 200
GDELT_MAX_RECORDS = 100
GNEWS_MAX_RESULTS = 100

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
    conn.execute("PRAGMA cache_size = -20000;")

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
    BALANCED FinBERT calibration to avoid extremes.
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
        
        # BALANCED CALIBRATION
        if max_score < 0.55:  # Not confident → neutral
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
        
        # Force diversity: If negative has >30% probability, make it neutral
        neg_prob = float(p[0])
        if label == "positive" and neg_prob > 0.30:
            label = "neutral"
            score = 0.5
            
        results.append({"label": label, "score": score})
    
    return results


# ========================== LANGUAGE DETECTION ==========================

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


# ========================== HYBRID SEARCH ==========================

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
            "id", "title", "summary", "published", "link", 
            "source", "content", "embedding", "language"
        ],
    )
    return df


def ensure_embeddings(df_ids_title_summary: pd.DataFrame):
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
    df = load_articles_for_range(start, end)
    if df.empty:
        return df

    df["embedding"] = df["embedding"].apply(
        lambda x: None if x in (None, "", "NULL") else x
    )

    ensure_embeddings(df[["id", "title", "summary", "embedding"]])

    df = load_articles_for_range(start, end)
    if df.empty:
        return df

    q_emb = np.array(get_embedding(query))
    sims = []
    for _, row in df.iterrows():
        emb_vec = np.array(json.loads(row["embedding"]))
        sims.append(cosine_similarity(q_emb, emb_vec))
    df["similarity"] = sims

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


# ========================== QUERY EXPANSION (FIXED - BALANCED) ==========================

def expand_query_with_llm(query: str) -> List[str]:
    """
    BALANCED query expansion that includes BOTH positive and negative contexts.
    This is what Professor wants: "Germany" → "German Industry"
    """
    client = get_openai_client()

    # DETERMINE QUERY TYPE
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['tech', 'technology', 'nvidia', 'apple', 'microsoft', 'ai']):
        prompt = f"""
Generate search terms for financial news about: "{query}"

Include:
1. Related companies/products
2. Industry terms
3. BOTH positive (growth, innovation) AND negative (risks, challenges) contexts
4. Regulatory/legal aspects
5. Competitors

For technology topics, include: chip manufacturing, AI regulation, antitrust, layoffs, earnings, competition

Return JSON list: ["term1", "term2", "term3"]
"""
    elif any(word in query_lower for word in ['germany', 'europe', 'china', 'us', 'uk', 'india']):
        prompt = f"""
Generate search terms for financial news about: "{query}"

Include:
1. Related industries in that region
2. Economic indicators
3. Trade relations
4. Government policies
5. Both growth opportunities and economic challenges

Return JSON list: ["term1", "term2", "term3"]
"""
    else:
        prompt = f"""
Generate balanced search terms for financial news about: "{query}"

Include BOTH:
- Positive aspects (growth, opportunities, strengths)
- Negative aspects (risks, challenges, weaknesses)
- Related industries/sectors
- Key companies/organizations

Return JSON list: ["term1", "term2", "term3"]
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate balanced financial news search terms."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,  # More creative
            max_tokens=300
        )
        content = resp.choices[0].message.content.strip()
        
        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        expansions = json.loads(content)
        expansions = [str(t) for t in expansions if isinstance(t, str)]
        
        # Add MANUAL expansions for common queries
        manual_expansions = []
        
        if "us tech" in query_lower or "technology" in query_lower:
            manual_expansions = [
                "technology stocks", "tech industry", "software companies",
                "hardware manufacturers", "semiconductor stocks", "AI companies",
                "tech layoffs", "tech regulation", "antitrust technology",
                "tech earnings", "startup funding", "venture capital tech"
            ]
        elif "nvidia" in query_lower:
            manual_expansions = [
                "NVIDIA stock", "GPU chips", "artificial intelligence chips",
                "semiconductor industry", "AMD competition", "chip manufacturing",
                "AI hardware", "graphics cards", "data center chips",
                "chip export bans", "semiconductor shortage", "earnings report"
            ]
        elif "germany" in query_lower:
            manual_expansions = [
                "German economy", "Germany industry", "Berlin startups",
                "German exports", "European Union Germany", "German automotive",
                "German engineering", "DAX index", "German manufacturing",
                "energy crisis Germany", "inflation Germany", "recession Germany"
            ]
        
        # Combine and deduplicate
        all_terms = [query] + expansions + manual_expansions
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
        
        return cleaned[:10]  # Limit to 10 terms
        
    except Exception as e:
        # Fallback: simple expansion
        query_lower = query.lower()
        if "tech" in query_lower:
            return [query, "technology", "software", "hardware", "AI", "semiconductors", "innovation"]
        elif "nvidia" in query_lower:
            return [query, "GPU", "artificial intelligence", "chips", "semiconductors", "graphics"]
        elif "germany" in query_lower:
            return [query, "German", "Europe", "EU", "Berlin", "manufacturing", "automotive"]
        else:
            return [query]


# ========================== BALANCED ARTICLE SELECTION ==========================

def select_diverse_articles(df: pd.DataFrame, max_articles: int = 200) -> pd.DataFrame:
    """
    Select articles ensuring diversity in sources and content.
    No LLM bias - simple algorithm.
    """
    if df.empty:
        return df
    
    if len(df) <= max_articles:
        return df
    
    # Sort by relevance
    df_sorted = df.sort_values("similarity", ascending=False)
    
    # Take top 70% by relevance
    top_count = int(max_articles * 0.7)
    selected = df_sorted.head(top_count).copy()
    
    # Add 30% from diverse sources
    remaining = df_sorted.iloc[top_count:].copy()
    
    # Ensure source diversity
    sources_in_selected = selected["source"].unique()
    
    for source in remaining["source"].unique():
        if source not in sources_in_selected:
            source_articles = remaining[remaining["source"] == source]
            if not source_articles.empty:
                selected = pd.concat([selected, source_articles.head(2)], ignore_index=True)
    
    # If still need more, add from remaining by date diversity
    if len(selected) < max_articles:
        needed = max_articles - len(selected)
        # Get articles from different dates
        remaining["published_date"] = pd.to_datetime(remaining["published"]).dt.date
        date_groups = remaining.groupby("published_date")
        
        for date, group in date_groups:
            if not group.empty and len(selected) < max_articles:
                selected = pd.concat([selected, group.head(1)], ignore_index=True)
    
    # Final trim
    if len(selected) > max_articles:
        selected = selected.head(max_articles)
    
    return selected


# ========================== SENTIMENT CALCULATION ==========================

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
        "RSS Feeds → **Semantic Search + Query Expansion** → FinBERT Sentiment Analysis"
    )

    # SIDEBAR
    with st.sidebar:
        st.header("Settings")

        if st.button("Fetch Latest RSS Articles"):
            with st.spinner("Fetching RSS feeds…"):
                n = fetch_rss_articles()
            st.success(f"Added {n} new articles")

        today = dt.date.today()
        default_start = today - dt.timedelta(days=30)
        start_date = st.date_input("Start date", default_start)
        end_date = st.date_input("End date", today)
        
        st.subheader("Analysis Options")
        use_query_expansion = st.checkbox("Use Query Expansion", True,
                                         help="Expand search terms for better coverage (e.g., Germany → German Industry)")
        filter_english = st.checkbox("Filter English articles only", True)
        
        max_articles = st.slider(
            "Maximum articles to analyze",
            min_value=50,
            max_value=500,
            value=MAX_ARTICLES_DEFAULT,
            step=50,
        )

    # MAIN QUERY
    st.markdown("### 🔍 Search Topic")
    query = st.text_input(
        "Enter topic (e.g. 'US Tech', 'Germany', 'Nvidia', 'European textiles')",
        key="main_query",
        placeholder="Type your topic here..."
    )

    analyze_clicked = st.button("Analyze", type="primary")

    if not analyze_clicked:
        st.info("Enter a topic and click **Analyze** to run the analysis.")
        return

    if not query.strip():
        st.warning("Please enter a non-empty topic.")
        return

    # QUERY EXPANSION
    if use_query_expansion:
        with st.spinner("Expanding search terms for better coverage…"):
            expanded_queries = expand_query_with_llm(query)
        
        st.write("**Expanded search terms:**")
        st.write(", ".join(expanded_queries[:8]) + ("..." if len(expanded_queries) > 8 else ""))
    else:
        expanded_queries = [query]
        st.info("Using original query only (no expansion)")

    # HYBRID SEARCH
    with st.spinner("Searching for relevant articles…"):
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
                "No articles found for this topic in the selected date range. "
                "Try broadening the date range or using different search terms."
            )
            return

        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["link"])

    st.write(f"🔎 Found **{len(df)}** relevant articles.")

    # FILTER ENGLISH ARTICLES
    if filter_english:
        with st.spinner("Filtering English articles…"):
            df_before = len(df)
            df = filter_english_articles(df)
            df_after = len(df)
            if df_after < df_before:
                st.info(f"Filtered out {df_before - df_after} non-English articles.")

    # SELECT DIVERSE ARTICLES
    with st.spinner("Selecting diverse articles for analysis…"):
        df = select_diverse_articles(df, max_articles)
    
    if df.empty:
        st.error("No articles remained after filtering.")
        return

    st.success(f"✅ Selected **{len(df)}** articles for sentiment analysis.")

    # SENTIMENT ANALYSIS
    with st.spinner(f"Analyzing sentiment for {len(df)} articles…"):
        texts = df["content"].fillna(df["summary"]).tolist()
        sents = finbert_sentiment(texts)
        df["sentiment_label"] = [s["label"] for s in sents]
        df["sentiment_score"] = [s["score"] for s in sents]
    
    df["relevance"] = (df["similarity"] * 100).round(1)
    df["source_domain"] = df["source"].apply(
        lambda x: x.split("//")[-1].split("/")[0] if "//" in str(x) else str(x)
    )

    # CALCULATE SENTIMENT METRICS
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

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Articles", total)
        c2.metric("Positive", pos, f"{(pos/total*100):.1f}%" if total else "0.0%")
        c3.metric("Negative", neg, f"{(neg/total*100):.1f}%" if total else "0.0%")
        c4.metric("Neutral", neu, f"{(neu/total*100):.1f}%" if total else "0.0%")
        
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
            st.markdown("#### Sentiment Index Gauge")
            gauge_color = "#2ecc71" if sentiment_index > 0 else "#e74c3c"
            fig_g = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=sentiment_index,
                    title={"text": "Sentiment Index"},
                    delta={"reference": 0},
                    gauge={
                        "axis": {"range": [-100, 100]},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [-100, 0], "color": "rgba(231, 76, 60, 0.2)"},
                            {"range": [0, 100], "color": "rgba(46, 204, 113, 0.2)"},
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
        
        # SOURCE DISTRIBUTION
        st.markdown("#### Articles by Source")
        src_counts = df["source_domain"].value_counts().head(10)
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
        st.subheader("Articles")
        
        sent_filter = st.multiselect(
            "Filter by sentiment",
            options=["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"],
        )
        
        filtered = df[df["sentiment_label"].isin(sent_filter)].copy()
        filtered = filtered.sort_values("relevance", ascending=False)
        
        for _, row in filtered.iterrows():
            icon = {"positive": "🟢", "negative": "🔴", "neutral": "🔵"}[row["sentiment_label"]]
            st.markdown(f"**{icon} {row['title']}**")
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            c1.caption(row["source_domain"])
            c2.caption(f"Relevance: {row['relevance']:.1f}%")
            c3.caption(f"Sentiment: {row['sentiment_label']} ({row['sentiment_score']*100:.1f}%)")
            c4.markdown(f"[Read →]({row['link']})")
            if row["summary"]:
                st.caption(row["summary"][:280] + "…")
            st.markdown("---")

    # KEYWORDS TAB
    with tab_keywords:
        st.subheader("Trending Keywords")
        keywords = extract_top_keywords(df["title"].tolist(), n=20)
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
        st.dataframe(dl_df.head(20), use_container_width=True)


if __name__ == "__main__":
    run_app()
