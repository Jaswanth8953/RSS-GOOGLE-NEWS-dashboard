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
    # Streamlit Cloud secrets: [openai] api_key="..."
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        api_key = st.secrets["openai"]["api_key"]

    # Fallback: environment variable (AWS, local)
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


# ========================== IMPROVED FINBERT SENTIMENT ==========================

@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def finbert_sentiment(texts: List[str]) -> List[dict]:
    """Improved FinBERT with better calibration to avoid too many neutrals."""
    if not texts:
        return []

    tokenizer, model, device = load_finbert()
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,  # Increased from 256
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
        
        # FIX: Less aggressive neutral conversion
        # Only force neutral if confidence is very low
        if max_score < 0.55:  # Lowered from 0.60
            label = "neutral"
            score = 0.55  # Slightly above threshold
        else:
            label = base_label
            score = max_score
            
            # Don't automatically convert positives with negative probability
            # This was causing too many neutrals
            if label == "positive" and p[0] > 0.40:  # Increased threshold
                # Only demote to neutral if negative probability is very high
                if max_score - p[0] < 0.15:  # If positive and negative are close
                    label = "neutral"
                    score = 0.55
        
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
    """Fetch from GDELT DOC API for this query and date range."""
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
    """Compute embeddings only for rows that are missing them."""
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

    # Ensure embeddings for rows in this date range
    ensure_embeddings(df)
    df = load_articles_for_range(start, end)
    if df.empty:
        return df

    # Semantic similarity
    q_emb = np.array(get_embedding(query))
    sims = []
    for _, row in df.iterrows():
        try:
            emb_vec = np.array(json.loads(row["embedding"]))
            sims.append(cosine_similarity(q_emb, emb_vec))
        except Exception:
            sims.append(0.0)
    df["similarity"] = sims

    # Keyword check
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


# ========================== IMPROVED LLM QUERY EXPANSION ==========================

def expand_query(query: str) -> List[str]:
    """Use GPT-4o-mini to create related search terms."""
    client = get_openai_client()
    prompt = f"""
You expand financial news search queries. Return ONLY a valid JSON array of strings.

Original query: "{query}"

Generate 8-12 related search terms for financial news. Focus on:
- Financial/business synonyms
- Related industries, companies, markets
- Broader and narrower financial concepts
- Common terms used in business journalism

Examples for "US Tech":
["US technology stocks", "Silicon Valley companies", "American tech sector", 
"NASDAQ tech companies", "US software industry", "big tech earnings", 
"tech startup funding", "US semiconductor stocks"]

Return JSON ONLY:
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You return only valid JSON arrays. No explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300,
        )
        content = resp.choices[0].message.content.strip()
        
        # Clean the response - remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
        if content.startswith("json"):
            content = content[4:].strip()
        
        # Parse JSON
        expansions = json.loads(content)
        
        if not isinstance(expansions, list):
            raise ValueError("Response is not a list")
        
        expansions = [str(t).strip() for t in expansions if t and str(t).strip()]
        
        # Combine with original and deduplicate
        all_terms = [query.strip()] + expansions
        seen = set()
        cleaned = []
        for t in all_terms:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(t)
        
        return cleaned[:12]  # Limit to 12 terms
        
    except Exception as e:
        st.warning(f"⚠️ Query expansion failed: {str(e)[:100]}. Using original term only.")
        return [query]


# ========================== IMPROVED LLM ARTICLE FILTER ==========================

def llm_select_articles(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """Let LLM choose genuinely relevant subset from hybrid search results."""
    if df.empty or len(df) <= 50:  # If small dataset, keep all
        return df
    
    # Limit to top 150 for LLM processing
    df_sample = df.head(150).copy()
    
    client = get_openai_client()
    
    # Format articles for LLM
    articles_list = []
    for _, row in df_sample.iterrows():
        articles_list.append({
            "id": int(row["id"]),
            "title": str(row["title"])[:150],
            "summary": (str(row["summary"] or "")[:200]).strip()
        })
    
    prompt = f"""Select articles relevant to the financial topic: "{query}"

Instructions:
1. Return ONLY a JSON array of article IDs that are relevant
2. Consider financial/business relevance, not general news
3. Only include articles clearly related to {query} in financial context
4. Exclude off-topic or irrelevant articles

Available articles:
{json.dumps(articles_list, indent=2)}

Return JSON array of selected IDs:"""
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You output only valid JSON arrays. No explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500,
        )
        
        content = resp.choices[0].message.content.strip()
        
        # Clean response
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
        
        # Parse JSON
        selected_ids = json.loads(content)
        
        if not isinstance(selected_ids, list):
            raise ValueError("Response is not a list")
        
        # Convert to integers
        selected_ids = [int(i) for i in selected_ids if str(i).isdigit()]
        
        if not selected_ids:
            st.warning("LLM selected 0 articles. Keeping original results.")
            return df_sample
        
        # Filter and return
        filtered = df_sample[df_sample["id"].isin(selected_ids)]
        st.success(f"✅ LLM selected {len(filtered)} highly relevant articles.")
        return filtered
        
    except json.JSONDecodeError as e:
        st.warning(f"⚠️ LLM returned invalid JSON. Keeping all {len(df_sample)} articles.")
        return df_sample
    except Exception as e:
        st.warning(f"⚠️ LLM filtering failed: {str(e)[:100]}. Keeping all articles.")
        return df_sample


# ========================== SENTIMENT INDEX ==========================

def calculate_sentiment_index(df: pd.DataFrame) -> float:
    """Compute composite index only on EN articles."""
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
    index = float(np.clip(index, -100.0, 100.0))
    return round(index, 1)


# ========================== KEYWORDS ==========================

def extract_top_keywords(titles: List[str], n: int = 20) -> List[Tuple[str, int]]:
    text = " ".join(titles).lower()
    words = re.findall(r"[a-zA-Z]{4,}", text)
    stop = {
        "this", "that", "with", "from", "have", "will", "been", "into", "after", "over",
        "under", "they", "them", "your", "their", "about", "which", "there", "where",
        "when", "than", "because", "while", "before", "through", "within", "without",
        "what", "would", "could", "should", "more", "most", "some", "also", "like",
        "make", "take", "give", "find", "need", "want", "look", "come", "good", "well",
        "very", "just", "only", "even", "still", "such", "much", "many", "must", "may",
        "might", "shall", "can", "cannot", "every", "each", "both", "either", "neither"
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
    
    # Add description
    st.markdown("""
    **RSS + GDELT + Google News → Hybrid Search → FinBERT Sentiment Analysis**
    
    Enter a financial topic (e.g., 'US Tech', 'Germany', 'Oil Prices') to analyze sentiment across news sources.
    """)

    # SIDEBAR - Updated to match your screenshot
    with st.sidebar:
        st.header("📥 Update Sources")
        
        if st.button("🔄 Fetch RSS", use_container_width=True):
            with st.spinner("Fetching RSS feeds..."):
                n = fetch_rss_articles()
            st.success(f"✓ Added {n} RSS articles.")

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
            "📊 Max articles per expanded term",
            50, 500, MAX_ARTICLES_DEFAULT, 50,
            help="Maximum number of articles to retrieve for each expanded search term"
        )

    # MAIN AREA
    st.markdown("### 🔍 Search Topic")
    
    # Create a nicer search interface
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        query = st.text_input(
            "Enter topic ('US Tech', 'Germany', 'oil', etc.)",
            placeholder="e.g., US Tech, Germany, Oil Prices, Inflation...",
            label_visibility="collapsed"
        )
    
    analyze_clicked = st.button("🚀 Analyze", type="primary", use_container_width=True)

    if not analyze_clicked or not query.strip():
        st.info("👆 Enter a financial topic above and click **Analyze** to begin.")
        
        # Show database stats
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM articles")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM articles WHERE language = 'en'")
        english = cur.fetchone()[0]
        
        st.markdown("---")
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("📊 Total Articles", f"{total:,}")
        with col_stat2:
            st.metric("🇬🇧 English Articles", f"{english:,}")
        
        return

    # ========== MAIN ANALYSIS PIPELINE ==========
    
    # Step 1: Query Expansion
    with st.spinner("🔄 Expanding query for better coverage..."):
        expanded_terms = expand_query(query)
    
    st.markdown("### 🔍 Expanded terms used:")
    st.write(", ".join(expanded_terms))

    # Step 2: Fetch additional articles from GDELT/GNews
    total_added = 0
    if use_gdelt or use_gnews:
        with st.spinner("🌐 Fetching additional articles from external sources..."):
            for q in expanded_terms[:6]:  # Limit to first 6 terms
                if use_gdelt:
                    added = fetch_gdelt_articles(q, start_date, end_date)
                    total_added += added
                if use_gnews:
                    added = fetch_gnews_articles(q, start_date, end_date)
                    total_added += added
        
        if total_added > 0:
            st.success(f"✅ Added {total_added} extra articles for this topic.")

    # Step 3: Hybrid Search
    with st.spinner("🔎 Running hybrid search (semantic + keyword)..."):
        dfs = []
        for q in expanded_terms:
            part = hybrid_search(q, start_date, end_date, max_articles)
            if not part.empty:
                part["query_term"] = q
                dfs.append(part)

        if not dfs:
            st.error("❌ No relevant articles found for this topic in the selected date range.")
            return

        df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["link"])
        
        if df.empty:
            st.error("❌ No articles found after deduplication.")
            return
    
    st.success(f"✅ Hybrid search found **{len(df)}** candidate articles.")

    # Step 4: LLM Article Selection (optional)
    if len(df) > 50:  # Only use LLM if we have many articles
        with st.spinner("🤖 LLM selecting most relevant articles..."):
            df = llm_select_articles(query, df)
    else:
        st.success(f"✅ Keeping all {len(df)} articles (small dataset).")

    if df.empty:
        st.error("❌ No articles remained after filtering.")
        return

    # Step 5: Ensure language detection
    df = ensure_language(df)

    # Step 6: Sentiment Analysis
    with st.spinner("😊 Analyzing sentiment with FinBERT (English articles only)..."):
        df["sentiment_label"] = "not_scored"
        df["sentiment_score"] = np.nan

        mask_en = df["language"] == "en"
        df_en = df[mask_en].copy()

        if not df_en.empty:
            texts = []
            indices = []
            for idx, row in df_en.iterrows():
                text = f"{row.get('title', '')} {row.get('summary', '')} {row.get('content', '')}"
                if text.strip():
                    texts.append(text[:1500])  # Increased length
                    indices.append(idx)
            
            if texts:
                sents = finbert_sentiment(texts)
                for i, idx in enumerate(indices):
                    if i < len(sents):
                        df.at[idx, "sentiment_label"] = sents[i]["label"]
                        df.at[idx, "sentiment_score"] = sents[i]["score"]

    df["relevance"] = (df["similarity"] * 100).round(1)
    df["source_domain"] = df["source"].apply(
        lambda x: x.split("//")[-1].split("/")[0]
        if isinstance(x, str) and "//" in x
        else (str(x)[:30] if x else "unknown")
    )

    # Calculate sentiment index
    sent_index = calculate_sentiment_index(df)

    # ================== TABS ==================
    
    # Create tabs like in your screenshot
    tab_dash, tab_articles, tab_keywords, tab_download = st.tabs(
        ["📈 Dashboard", "📰 Articles", "🔑 Keywords", "📥 Download"]
    )

    # ----- Dashboard tab -----
    with tab_dash:
        st.subheader("📊 Sentiment Overview")
        
        # Sentiment counts
        counts = df["sentiment_label"].value_counts()
        total = int(counts.sum())
        pos = int(counts.get("positive", 0))
        neg = int(counts.get("negative", 0))
        neu = int(counts.get("neutral", 0))
        not_scored = int(counts.get("not_scored", 0))
        
        # Create metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📄 Total Articles", total)
        
        with col2:
            st.metric("🟢 Positive", pos)
        
        with col3:
            st.metric("🔴 Negative", neg)
        
        with col4:
            st.metric("🔵 Neutral", neu)
        
        with col5:
            # Sentiment index with direction
            if sent_index > 15:
                direction = "📈 Bullish"
                delta_color = "normal"
            elif sent_index < -15:
                direction = "📉 Bearish"
                delta_color = "inverse"
            else:
                direction = "➡️ Neutral"
                delta_color = "off"
            
            st.metric(
                "📊 Sentiment Index", 
                f"{sent_index:.1f}", 
                direction,
                delta_color=delta_color
            )
        
        st.markdown("---")
        
        # Visualizations
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### 📊 Sentiment Distribution")
            # Filter only scored sentiments
            scored_df = df[df["sentiment_label"].isin(["positive", "negative", "neutral"])]
            if not scored_df.empty:
                dist = scored_df["sentiment_label"].value_counts()
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
                fig_pie.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
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
                default=["positive", "negative", "neutral", "not_scored"],
            )
        
        with col_filter2:
            sort_by = st.selectbox(
                "Sort by",
                options=["Relevance (High to Low)", "Date (New to Old)", "Sentiment Score"],
                index=0
            )
        
        # Apply filters
        df_art = df[df["sentiment_label"].isin(sent_filter)].copy()
        
        # Apply sorting
        if sort_by == "Relevance (High to Low)":
            df_art = df_art.sort_values("relevance", ascending=False)
        elif sort_by == "Date (New to Old)":
            df_art = df_art.sort_values("published", ascending=False)
        elif sort_by == "Sentiment Score":
            df_art = df_art.sort_values("sentiment_score", ascending=False)
        
        # Display articles
        for idx, row in df_art.iterrows():
            icon_map = {
                "positive": "🟢",
                "negative": "🔴",
                "neutral": "🔵",
                "not_scored": "⚪",
            }
            icon = icon_map.get(row["sentiment_label"], "⚪")
            
            # Article card
            with st.container():
                col_title, col_link = st.columns([5, 1])
                with col_title:
                    st.markdown(f"**{icon} {row['title']}**")
                with col_link:
                    st.markdown(f"[📖 Read]({row['link']})", unsafe_allow_html=True)
                
                # Metadata row
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                with col_meta1:
                    st.caption(f"📰 {row['source_domain']}")
                with col_meta2:
                    st.caption(f"🎯 {row['relevance']:.1f}% relevant")
                with col_meta3:
                    if row["sentiment_label"] in ["positive", "negative", "neutral"]:
                        score_pct = row['sentiment_score'] * 100
                        st.caption(f"{row['sentiment_label'].title()} ({score_pct:.1f}%)")
                    else:
                        st.caption("Not scored")
                
                # Summary
                if row.get("summary"):
                    st.caption((row["summary"] or "")[:300] + ("…" if len(str(row.get("summary", ""))) > 300 else ""))
                
                st.markdown("---")

    # ----- Keywords tab -----
    with tab_keywords:
        st.subheader("🔑 Trending Keywords")
        
        # Extract from titles and summaries
        all_text = " ".join(df["title"].fillna("").tolist() + df["summary"].fillna("").tolist())
        keywords = extract_top_keywords(df["title"].tolist(), n=25)
        
        if keywords:
            # Display as word cloud style metrics
            st.markdown("#### Top Keywords")
            cols = st.columns(4)
            for i, (word, freq) in enumerate(keywords[:12]):
                with cols[i % 4]:
                    st.metric(word.capitalize(), freq)
            
            # Bar chart
            st.markdown("#### Keyword Frequency")
            kw_df = pd.DataFrame(keywords[:15], columns=["Keyword", "Frequency"])
            fig_kw = px.bar(
                kw_df,
                x="Frequency",
                y="Keyword",
                orientation="h",
                color="Frequency",
                color_continuous_scale="Viridis",
            )
            fig_kw.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("Not enough text data to extract keywords.")

    # ----- Download tab -----
    with tab_download:
        st.subheader("📥 Download Results")
        
        # Prepare data
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
        
        # Format dates
        dl_df["published"] = pd.to_datetime(dl_df["published"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Download button
        csv = dl_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name=f"sentiment_{query.replace(' ', '_')}_{dt.date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("#### 📋 Data Preview")
        st.dataframe(
            dl_df.head(20),
            use_container_width=True,
            hide_index=True,
            column_config={
                "link": st.column_config.LinkColumn("Link"),
                "sentiment_score": st.column_config.ProgressColumn(
                    "Sentiment Score",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                )
            }
        )


if __name__ == "__main__":
    run_app()
