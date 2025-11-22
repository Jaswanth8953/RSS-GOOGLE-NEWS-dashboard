import os
import json
import sqlite3
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

MIN_SIMILARITY = 0.30          # a bit lower to get more matches
MAX_ARTICLES_DEFAULT = 200
GDELT_MAX_RECORDS = 250
GNEWS_MAX_RESULTS = 200


# ========================== DB SETUP ==========================

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
    return conn


conn = get_connection()


# ========================== OPENAI CLIENT ==========================

# ========================== OPENAI CLIENT ==========================

@st.cache_resource
def get_openai_client():
    """
    Reads OPENAI_API_KEY from Streamlit Secrets.
    """
    api_key = st.secrets["openai"]["api_key"]
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
        results.append({"label": id2label[idx], "score": float(p[idx])})
    return results


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
    cur = conn.cursor()
    new_count = 0
    for url in RSS_FEEDS:
        try:
            arts = parse_rss(url)
        except Exception:
            continue
        for art in arts:
            try:
                cur.execute(
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
                if cur.rowcount > 0:
                    new_count += 1
            except Exception:
                pass
    conn.commit()
    return new_count


# ========================== GDELT INGESTION ==========================

def fetch_gdelt_articles(query: str, start: dt.date, end: dt.date) -> int:
    """
    Pulls articles from GDELT 2.0 DOC API for a given query and date range.
    """
    cur = conn.cursor()
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
            cur.execute(
                """
                INSERT OR IGNORE INTO articles
                (title, summary, published, link, source, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (title, summary, published, link, src, content),
            )
            if cur.rowcount > 0:
                new_count += 1
        except Exception:
            pass

    conn.commit()
    return new_count


# ========================== GOOGLE NEWS (GNews) ==========================

def fetch_gnews_articles(query: str, start: dt.date, end: dt.date) -> int:
    cur = conn.cursor()
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
            cur.execute(
                """
                INSERT OR IGNORE INTO articles
                (title, summary, published, link, source, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (title, desc, published, link, source, content),
            )
            if cur.rowcount > 0:
                new_count += 1
        except Exception:
            pass

    conn.commit()
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
    cur = conn.cursor()
    for _, row in df_ids_title_summary.iterrows():
        if row["embedding"] is None:
            text = (row["title"] or "") + " " + (row["summary"] or "")
            if not text.strip():
                continue
            emb = get_embedding(text)
            cur.execute(
                "UPDATE articles SET embedding = ? WHERE id = ?",
                (json.dumps(emb), int(row["id"])),
            )
    conn.commit()


def hybrid_search(
    query: str,
    start: dt.date,
    end: dt.date,
    top_k: int,
    min_sim: float = MIN_SIMILARITY,
) -> pd.DataFrame:
    """
    Hybrid keyword + semantic search within date range.
    """
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
        return [query] + expansions
    except Exception:
        return [query]


# ========================== LLM ARTICLE SELECTION (optional extra filter) ==========================

def llm_select_articles(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Let the LLM decide which of the already-matched articles
    are truly relevant to the user's topic.
    We send only (id, title, summary) to the LLM to keep it light.
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

        today = dt.date.today()
        default_start = today - dt.timedelta(days=14)
        start_date = st.date_input("Start date", default_start)
        end_date = st.date_input("End date", today)

        st.subheader("Extra sources for this topic")
        use_gdelt = st.checkbox("Include GDELT", True)
        use_gnews = st.checkbox("Include Google News", True)

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
            for q in expanded_queries:
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

    # Optional LLM article filter / selector
    with st.spinner("Letting LLM choose the most relevant subset of articles…"):
        df = llm_select_articles(query, df)

    if df.empty:
        st.error("After LLM filtering, no strongly relevant articles remained.")
        return

    st.success(f"✅ LLM selected **{len(df)}** final articles for analysis.")

    # SENTIMENT
    with st.spinner("Running FinBERT sentiment classification…"):
        texts = df["content"].fillna(df["summary"]).tolist()
        sents = finbert_sentiment(texts)

    df["sentiment_label"] = [s["label"] for s in sents]
    df["sentiment_score"] = [s["score"] for s in sents]
    df["relevance"] = (df["similarity"] * 100).round(1)
    df["source_domain"] = df["source"].fillna("").replace("", "unknown")

    sentiment_index = calculate_sentiment_index(df)

    # TABS
    tab_dash, tab_articles, tab_keywords, tab_download = st.tabs(
        ["📊 Dashboard", "📰 Articles", "🔑 Keywords", "📥 Download"]
    )

    # DASHBOARD TAB
    with tab_dash:
        st.subheader("Executive Sentiment Overview")

        counts = df["sentiment_label"].value_counts()
        total = int(counts.sum())
        pos = int(counts.get("positive", 0))
        neg = int(counts.get("negative", 0))
        neu = int(counts.get("neutral", 0))

        c_a, c_b, c_c, c_d, c_e = st.columns(5)
        c_a.metric("Total Articles", total)
        c_b.metric("Positive", pos, f"{(pos/total*100):.1f}%" if total else "0.0%")
        c_c.metric("Negative", neg, f"{(neg/total*100):.1f}%" if total else "0.0%")
        c_d.metric("Neutral", neu, f"{(neu/total*100):.1f}%" if total else "0.0%")

        direction = (
            "Bullish" if sentiment_index > 0 else "Bearish" if sentiment_index < 0 else "Neutral"
        )
        c_e.metric("Sentiment Index", f"{sentiment_index:.1f}", direction)

        st.markdown("---")

        # Pie chart
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

        # Gauge chart
        with r1c2:
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

        # Source + confidence
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("#### Articles by Source")
            src_counts = df["source_domain"].value_counts()
            fig_src = px.bar(
                x=src_counts.values,
                y=src_counts.index,
                orientation="h",
                labels={"x": "Count", "y": "Source"},
                color=src_counts.values,
                color_continuous_scale="Viridis",
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
        sent_filter = st.multiselect(
            "Filter by sentiment",
            options=["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"],
        )
        filtered = df[df["sentiment_label"].isin(sent_filter)].copy()
        filtered = filtered.sort_values(
            ["relevance", "sentiment_score"], ascending=[False, False]
        )
        for _, row in filtered.iterrows():
            icon = {"positive": "🟢", "negative": "🔴", "neutral": "🔵"}[row["sentiment_label"]]
            st.markdown(f"**{icon} {row['title']}**")
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            c1.caption(row["source_domain"])
            c2.caption(f"Relevance: {row['relevance']:.1f}%")
            c3.caption(f"Sentiment conf: {row['sentiment_score']*100:.1f}%")
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
            col_a, col_b = st.columns(2)
            with col_a:
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
            with col_b:
                fig_tree = px.treemap(
                    kw_df,
                    path=["Keyword"],
                    values="Frequency",
                    color="Frequency",
                    color_continuous_scale="Viridis",
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


if __name__ == "__main__":
    run_app()

    
    
    
 
      
   
    
       
