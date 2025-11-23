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
    "https://feeds.reuters.com/reuters/technologyNews",  # Added tech-specific
    "https://feeds.reuters.com/reuters/companiesNews",   # Added companies
    "https://rss.bloomberg.com/markets/news.rss",
    "https://www.wsj.com/news/markets?mod=rss_markets_main",
]

OPENAI_EMBED_MODEL = "text-embedding-3-small"
FINBERT_MODEL = "ProsusAI/finbert"

# Tuned parameters
MIN_SIMILARITY = 0.25  # Lower threshold for broader matching
MAX_ARTICLES_DEFAULT = 200
GDELT_MAX_RECORDS = 100
GNEWS_MAX_RESULTS = 100

# ========================== DATABASE SETUP ==========================

def init_database():
    """Initialize database with proper settings"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size = -10000;")
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            summary TEXT,
            published TEXT,
            link TEXT UNIQUE,
            source TEXT,
            content TEXT,
            embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index for better performance
    conn.execute("CREATE INDEX IF NOT EXISTS idx_published ON articles(published)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding ON articles(embedding IS NOT NULL)")
    conn.commit()
    return conn

conn = init_database()

def safe_db_execute(query: str, params: tuple = ()):
    """Safe database execution with retry logic"""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            conn.commit()
            return cur
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                time.sleep(0.2)
                continue
            raise
    raise Exception("Database operation failed after retries")

# ========================== OPENAI SETUP ==========================

@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client"""
    api_key = st.secrets["openai"]["api_key"]
    return OpenAI(api_key=api_key)

# ========================== IMPROVED QUERY EXPANSION ==========================

def expand_query_advanced(query: str) -> List[str]:
    """
    Advanced query expansion with multiple strategies to ensure balanced results
    """
    client = get_openai_client()
    
    # Pre-defined expansions for common queries to ensure consistency
    predefined_expansions = {
        "us tech": [
            "US technology sector", "American tech companies", "Silicon Valley", 
            "tech stocks", "NASDAQ composite", "US software industry", 
            "US hardware manufacturers", "American innovation", "digital technology USA",
            "US tech giants", "technology sector United States", "US computer industry"
        ],
        "nvidia": [
            "NVDA stock", "GPU manufacturer", "AI chips", "semiconductor company",
            "graphics processing units", "Jensen Huang", "CUDA technology",
            "GeForce RTX", "data center chips", "gaming GPUs", "artificial intelligence hardware",
            "chip stocks", "semiconductor stocks"
        ],
        "technology": [
            "tech industry", "information technology", "digital technology",
            "computer technology", "tech sector", "innovation", "digital transformation",
            "emerging technologies", "tech companies", "software development",
            "hardware manufacturing", "IT sector"
        ],
        "technology companies": [
            "tech firms", "software companies", "hardware manufacturers",
            "IT corporations", "technology stocks", "digital companies",
            "tech enterprises", "computer companies", "internet companies"
        ]
    }
    
    query_lower = query.lower().strip()
    
    # Use predefined expansions if available
    if query_lower in predefined_expansions:
        base_terms = predefined_expansions[query_lower]
    else:
        # Generate expansions for other queries
        try:
            prompt = f"""
            Expand this financial/news search query: "{query}"
            
            Generate 10-15 alternative terms that capture:
            - Exact synonyms and related terms
            - Broader industry categories
            - Specific companies/products in that space
            - Financial/market terminology
            - Common abbreviations and variations
            
            Return ONLY a JSON array of strings.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial news search expert. Generate comprehensive search term expansions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            base_terms = json.loads(content)
        except:
            # Fallback basic expansion
            base_terms = [query]
    
    # Always include the original query and ensure diversity
    all_terms = [query] + base_terms
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in all_terms:
        term_lower = term.lower()
        if term_lower not in seen and len(term.strip()) > 0:
            unique_terms.append(term)
            seen.add(term_lower)
    
    return unique_terms[:15]  # Limit to 15 terms

# ========================== ENHANCED SEMANTIC SEARCH ==========================

def get_embedding(text: str) -> List[float]:
    """Get embedding for text"""
    client = get_openai_client()
    text = text.replace("\n", " ").strip()
    if not text:
        return [0.0] * 1536  # Default embedding dimension
    
    response = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity"""
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def hybrid_semantic_search(
    queries: List[str], 
    start_date: dt.date, 
    end_date: dt.date, 
    top_k: int = 200
) -> pd.DataFrame:
    """
    Enhanced hybrid search with better query processing
    """
    # Load articles from date range
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, summary, published, link, source, content, embedding FROM articles WHERE date(published) BETWEEN ? AND ?",
        (start_date.isoformat(), end_date.isoformat())
    )
    
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows, columns=[
        "id", "title", "summary", "published", "link", "source", "content", "embedding"
    ])
    
    # Ensure embeddings exist
    missing_embeddings = df[df["embedding"].isna() | (df["embedding"] == "") | (df["embedding"] == "NULL")]
    for _, row in missing_embeddings.iterrows():
        text = f"{row['title']} {row['summary']}".strip()
        if text:
            embedding = get_embedding(text)
            safe_db_execute(
                "UPDATE articles SET embedding = ? WHERE id = ?",
                (json.dumps(embedding), row["id"])
            )
    
    # Reload with embeddings
    cur.execute(
        "SELECT id, title, summary, published, link, source, content, embedding FROM articles WHERE date(published) BETWEEN ? AND ? AND embedding IS NOT NULL",
        (start_date.isoformat(), end_date.isoformat())
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows, columns=[
        "id", "title", "summary", "published", "link", "source", "content", "embedding"
    ])
    
    # Get query embeddings for all expanded terms
    query_embeddings = []
    for query in queries:
        if query.strip():
            query_embeddings.append(get_embedding(query))
    
    if not query_embeddings:
        return pd.DataFrame()
    
    # Average all query embeddings
    avg_query_embedding = np.mean(query_embeddings, axis=0)
    
    # Calculate similarities
    similarities = []
    for _, row in df.iterrows():
        try:
            article_embedding = np.array(json.loads(row["embedding"]))
            similarity = cosine_similarity(avg_query_embedding, article_embedding)
            similarities.append(similarity)
        except:
            similarities.append(0.0)
    
    df["similarity"] = similarities
    
    # Keyword matching boost
    query_terms = [q.lower() for q in queries if q.strip()]
    
    def keyword_match_score(text):
        if not text or not query_terms:
            return 0
        text_lower = str(text).lower()
        matches = sum(1 for term in query_terms if term in text_lower)
        return matches * 0.1  # Small boost for keyword matches
    
    df["keyword_boost"] = df["title"].fillna("").apply(keyword_match_score) + \
                         df["summary"].fillna("").apply(keyword_match_score) + \
                         df["content"].fillna("").apply(keyword_match_score)
    
    df["combined_score"] = df["similarity"] + df["keyword_boost"]
    
    # Filter and return top results
    filtered = df[df["combined_score"] >= MIN_SIMILARITY]
    if filtered.empty:
        # Fallback: return top by similarity regardless of threshold
        filtered = df.nlargest(top_k, "combined_score")
    else:
        filtered = filtered.nlargest(top_k, "combined_score")
    
    return filtered

# ========================== DATA INGESTION ==========================

def fetch_rss_articles() -> int:
    """Fetch articles from RSS feeds"""
    new_count = 0
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:  # Limit per feed
                title = entry.get("title", "").strip()
                summary = entry.get("summary", "").strip()
                link = entry.get("link", "")
                
                # Parse publication date
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = dt.datetime(*entry.published_parsed[:6])
                else:
                    pub_date = dt.datetime.utcnow()
                
                content = summary
                if "content" in entry and entry.content:
                    content = entry.content[0].get("value", summary)
                
                # Insert article
                try:
                    safe_db_execute(
                        "INSERT OR IGNORE INTO articles (title, summary, published, link, source, content) VALUES (?, ?, ?, ?, ?, ?)",
                        (title, summary, pub_date.isoformat(), link, url, content)
                    )
                    new_count += 1
                except:
                    continue
                    
        except Exception as e:
            continue
    
    return new_count

def fetch_external_articles(query: str, start_date: dt.date, end_date: dt.date) -> int:
    """Fetch additional articles from external sources"""
    new_count = 0
    
    # Google News
    try:
        google_news = GNews(language="en", max_results=50)
        google_news.start_date = (start_date.year, start_date.month, start_date.day)
        google_news.end_date = (end_date.year, end_date.month, end_date.day)
        
        news_results = google_news.get_news(query)
        for article in news_results:
            try:
                safe_db_execute(
                    "INSERT OR IGNORE INTO articles (title, summary, published, link, source, content) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        article.get("title", ""),
                        article.get("description", ""),
                        article.get("published date", dt.datetime.utcnow().isoformat()),
                        article.get("url", ""),
                        "Google News",
                        article.get("description", "")
                    )
                )
                new_count += 1
            except:
                continue
    except:
        pass
    
    return new_count

# ========================== SENTIMENT ANALYSIS ==========================

@st.cache_resource
def load_finbert():
    """Load FinBERT model for sentiment analysis"""
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

def analyze_sentiment_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """Analyze sentiment for a batch of texts"""
    if not texts:
        return []
    
    tokenizer, model, device = load_finbert()
    
    # Fallback if model not available
    if tokenizer is None:
        return [{"label": "neutral", "score": 0.8} for _ in texts]
    
    try:
        # Process in batches
        batch_size = 32
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Clean and prepare texts
            cleaned_texts = []
            for text in batch_texts:
                if not text or pd.isna(text):
                    cleaned_texts.append("No content")
                else:
                    cleaned_text = ' '.join(str(text).split())[:400]
                    cleaned_texts.append(cleaned_text)
            
            # Tokenize and predict
            encodings = tokenizer(
                cleaned_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**encodings)
                probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            
            # Convert to labels
            id2label = {0: "negative", 1: "neutral", 2: "positive"}
            for prob in probabilities:
                label_idx = np.argmax(prob)
                results.append({
                    "label": id2label[label_idx],
                    "score": float(prob[label_idx])
                })
        
        return results
        
    except Exception as e:
        st.error(f"Sentiment analysis error: {e}")
        # Return neutral as fallback
        return [{"label": "neutral", "score": 0.8} for _ in texts]

# ========================== DASHBOARD COMPONENTS ==========================

def create_sentiment_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate sentiment metrics"""
    total = len(df)
    if total == 0:
        return {
            "total": 0,
            "positive": 0, "positive_pct": 0,
            "negative": 0, "negative_pct": 0, 
            "neutral": 0, "neutral_pct": 0,
            "sentiment_index": 0.0
        }
    
    sentiment_counts = df["sentiment_label"].value_counts()
    positive = sentiment_counts.get("positive", 0)
    negative = sentiment_counts.get("negative", 0)
    neutral = sentiment_counts.get("neutral", 0)
    
    sentiment_index = ((positive - negative) / total) * 100
    
    return {
        "total": total,
        "positive": positive, "positive_pct": (positive / total) * 100,
        "negative": negative, "negative_pct": (negative / total) * 100,
        "neutral": neutral, "neutral_pct": (neutral / total) * 100,
        "sentiment_index": sentiment_index
    }

def create_sentiment_charts(metrics: Dict[str, Any], df: pd.DataFrame):
    """Create sentiment visualization charts"""
    # Pie chart
    fig_pie = px.pie(
        names=["Positive", "Negative", "Neutral"],
        values=[metrics["positive"], metrics["negative"], metrics["neutral"]],
        color=["Positive", "Negative", "Neutral"],
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"},
        hole=0.4
    )
    fig_pie.update_layout(title="Sentiment Distribution")
    
    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=metrics["sentiment_index"],
        title={"text": "Sentiment Index"},
        delta={"reference": 0},
        gauge={
            "axis": {"range": [-100, 100]},
            "bar": {"color": "#2ecc71" if metrics["sentiment_index"] > 0 else "#e74c3c"},
            "steps": [
                {"range": [-100, 0], "color": "lightgray"},
                {"range": [0, 100], "color": "lightgreen"}
            ]
        }
    ))
    fig_gauge.update_layout(height=300)
    
    return fig_pie, fig_gauge

# ========================== STREAMLIT APP ==========================

def main():
    st.set_page_config(
        page_title="Financial News Sentiment Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Financial News Sentiment Dashboard")
    st.markdown("""
    **SMART QUERY SYSTEM** - Understands meaning, not just keywords  
    **LLM-BASED SELECTION** - Semantic article matching  
    **FINBERT SENTIMENT** - Financial-specific sentiment analysis  
    **FUSED OUTPUT** - Comprehensive results with trends and insights
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Data Management")
        
        if st.button("🔄 Update RSS Feeds", use_container_width=True):
            with st.spinner("Fetching latest articles..."):
                new_count = fetch_rss_articles()
                st.success(f"Added {new_count} new articles")
        
        st.header("Search Settings")
        
        today = dt.date.today()
        start_date = st.date_input("Start Date", today - dt.timedelta(days=30))
        end_date = st.date_input("End Date", today)
        
        use_external_sources = st.checkbox("Include External Sources", value=True)
        max_articles = st.slider("Max Articles", 50, 500, 200)
    
    # Main search interface
    st.subheader("🔍 Semantic Search")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input(
            "Enter your topic:",
            placeholder="e.g., US Tech, Nvidia, European textiles, AI semiconductor...",
            key="search_query"
        )
    with col2:
        search_clicked = st.button("🚀 Analyze", type="primary", use_container_width=True)
    with col3:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_clicked:
        st.rerun()
    
    if not search_clicked or not query.strip():
        st.info("💡 Enter a search topic and click 'Analyze' to see sentiment insights")
        return
    
    # Process query
    with st.spinner("🔍 Expanding search terms..."):
        expanded_queries = expand_query_advanced(query)
    
    st.success(f"**Expanded to {len(expanded_queries)} search terms**")
    st.write("📋 Search terms:", ", ".join(expanded_queries))
    
    # Fetch additional data if requested
    if use_external_sources:
        with st.spinner("🌐 Fetching additional articles..."):
            for search_term in expanded_queries[:5]:  # Limit to first 5 terms
                fetch_external_articles(search_term, start_date, end_date)
    
    # Perform semantic search
    with st.spinner("🎯 Finding relevant articles..."):
        results_df = hybrid_semantic_search(expanded_queries, start_date, end_date, max_articles)
    
    if results_df.empty:
        st.error("❌ No relevant articles found. Try broadening your search or date range.")
        return
    
    st.success(f"✅ Found {len(results_df)} relevant articles")
    
    # Sentiment analysis
    with st.spinner("😊 Analyzing sentiment with FinBERT..."):
        # Prepare texts for analysis
        analysis_texts = []
        for _, row in results_df.iterrows():
            text = f"{row['title']} {row['summary']}".strip()
            analysis_texts.append(text)
        
        sentiment_results = analyze_sentiment_batch(analysis_texts)
        
        # Add sentiment to dataframe
        results_df["sentiment_label"] = [r["label"] for r in sentiment_results]
        results_df["sentiment_score"] = [r["score"] for r in sentiment_results]
        results_df["relevance_score"] = (results_df["combined_score"] * 100).round(1)
    
    # Calculate metrics
    metrics = create_sentiment_metrics(results_df)
    
    # Display results in tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", 
        "📰 Articles", 
        "🔍 Analysis", 
        "📥 Export"
    ])
    
    with tab1:
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Articles", metrics["total"])
        with col2:
            st.metric("Positive", f"{metrics['positive']} ({metrics['positive_pct']:.1f}%)")
        with col3:
            st.metric("Negative", f"{metrics['negative']} ({metrics['negative_pct']:.1f}%)")
        with col4:
            st.metric("Neutral", f"{metrics['neutral']} ({metrics['neutral_pct']:.1f}%)")
        with col5:
            direction = "Bullish" if metrics["sentiment_index"] > 0 else "Bearish"
            st.metric("Sentiment Index", f"{metrics['sentiment_index']:.1f}", direction)
        
        st.markdown("---")
        
        # Charts
        fig_pie, fig_gauge = create_sentiment_charts(metrics, results_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Additional insights
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Sentiment Confidence")
            avg_confidence = results_df.groupby("sentiment_label")["sentiment_score"].mean()
            fig_conf = px.bar(
                x=avg_confidence.index,
                y=avg_confidence.values,
                color=avg_confidence.index,
                color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
            )
            fig_conf.update_layout(xaxis_title="Sentiment", yaxis_title="Average Confidence")
            st.plotly_chart(fig_conf, use_container_width=True)
        
        with col2:
            st.subheader("🌐 Top Sources")
            source_counts = results_df["source"].value_counts().head(10)
            fig_sources = px.bar(
                x=source_counts.values,
                y=source_counts.index,
                orientation='h',
                title="Articles by Source"
            )
            st.plotly_chart(fig_sources, use_container_width=True)
    
    with tab2:
        st.subheader("📋 Article List")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            sentiment_filter = st.multiselect(
                "Filter by sentiment:",
                options=["positive", "negative", "neutral"],
                default=["positive", "negative", "neutral"]
            )
        with col2:
            sort_by = st.selectbox(
                "Sort by:",
                options=["Relevance", "Date", "Sentiment Confidence"],
                index=0
            )
        
        # Filter and sort
        filtered_df = results_df[results_df["sentiment_label"].isin(sentiment_filter)].copy()
        
        if sort_by == "Relevance":
            filtered_df = filtered_df.sort_values("relevance_score", ascending=False)
        elif sort_by == "Date":
            filtered_df = filtered_df.sort_values("published", ascending=False)
        else:
            filtered_df = filtered_df.sort_values("sentiment_score", ascending=False)
        
        # Display articles
        for _, article in filtered_df.iterrows():
            sentiment_icon = {
                "positive": "🟢", 
                "negative": "🔴", 
                "neutral": "🔵"
            }[article["sentiment_label"]]
            
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{sentiment_icon} {article['title']}**")
                    if article["summary"]:
                        st.caption(article["summary"][:200] + "...")
                with col2:
                    st.caption(f"Relevance: {article['relevance_score']}%")
                    st.caption(f"Sentiment: {article['sentiment_label']}")
                    st.caption(f"Confidence: {article['sentiment_score']*100:.1f}%")
                    st.markdown(f"[📖 Read]({article['link']})")
                
                st.markdown("---")
    
    with tab3:
        st.subheader("🔍 Detailed Analysis")
        
        # Keyword analysis
        st.write("### 📊 Trending Keywords")
        all_titles = " ".join(results_df["title"].fillna("").tolist())
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_titles.lower())
        
        stop_words = {"this", "that", "with", "from", "will", "have", "been", "their", "about", "would"}
        filtered_words = [w for w in words if w not in stop_words]
        
        word_freq = Counter(filtered_words)
        top_keywords = word_freq.most_common(15)
        
        if top_keywords:
            keywords_df = pd.DataFrame(top_keywords, columns=["Keyword", "Frequency"])
            fig_keywords = px.bar(
                keywords_df,
                x="Frequency",
                y="Keyword",
                orientation='h',
                title="Most Frequent Keywords"
            )
            st.plotly_chart(fig_keywords, use_container_width=True)
        else:
            st.info("No significant keywords found")
        
        # Sentiment over time (if enough data)
        if len(results_df) > 5:
            st.write("### 📅 Sentiment Trend")
            results_df["published_date"] = pd.to_datetime(results_df["published"]).dt.date
            daily_sentiment = results_df.groupby("published_date")["sentiment_label"].value_counts().unstack(fill_value=0)
            
            if not daily_sentiment.empty:
                fig_trend = px.line(
                    daily_sentiment,
                    title="Daily Sentiment Distribution"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab4:
        st.subheader("📥 Export Results")
        
        # Prepare download data
        export_df = results_df[[
            "title", "summary", "published", "link", "source", 
            "sentiment_label", "sentiment_score", "relevance_score"
        ]].copy()
        
        st.write(f"**Export {len(export_df)} articles**")
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                "💾 Download CSV",
                csv_data,
                file_name=f"sentiment_analysis_{query.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = export_df.to_json(orient="records", indent=2)
            st.download_button(
                "📄 Download JSON",
                json_data,
                file_name=f"sentiment_analysis_{query.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.write("### Data Preview")
        st.dataframe(export_df.head(10), use_container_width=True)

if __name__ == "__main__":
    main()
