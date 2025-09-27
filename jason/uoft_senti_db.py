import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="UofT Reddit Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load and preprocess the Reddit data"""
    # Get the path to the CSV file
    today = datetime.now().strftime('%Y%m')
    csv_path = Path(f"monthly_top100/{today}/top_100_reddits_{today}.csv")

    if not csv_path.exists():
        st.error(f"Data file not found at {csv_path}")
        return None
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Convert created_utc to datetime
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    df['date'] = df['created_utc'].dt.date
    df['hour'] = df['created_utc'].dt.hour
    
    # Parse sentiment probabilities (convert string to dict)
    def parse_sentiment_prob(prob_str):
        try:
            return json.loads(prob_str.replace("'", '"'))
        except:
            return {"NEG": 0, "NEU": 0, "POS": 0}
    
    df['sentiment_probs'] = df['sentiment_prob'].apply(parse_sentiment_prob)
    
    # Extract individual sentiment probabilities
    df['neg_prob'] = df['sentiment_probs'].apply(lambda x: x.get('NEG', 0))
    df['neu_prob'] = df['sentiment_probs'].apply(lambda x: x.get('NEU', 0))
    df['pos_prob'] = df['sentiment_probs'].apply(lambda x: x.get('POS', 0))
    
    # Clean body text for length analysis
    df['body_length'] = df['body'].fillna('').str.len()
    df['word_count'] = df['body'].fillna('').str.split().str.len()
    
    return df

def plot_sentiment_distribution(df):
    """Create a pie chart showing sentiment distribution"""
    sentiment_counts = df['sentiment_prediction'].value_counts()
    fig_pie = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color_discrete_map={'POS': '#2E8B57', 'NEU': '#4682B4', 'NEG': '#DC143C'}
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    return fig_pie

def plot_sentiment_counts(df):
    """Create a bar chart showing sentiment counts"""
    sentiment_counts = df['sentiment_prediction'].value_counts()
    fig_bar = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        color=sentiment_counts.index,
        color_discrete_map={'POS': '#2E8B57', 'NEU': '#4682B4', 'NEG': '#DC143C'}
    )
    fig_bar.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Number of Posts/Comments"
    )
    return fig_bar

def plot_sentiment_score_distribution(df):
    """Create a box plot showing sentiment score distribution"""
    fig_box = px.box(
        df, 
        x='sentiment_prediction', 
        y='sentiment_score',
        color='sentiment_prediction',
        color_discrete_map={'POS': '#2E8B57', 'NEU': '#4682B4', 'NEG': '#DC143C'}
    )
    return fig_box

def plot_sentiment_over_time(df):
    """Create a time series plot showing sentiment trends over time"""
    daily_sentiment = df.groupby(['date', 'sentiment_prediction']).size().unstack(fill_value=0)
    
    fig_time = go.Figure()
    for sentiment in ['POS', 'NEU', 'NEG']:
        if sentiment in daily_sentiment.columns:
            color_map = {'POS': '#2E8B57', 'NEU': '#4682B4', 'NEG': '#DC143C'}
            fig_time.add_trace(go.Scatter(
                x=daily_sentiment.index, 
                y=daily_sentiment[sentiment],
                name=sentiment,
                line=dict(color=color_map[sentiment]),
                mode='lines+markers'
            ))
    
    fig_time.update_layout(
        xaxis_title="Date", 
        yaxis_title="Number of Posts/Comments",
        hovermode='x unified'
    )
    return fig_time

def plot_sentiment_score_over_time(df):
    """Create a time series plot showing overall average sentiment scores over time"""
    daily_avg_scores = df.groupby('date')['sentiment_score'].mean()
    
    fig_score_time = go.Figure()
    fig_score_time.add_trace(go.Scatter(
        x=daily_avg_scores.index, 
        y=daily_avg_scores.values,
        name="Average Sentiment Score",
        line=dict(color='#4682B4', width=3),
        mode='lines+markers'
    ))
    
    fig_score_time.update_layout(
        xaxis_title="Date", 
        yaxis_title="Average Sentiment Score",
        hovermode='x unified',
        yaxis=dict(
            tickvals=[-1, 0, 1],
            ticktext=['NEG (-1)', 'NEU (0)', 'POS (1)'],
            range=[-1.1, 1.1]
        )
    )
    return fig_score_time

def get_top_posts(df, n=10):
    """Get top N posts by score"""
    return df.nlargest(n, 'score')[['submission_title', 'author', 'score', 'sentiment_prediction', 'type']]

def main():
    st.title("📊 UofT Reddit Sentiment Dashboard")
    st.markdown("### Analyzing sentiment and engagement patterns from r/UofT top posts")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Posts/Comments", len(df))
        
    with col2:
        positive_pct = (df['sentiment_prediction'] == 'POS').mean() * 100
        st.metric("Positive Sentiment %", f"{positive_pct:.1f}%")
    with col3:
        neutral_pct = (df['sentiment_prediction'] == 'NEU').mean() * 100
        st.metric("Neutral Sentiment %", f"{neutral_pct:.1f}%")
    with col4:
        negative_pct = (df['sentiment_prediction'] == 'NEG').mean() * 100
        st.metric("Negative Sentiment %", f"{negative_pct:.1f}%")
    
    # Main visualizations
    st.markdown("---")
    
    # Row 1: Time Series Analysis
    st.subheader("📅 Sentiment Over Time")
    fig_time = plot_sentiment_over_time(df)
    st.plotly_chart(fig_time, use_container_width=True)
    
    st.subheader("📊 Sentiment Score Over Time")
    fig_score_time = plot_sentiment_score_over_time(df)
    st.plotly_chart(fig_score_time, use_container_width=True)
    
    # Row 2: Sentiment Analysis
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("� Sentiment Distribution")
        fig_pie = plot_sentiment_distribution(df)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📊 Sentiment Counts")
        fig_bar = plot_sentiment_counts(df)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Row 3: Sentiment Analysis Details
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Score Distribution")
        fig_box = plot_sentiment_score_distribution(df)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        st.subheader("🔥 Top Posts by Score")
        top_posts = get_top_posts(df)
        st.dataframe(top_posts, use_container_width=True)
    


if __name__ == "__main__":
    main()