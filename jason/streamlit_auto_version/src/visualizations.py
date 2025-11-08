import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    # Convert date column to datetime if it's not already
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['created_utc']).dt.date
    
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
    # Convert date column to datetime if it's not already
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['created_utc']).dt.date
    
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


def calculate_metrics(df):
    """Calculate key metrics for display"""
    total_posts = len(df)
    positive_pct = (df['sentiment_prediction'] == 'POS').mean() * 100
    neutral_pct = (df['sentiment_prediction'] == 'NEU').mean() * 100
    negative_pct = (df['sentiment_prediction'] == 'NEG').mean() * 100
    
    return {
        'total_posts': total_posts,
        'positive_pct': positive_pct,
        'neutral_pct': neutral_pct,
        'negative_pct': negative_pct
    }
