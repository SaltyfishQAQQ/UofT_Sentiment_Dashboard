import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_collector import collect_reddit_data
from src.data_processor import process_reddit_data
from src.visualizations import (
    plot_sentiment_distribution,
    plot_sentiment_counts,
    plot_sentiment_score_distribution,
    plot_sentiment_over_time,
    plot_sentiment_score_over_time,
    get_top_posts,
    calculate_metrics
)

# Configure Streamlit page
st.set_page_config(
    page_title="UofT Reddit Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_existing_data():
    """Load the most recent processed data if available"""
    try:
        date = datetime.now().strftime('%Y%m')
        csv_path = Path(f"monthly_top100/{date}/top_100_reddits_{date}.csv")

        if not csv_path.exists():
            return None
        
        # Load the data
        df = pd.read_csv(csv_path)
        
        # Convert created_utc to datetime if needed
        if df['created_utc'].dtype == 'object':
            df['created_utc'] = pd.to_datetime(df['created_utc'])
        
        # Add date and hour columns if not present
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['created_utc']).dt.date
        if 'hour' not in df.columns:
            df['hour'] = pd.to_datetime(df['created_utc']).dt.hour
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def run_full_pipeline(subreddit='UofT', limit=100, time_filter='month'):
    """Run the complete data collection and processing pipeline"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Data Collection
        status_text.text("🔄 Step 1/3: Collecting Reddit data...")
        progress_bar.progress(10)
        
        master_df, master_csv_path = collect_reddit_data(
            subreddit_name=subreddit,
            limit=limit,
            time_filter=time_filter
        )
        
        progress_bar.progress(40)
        status_text.text(f"✅ Collected {len(master_df)} posts/comments")
        
        # Step 2: Data Processing
        status_text.text("🔄 Step 2/3: Processing and analyzing sentiment...")
        progress_bar.progress(50)
        
        date = datetime.now().strftime('%Y%m')
        folder_path = Path(f'monthly_top100/{date}')
        
        processed_df = process_reddit_data(folder_path)
        
        progress_bar.progress(90)
        status_text.text(f"✅ Processed {len(processed_df)} posts/comments")
        
        # Step 3: Complete
        progress_bar.progress(100)
        status_text.text("✅ Pipeline complete! Loading visualizations...")
        
        return processed_df
        
    except Exception as e:
        status_text.text(f"❌ Error: {str(e)}")
        st.error(f"Pipeline failed: {str(e)}")
        return None


def display_dashboard(df):
    """Display the main dashboard with all visualizations"""
    if df is None or len(df) == 0:
        st.warning("No data available. Please run the data collection pipeline.")
        return
    
    # Display key metrics
    st.markdown("### 📈 Key Metrics")
    metrics = calculate_metrics(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Posts/Comments", metrics['total_posts'])
        
    with col2:
        st.metric("Positive Sentiment %", f"{metrics['positive_pct']:.1f}%")
    
    with col3:
        st.metric("Neutral Sentiment %", f"{metrics['neutral_pct']:.1f}%")
    
    with col4:
        st.metric("Negative Sentiment %", f"{metrics['negative_pct']:.1f}%")
    
    # Main visualizations
    st.markdown("---")
    
    # Row 1: Time Series Analysis
    st.subheader("📅 Sentiment Over Time")
    fig_time = plot_sentiment_over_time(df)
    st.plotly_chart(fig_time, use_container_width=True)
    
    st.subheader("📊 Sentiment Score Over Time")
    fig_score_time = plot_sentiment_score_over_time(df)
    st.plotly_chart(fig_score_time, use_container_width=True)
    
    # Row 2: Sentiment Distribution
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥧 Sentiment Distribution")
        fig_pie = plot_sentiment_distribution(df)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📊 Sentiment Counts")
        fig_bar = plot_sentiment_counts(df)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Row 3: Detailed Analysis
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Score Distribution")
        fig_box = plot_sentiment_score_distribution(df)
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("*Box plot showing the distribution of sentiment scores across positive, neutral, and negative posts/comments.*")
    
    with col2:
        st.subheader("🔥 Top Posts by Score")
        top_posts = get_top_posts(df)
        st.dataframe(top_posts, use_container_width=True)


def main():
    st.title("📊 UofT Reddit Sentiment Dashboard")
    st.markdown("### Analyzing sentiment and engagement patterns from r/UofT top posts")
    
    # Sidebar controls
    st.sidebar.title("⚙️ Controls")
    
    # Pipeline configuration
    st.sidebar.markdown("### 🔧 Pipeline Configuration")
    subreddit = st.sidebar.text_input("Subreddit", value="UofT")
    limit = st.sidebar.number_input("Number of posts to collect", min_value=10, max_value=100, value=100, step=10)
    time_filter = st.sidebar.selectbox("Time filter", options=['day', 'week', 'month', 'year', 'all'], index=2)
    
    # Run pipeline button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run Full Pipeline", type="primary", use_container_width=True):
        st.session_state['running_pipeline'] = True
        st.session_state['data'] = None
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        """
        This dashboard collects, processes, and visualizes sentiment data from Reddit.
        
        **Pipeline Steps:**
        1. 🔍 Data Collection - Fetch posts from Reddit
        2. ⚙️ Data Processing - Clean and analyze sentiment
        3. 📊 Visualization - Display insights
        
        Click **Run Full Pipeline** to start!
        """
    )
    
    # Run pipeline if button was clicked
    if st.session_state.get('running_pipeline', False):
        with st.spinner("Running pipeline..."):
            df = run_full_pipeline(subreddit, limit, time_filter)
            if df is not None:
                st.session_state['data'] = df
                st.session_state['running_pipeline'] = False
                st.success("✅ Pipeline completed successfully!")
                st.rerun()
    
    # Try to load existing data
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.session_state['data'] = load_existing_data()
    
    # Display dashboard
    if st.session_state.get('data') is not None:
        display_dashboard(st.session_state['data'])
    else:
        st.info("👆 Click 'Run Full Pipeline' in the sidebar to collect and analyze Reddit data.")
        st.markdown("""
        ### What this dashboard does:
        
        1. **Collects** the top posts from r/UofT (or any subreddit you specify)
        2. **Processes** the text data by cleaning and normalizing it
        3. **Analyzes** sentiment using state-of-the-art NLP models
        4. **Visualizes** the results with interactive charts and metrics
        
        Get started by clicking the button in the sidebar!
        """)


if __name__ == "__main__":
    main()
