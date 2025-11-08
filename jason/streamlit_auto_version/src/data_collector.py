import praw
import pandas as pd
from datetime import datetime
from pathlib import Path


def get_reddit_instance():
    """Initialize and return a Reddit instance using PRAW."""
    reddit = praw.Reddit(
        client_id='LDPs59MIc82xLXPftHQ3Sw',
        client_secret="uZSSG2q3ueggUkUReIMzaKe8xnC6iw",
        password='Linyihao.041228',
        user_agent="Comment Extraction (by SaltyfishQAQ)",
        username='Few-Strength-2343'
    )
    return reddit


def get_monthly_top_posts(reddit, subreddit_name, limit=10, time_filter='month'):
    """
    Fetch top posts and their comments from a subreddit.
    
    Args:
        reddit: PRAW Reddit instance
        subreddit_name: Name of the subreddit to fetch from
        limit: Number of top posts to fetch
        time_filter: Time filter for top posts ('month', 'week', 'year', etc.)
        
    Returns:
        tuple: (master_df, all_submissions) where master_df is a DataFrame of all posts/comments
               and all_submissions is a list of DataFrames, one per submission
    """
    count = 0
    subreddit = reddit.subreddit(subreddit_name)
    
    master_df = []
    all_submissions = []
    
    for submission in subreddit.top(time_filter=time_filter, limit=limit):
        single_submission = []
        
        # Collect submission information
        submission_info = {
            "submission_title": submission.title,
            "author": submission.author.name if submission.author else "deleted",
            "id": submission.id,
            "url": submission.url,
            "created_utc": submission.created_utc,
            "body": submission.selftext,  
            "type": "submission",
            "score": submission.score
        }
        master_df.append(submission_info)
        single_submission.append(submission_info)
        
        if submission.num_comments > 0:
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                # Skip MoreComments objects if any remain
                if isinstance(comment, praw.models.MoreComments):
                    continue
                
                comment_info = {
                    "submission_title": submission.title,
                    "author": comment.author.name if comment.author else "deleted",
                    "id": comment.id,
                    "url": f"https://www.reddit.com{comment.permalink}",
                    "created_utc": comment.created_utc,
                    "body": comment.body,
                    "type": "comment",
                    "score": comment.score
                }
                master_df.append(comment_info)
                single_submission.append(comment_info)
        
        all_submissions.append(single_submission)
        count += 1
        print(f"Processed submission {count}/{limit}: {submission.title}")
        
    master_df = pd.DataFrame(master_df).reset_index(drop=True)
        
    return master_df, all_submissions


def save_submissions_to_csv(all_submissions, master_df, folder_path):
    """
    Save individual submissions and master DataFrame to CSV files.
    
    Args:
        all_submissions: List of submission DataFrames
        master_df: Combined DataFrame of all submissions and comments
        folder_path: Path object or string for the folder to save files
        
    Returns:
        Path: Path to the master CSV file
    """
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # Save individual submissions
    for i, submission in enumerate(all_submissions):
        submission_df = pd.DataFrame(submission).reset_index(drop=True)
        submission_df.to_csv(folder_path / f'submission_{i}.csv', index=False)
    
    # Save master DataFrame
    today = datetime.now().strftime('%Y%m')
    top100_filename = f'top_100_reddits_{today}.csv'
    master_csv_path = folder_path / top100_filename
    master_df.to_csv(master_csv_path, index=False)
    
    return master_csv_path


def collect_reddit_data(subreddit_name='UofT', limit=100, time_filter='month'):
    """
    Complete data collection pipeline: fetch posts, save to CSV files.
    
    Args:
        subreddit_name: Name of the subreddit to collect from
        limit: Number of top posts to collect
        time_filter: Time filter for top posts
        
    Returns:
        tuple: (master_df, master_csv_path) - the combined DataFrame and path to saved CSV
    """
    print(f"Connecting to Reddit API...")
    reddit = get_reddit_instance()
    
    print(f"Fetching top {limit} posts from r/{subreddit_name}...")
    master_df, all_submissions = get_monthly_top_posts(reddit, subreddit_name, limit, time_filter)
    
    today = datetime.now().strftime('%Y%m')
    folder_path = Path(f'monthly_top100/{today}')
    
    print(f"Saving data to {folder_path}...")
    master_csv_path = save_submissions_to_csv(all_submissions, master_df, folder_path)
    
    print(f"Data collection complete! Collected {len(master_df)} posts/comments")
    return master_df, master_csv_path
