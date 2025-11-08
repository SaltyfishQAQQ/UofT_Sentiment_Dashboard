import pandas as pd
import re
from datetime import datetime
import emoji
import string
from pathlib import Path
import os
from pysentimiento import create_analyzer
from transformers import pipeline


# Initialize analyzers globally to avoid re-initialization
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
sentiment_analyzer = create_analyzer(task="sentiment", lang="en")


def clean_text(text):
    """Remove all special characters from the text except for letters, numbers, and spaces."""
    return re.sub(r'[^\w\s]', '', text)


def convert_emoji_to_text(text, emoji_wrapper="emoji"):
    """
    Converts emoji in the text to descriptive text.
    
    Args:
        text (str): The original text containing emoji.
        emoji_wrapper (str): A string that will be used to wrap the emoji text.
        
    Returns:
        str: The text with emojis converted to wrapped text.
    """
    # Convert emojis to descriptive text (e.g., :smile:)
    demojized = emoji.demojize(text)
    # Define a wrapper string (e.g., " emoji ")
    wrapper = f" {emoji_wrapper} ".replace("  ", " ")
    # Replace the demojized emoji pattern :emoji_name: with the wrapped emoji_name
    result = re.sub(r':([^:\s]+):', lambda m: wrapper + m.group(1) + wrapper, demojized)
    return result


def is_valid_word(word):
    """
    Returns True if the word is either:
    - Composed solely of punctuation
    - Contains at least one English letter (a-z or A-Z)
    """
    # Keep if word is only punctuation
    if all(ch in string.punctuation for ch in word):
        return True
    # Keep if the word contains at least one ASCII letter
    if re.search(r'[a-zA-Z]', word):
        return True
    return False


def filter_non_english(text):
    """
    Splits the text into words and filters out any word that is not an English word,
    an emoji, or punctuation.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: The cleaned text.
    """
    words = text.split()
    filtered_words = [word for word in words if is_valid_word(word)]
    return ' '.join(filtered_words)


def convert_utc_to_readable(utc_timestamp):
    """
    Convert UTC timestamp to a readable date format (YYYY-MM-DD HH:MM:SS).
    
    Args:
        utc_timestamp (float): UTC timestamp in seconds since epoch.
        
    Returns:
        str: Formatted date string or empty string if input is NaN.
    """
    try:
        # Handle NaN values
        if pd.isna(utc_timestamp):
            return ""
        
        # Convert to datetime object
        dt = datetime.fromtimestamp(utc_timestamp)
        
        # Format as YYYY-MM-DD HH:MM:SS
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    
    except (ValueError, TypeError, OSError):
        return ""


def sentiment_analysis(master_df, text_column='body'):
    """
    Iterates through each text in the DataFrame, applies sentiment_analyzer,
    and adds two columns: sentiment_prediction and sentiment_prob.
    
    Args:
        master_df (pd.DataFrame): The DataFrame to analyze
        text_column (str): The column containing text to analyze
        
    Returns:
        pd.DataFrame: DataFrame with added sentiment columns
    """
    senti_pred = []
    senti_prob = []
    
    for index, row in master_df.iterrows():
        text = row[text_column]
        
        if pd.isna(text) or text == "":
            senti_pred.append("NEU")
            senti_prob.append({"NEU": 0.34, "NEG": 0.33, "POS": 0.33})
        else:
            result = sentiment_analyzer.predict(text)
            senti_pred.append(result.output)
            senti_prob.append(result.probas)
    
    master_df['sentiment_prediction'] = senti_pred
    master_df['sentiment_prob'] = senti_prob
    master_df['sentiment_score'] = master_df['sentiment_prob'].apply(lambda x: x.get('POS', 0) - x.get('NEG', 0))
    
    return master_df


def emotion_analysis(input_df, text_column='body'):
    """
    Apply emotion analysis to text in the DataFrame.
    
    Args:
        input_df (pd.DataFrame): The DataFrame to analyze
        text_column (str): The column containing text to analyze
        
    Returns:
        pd.DataFrame: DataFrame with added emotion columns
    """
    emotion_pred = []
    emotion_prob = []
    
    for index, row in input_df.iterrows():
        text = row[text_column]
        result = classifier(text)
        
        # Sort results by score in descending order
        sorted_results = sorted(result[0], key=lambda x: x['score'], reverse=True)
        
        # Get top emotion and its confidence
        top_emotion = sorted_results[0]
        top_confidence = top_emotion['score']
        
        if top_confidence >= 0.7:
            # Use only top emotion if confidence >= 70%
            emotion_pred.append(top_emotion['label'])
            emotion_prob.append({top_emotion['label']: top_emotion['score']})
        else:
            # Use top 2 emotions if confidence < 70%
            top_2_emotions = sorted_results[:2]
            labels = [emotion['label'] for emotion in top_2_emotions]
            emotion_pred.append(labels)  # Store as list of 2 emotions
            emotion_prob.append({emotion['label']: emotion['score'] for emotion in top_2_emotions})

    input_df['emotion_prediction'] = emotion_pred
    input_df['emotion_prob'] = emotion_prob

    return input_df


def preprocess_dataframe(df):
    """
    Apply all preprocessing steps to a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to preprocess
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # 1. Keep only posts with URLs starting with UofT comments
    processed_df = processed_df[processed_df['url'].str.startswith('https://www.reddit.com/r/UofT/comments/')].reset_index(drop=True).copy()
    
    # 2. Clean body text: lowercase, strip
    processed_df['body'] = processed_df['body'].str.lower()
    processed_df['body'] = processed_df['body'].str.strip()
    
    # 3. Remove special characters except spaces
    processed_df['body'] = processed_df['body'].apply(clean_text)
    
    # 4. Converts emoji in the text to descriptive text
    processed_df['body'] = processed_df['body'].apply(convert_emoji_to_text)
    
    # 5. Filter out non-English words
    processed_df['body'] = processed_df['body'].apply(filter_non_english)
    
    # 6. Convert created_utc to readable date format
    processed_df['created_utc'] = processed_df['created_utc'].apply(convert_utc_to_readable)
    
    return processed_df


def process_all_monthly_submissions(folder_path, senti=True, emo=True):
    """
    Process all CSV files in the monthly submissions folder.
    Apply preprocessing and sentiment analysis to each file.
    
    Args:
        folder_path (str): Path to the folder containing CSV files
        senti (bool): Whether to run sentiment analysis
        emo (bool): Whether to run emotion analysis
        
    Returns:
        tuple: (all_processed_data, combined_df) - list of individual DataFrames and combined DataFrame
    """
    # Get list of all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    csv_files.sort()  # Sort to process in order
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    all_processed_data = []
    
    for i, filename in enumerate(csv_files):
        if filename.startswith('top_100_reddits_'):
            continue  # Skip the combined file if it exists
            
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file {i+1}/{len(csv_files)}: {filename}")
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            print(f"  - Loaded {len(df)} rows")
            
            # Apply preprocessing
            processed_df = preprocess_dataframe(df)
            print(f"  - After filtering: {len(processed_df)} rows")
            
            # Apply sentiment analysis
            if len(processed_df) > 0:
                if senti:
                    processed_df = sentiment_analysis(processed_df, text_column='body')
                    print(f"  - Sentiment analysis completed")

                if emo:
                    processed_df = emotion_analysis(processed_df, text_column='body')
                    print(f"  - Emotion analysis completed")
                
                # Add source file information
                processed_df['source_file'] = filename
                
                all_processed_data.append(processed_df)
            else:
                print(f"  - No data remaining after filtering")
                
        except Exception as e:
            print(f"  - Error processing {filename}: {str(e)}")
            continue
    
    # Combine all processed data
    if all_processed_data:
        combined_df = pd.concat(all_processed_data, ignore_index=True)
        print(f"\nProcessing complete! Combined dataset has {len(combined_df)} rows")
        return all_processed_data, combined_df
    else:
        print("\nNo data was successfully processed")
        return [], pd.DataFrame()


def process_reddit_data(folder_path, output_path=None):
    """
    Complete data processing pipeline: preprocess, analyze sentiment/emotion, and save.
    
    Args:
        folder_path (str): Path to the folder containing CSV files
        output_path (str, optional): Path to save the processed CSV. If None, saves to folder_path
        
    Returns:
        pd.DataFrame: The processed and combined DataFrame
    """
    print(f"Starting to process all submission CSV files in {folder_path}...")
    
    all_processed_data, processed_monthly_data = process_all_monthly_submissions(
        folder_path, senti=True, emo=True
    )
    
    # Display summary statistics
    if len(processed_monthly_data) > 0:
        print(f"\n=== PROCESSING SUMMARY ===")
        print(f"Total processed rows: {len(processed_monthly_data)}")
        print(f"Unique submissions: {processed_monthly_data['submission_title'].nunique()}")
        print(f"Date range: {processed_monthly_data['created_utc'].min()} to {processed_monthly_data['created_utc'].max()}")
        
        # Sentiment distribution
        print(f"\n=== SENTIMENT DISTRIBUTION ===")
        sentiment_counts = processed_monthly_data['sentiment_prediction'].value_counts()
        print(sentiment_counts)
        print(f"Sentiment percentages:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(processed_monthly_data)) * 100
            print(f"  {sentiment}: {percentage:.1f}%")
        
        # Save processed data
        if output_path is None:
            today = datetime.now().strftime('%Y%m')
            file_name = f'top_100_reddits_{today}.csv'
            output_path = Path(folder_path) / file_name
        
        processed_monthly_data.to_csv(output_path, index=False)
        print(f"\nProcessed data saved to: {output_path}")
    else:
        print("No data was processed successfully.")
    
    return processed_monthly_data
