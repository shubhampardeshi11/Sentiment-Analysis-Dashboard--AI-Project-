import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def analyze_text(self, text):
        """Analyze sentiment of a single text using both VADER and TextBlob"""
        # VADER analysis
        vader_scores = self.sia.polarity_scores(text)
        
        # TextBlob analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_pos': vader_scores['pos'],
            'vader_neg': vader_scores['neg'],
            'vader_neu': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity
        }
    
    def analyze_batch(self, texts):
        """Analyze a list of texts and return a DataFrame with results"""
        results = []
        for text in texts:
            if text.strip():  # Only analyze non-empty texts
                result = self.analyze_text(text)
                results.append(result)
        
        return pd.DataFrame(results)
    
    def get_sentiment_label(self, compound_score):
        """Convert compound score to sentiment label"""
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    def plot_sentiment_distribution(self, df):
        """Create a pie chart of sentiment distribution"""
        sentiment_labels = df['vader_compound'].apply(self.get_sentiment_label)
        sentiment_counts = sentiment_labels.value_counts()
        
        plt.figure(figsize=(8, 6))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
        plt.title('Sentiment Distribution')
        return plt
    
    def plot_sentiment_scores(self, df):
        """Create a box plot of sentiment scores"""
        plt.figure(figsize=(10, 6))
        # Create a boxplot manually with matplotlib instead of seaborn
        plt.boxplot([df['vader_pos'], df['vader_neg'], df['vader_neu']], 
                   labels=['Positive', 'Negative', 'Neutral'])
        plt.title('Sentiment Score Distribution')
        plt.ylabel('Score')
        return plt 