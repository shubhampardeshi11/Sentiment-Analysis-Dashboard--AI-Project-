import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sentiment_analyzer import SentimentAnalyzer
import nltk
from nltk.corpus import stopwords
from datetime import datetime
import re
from collections import Counter

# Remove wordcloud import and replace with our own implementation
# Try to use wordcloud if installed, otherwise use fallback
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# Download necessary NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    pass  # Handle silently if download fails

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize the sentiment analyzer
analyzer = SentimentAnalyzer()

# Create a custom implementation of word cloud using matplotlib
def create_word_frequency_chart(text, max_words=50):
    """Create a word frequency chart as an alternative to word cloud"""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Filter out stopwords
    try:
        stop_words = set(stopwords.words('english'))
        words = [word for word in text.split() if word not in stop_words and len(word) > 1]
    except:
        words = [word for word in text.split() if len(word) > 1]
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Get the most common words
    most_common = word_counts.most_common(max_words)
    
    if not most_common:
        return None
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    words = [word for word, count in most_common]
    counts = [count for word, count in most_common]
    
    # Use horizontal bar chart for better readability
    y_pos = range(len(words))
    ax.barh(y_pos, counts, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Frequency')
    ax.set_title('Word Frequency')
    
    return fig

# Title and description
st.title("📊 Sentiment Analysis Dashboard")
st.markdown("""
This dashboard allows you to analyze the sentiment of text using both VADER and TextBlob.
Upload a CSV file with a 'text' column or enter text manually to get started.
""")

# Add tabs for different functions
tab1, tab2, tab3 = st.tabs(["Text Analysis", "Batch Analysis", "About"])

with tab1:
    # Text input
    st.subheader("Enter text to analyze")
    text_input = st.text_area("Enter text to analyze", height=150)
    
    if text_input:
        result = analyzer.analyze_text(text_input)
        
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("VADER Analysis")
            # Create a more visually appealing display with gauges
            vader_compound = result['vader_compound']
            sentiment = analyzer.get_sentiment_label(vader_compound)
            
            # Use color coding based on sentiment
            if sentiment == 'Positive':
                color = 'green'
            elif sentiment == 'Negative':
                color = 'red'
            else:
                color = 'blue'
                
            st.markdown(f"<h3 style='color: {color}'>Overall Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
            st.write(f"Compound Score: {vader_compound:.2f}")
            st.write(f"Positive Score: {result['vader_pos']:.2f}")
            st.write(f"Negative Score: {result['vader_neg']:.2f}")
            st.write(f"Neutral Score: {result['vader_neu']:.2f}")
            
            # Add a simple visualization
            fig, ax = plt.subplots()
            scores = [result['vader_pos'], result['vader_neg'], result['vader_neu']]
            labels = ['Positive', 'Negative', 'Neutral']
            ax.bar(labels, scores, color=['green', 'red', 'blue'])
            ax.set_ylim(0, 1)
            ax.set_title('VADER Sentiment Scores')
            st.pyplot(fig)
        
        with col2:
            st.subheader("TextBlob Analysis")
            
            # Determine sentiment for TextBlob
            polarity = result['textblob_polarity']
            if polarity > 0.05:
                tb_sentiment = 'Positive'
                tb_color = 'green'
            elif polarity < -0.05:
                tb_sentiment = 'Negative'
                tb_color = 'red'
            else:
                tb_sentiment = 'Neutral'
                tb_color = 'blue'
                
            st.markdown(f"<h3 style='color: {tb_color}'>Overall Sentiment: {tb_sentiment}</h3>", unsafe_allow_html=True)
            st.write(f"Polarity: {polarity:.2f}")
            st.write(f"Subjectivity: {result['textblob_subjectivity']:.2f}")
            
            # Add a visualization for TextBlob
            fig2, ax2 = plt.subplots()
            ax2.scatter(polarity, result['textblob_subjectivity'], s=100, color=tb_color)
            ax2.set_xlim(-1, 1)
            ax2.set_ylim(0, 1)
            ax2.set_xlabel('Polarity')
            ax2.set_ylabel('Subjectivity')
            ax2.set_title('TextBlob Sentiment Analysis')
            ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
        
        # Generate word visualization based on available libraries
        st.subheader("Word Frequency Visualization")
        
        if HAS_WORDCLOUD:
            try:
                # Filter out stopwords
                stop_words = set(stopwords.words('english'))
                filtered_words = [word.lower() for word in text_input.split() if word.lower() not in stop_words]
                if filtered_words:
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
                    fig3, ax3 = plt.subplots(figsize=(10, 5))
                    ax3.imshow(wordcloud, interpolation='bilinear')
                    ax3.axis('off')
                    st.pyplot(fig3)
                else:
                    st.info("Not enough significant words for visualization.")
            except Exception as e:
                st.warning(f"Could not generate word cloud: {str(e)}")
                # Fallback to frequency chart
                freq_fig = create_word_frequency_chart(text_input)
                if freq_fig:
                    st.pyplot(freq_fig)
                else:
                    st.info("Not enough significant words for visualization.")
        else:
            # Use our custom implementation
            freq_fig = create_word_frequency_chart(text_input)
            if freq_fig:
                st.pyplot(freq_fig)
            else:
                st.info("Not enough significant words for visualization.")

with tab2:
    # File uploader
    st.subheader("Upload a CSV file")
    uploaded_file = st.file_uploader("File must contain a 'text' column", type=['csv'])
    
    # Sample data option
    st.markdown("### Or use sample data")
    use_sample = st.checkbox("Use sample data instead")
    
    if use_sample:
        # Create sample data with different sentiments
        sample_data = {
            'text': [
                "I love this product! It's amazing and works perfectly.",
                "This is terrible. Doesn't work as advertised and broke after a week.",
                "The product arrived on time. It works as expected.",
                "I'm very disappointed with the quality. Would not recommend.",
                "It's okay. Nothing special but gets the job done.",
                "Absolutely fantastic service and great quality.",
                "Not happy with my purchase. Customer service was unhelpful.",
                "Pretty good overall. Some minor issues but nothing major.",
                "Best purchase I've made this year! Can't recommend enough.",
                "Average product for the price. Neither good nor bad."
            ]
        }
        df = pd.DataFrame(sample_data)
        results_df = analyzer.analyze_batch(df['text'])
        
        # Add timestamp for when analysis was performed
        results_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add text column and user friendly sentiment label
        results_df['text'] = df['text']
        results_df['sentiment'] = results_df['vader_compound'].apply(analyzer.get_sentiment_label)
        
        st.success("Analysis complete on sample data!")
        
        # Display visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            fig1 = analyzer.plot_sentiment_distribution(results_df)
            st.pyplot(fig1)
        
        with col2:
            st.subheader("Sentiment Scores")
            fig2 = analyzer.plot_sentiment_scores(results_df)
            st.pyplot(fig2)
        
        # Display detailed results with better formatting
        st.subheader("Detailed Results")
        
        # Reorder columns for better display
        display_df = results_df[['text', 'sentiment', 'vader_compound', 'vader_pos', 
                                'vader_neg', 'vader_neu', 'textblob_polarity', 
                                'textblob_subjectivity', 'timestamp']]
        
        # Apply color formatting based on sentiment
        def color_sentiment(val):
            if val == 'Positive':
                return 'background-color: rgba(0, 128, 0, 0.2)'
            elif val == 'Negative':
                return 'background-color: rgba(255, 0, 0, 0.2)'
            else:
                return 'background-color: rgba(0, 0, 255, 0.2)'
        
        # Display the styled dataframe
        st.dataframe(display_df.style.applymap(color_sentiment, subset=['sentiment']))
        
        # Option to download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download CSV with results",
            data=csv,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )
        
    elif uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'text' not in df.columns:
                st.error("The CSV file must contain a 'text' column")
            else:
                results_df = analyzer.analyze_batch(df['text'])
                
                # Add timestamp for when analysis was performed
                results_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Add text column and user friendly sentiment label
                results_df['text'] = df['text']
                results_df['sentiment'] = results_df['vader_compound'].apply(analyzer.get_sentiment_label)
                
                st.success("Analysis complete!")
                
                # Display visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment Distribution")
                    fig1 = analyzer.plot_sentiment_distribution(results_df)
                    st.pyplot(fig1)
                
                with col2:
                    st.subheader("Sentiment Scores")
                    fig2 = analyzer.plot_sentiment_scores(results_df)
                    st.pyplot(fig2)
                
                # Display detailed results with better formatting
                st.subheader("Detailed Results")
                
                # Reorder columns for better display
                display_df = results_df[['text', 'sentiment', 'vader_compound', 'vader_pos', 
                                        'vader_neg', 'vader_neu', 'textblob_polarity', 
                                        'textblob_subjectivity', 'timestamp']]
                
                # Apply color formatting based on sentiment
                def color_sentiment(val):
                    if val == 'Positive':
                        return 'background-color: rgba(0, 128, 0, 0.2)'
                    elif val == 'Negative':
                        return 'background-color: rgba(255, 0, 0, 0.2)'
                    else:
                        return 'background-color: rgba(0, 0, 255, 0.2)'
                
                # Display the styled dataframe
                st.dataframe(display_df.style.applymap(color_sentiment, subset=['sentiment']))
                
                # Option to download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV with results",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with tab3:
    st.subheader("About This Dashboard")
    st.markdown("""
    ### Sentiment Analysis Dashboard
    
    This interactive web dashboard analyzes the sentiment (positive/neutral/negative) of text input or batches of text from CSV files.
    
    **Features:**
    
    * Text analysis using both VADER and TextBlob sentiment analyzers
    * Batch processing of CSV files
    * Visualization of sentiment distribution and scores
    * Word frequency visualization for text analysis
    * Sample data for demonstration
    * Download results as CSV
    
    **Technologies Used:**
    
    * Python - Programming language
    * Streamlit - Interactive web application framework
    * NLTK - Natural Language Toolkit for text processing and VADER sentiment analysis
    * TextBlob - Simplified text processing library
    * Matplotlib - Data visualization
    * Pandas - Data manipulation and analysis
    
    **How It Works:**
    
    1. **VADER Analysis:** Uses the VADER (Valence Aware Dictionary and sEntiment Reasoner) model from NLTK, which is specifically attuned to sentiments expressed in social media.
    
    2. **TextBlob Analysis:** Uses the TextBlob library which provides a simple API for common NLP tasks.
    
    3. **Scoring:**
       - Compound score: A normalized score between -1 (most extreme negative) and +1 (most extreme positive)
       - Positive/Negative/Neutral scores: The proportion of the text that falls into each category
       - Polarity: TextBlob's sentiment score from -1 to 1
       - Subjectivity: TextBlob's subjectivity score from 0 (objective) to 1 (subjective)
    """)

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit, NLTK, and TextBlob") 