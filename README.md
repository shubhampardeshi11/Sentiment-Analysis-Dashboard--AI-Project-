# Sentiment Analysis Dashboard

A powerful sentiment analysis dashboard built with Python, Streamlit, NLTK, and TextBlob. This application allows you to analyze the sentiment of text using both VADER and TextBlob sentiment analyzers.

## Features

- **Text Analysis**: Analyze individual text for sentiment using two methods (VADER and TextBlob)
- **Batch Analysis**: Upload CSV files with text data for batch processing
- **Interactive Visualizations**: Visual representation of sentiment scores and distributions
- **Word Frequency Visualization**: Visualize the most frequent words in your text (uses WordCloud if installed, otherwise falls back to matplotlib)
- **Sample Data**: Built-in sample dataset for testing and exploration
- **Color-coded Results**: Results are color-coded for easier interpretation
- **Export Functionality**: Download your analysis results as CSV files

## Screenshots

![Dashboard Screenshot](https://example.com/dashboard_screenshot.png)

## Requirements

- Python 3.6+
- Streamlit
- NLTK
- TextBlob
- Pandas
- Matplotlib
- WordCloud (optional)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd sentiment-analysis-dashboard
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Unix or MacOS:
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. You can either:
   - Enter text directly in the Text Analysis tab
   - Upload a CSV file containing a 'text' column in the Batch Analysis tab
   - Use the provided sample data for testing

## Input Format

For CSV files:
- The file must contain a column named 'text'
- Each row should contain the text to be analyzed

## How It Works

The dashboard uses two main sentiment analysis techniques:

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: A lexicon and rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media.

2. **TextBlob**: A simple library that provides a consistent API for common natural language processing tasks including sentiment analysis.

Results include:
- Compound scores (VADER)
- Positive, negative, and neutral scores (VADER)
- Polarity scores (TextBlob)
- Subjectivity scores (TextBlob)
- Visualizations of sentiment distributions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 