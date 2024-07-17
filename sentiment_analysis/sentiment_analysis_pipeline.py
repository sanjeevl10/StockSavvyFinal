# !pip install transformers
from transformers import pipeline
from client import AlpacaNewsFetcher
from alpaca_trade_api import REST
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import date



class NewsSentimentAnalysis:
    """
  A class for sentiment analysis of news articles using the Transformers library.

  Attributes:
  - classifier (pipeline): Sentiment analysis pipeline from Transformers.
  """

    def __init__(self):
        """
    Initializes the NewsSentimentAnalysis object.
    """
        self.classifier = pipeline('sentiment-analysis')

    def analyze_sentiment(self, news_article):
        """
    Analyzes the sentiment of a given news article.

    Args:
    - news_article (dict): Dictionary containing 'summary', 'headline', and 'created_at' keys.

    Returns:
    - dict: A dictionary containing sentiment analysis results.
    """
        summary = news_article['summary']
        title = news_article['title']
        timestamp = news_article['timestamp']

        relevant_text = summary + title
        sentiment_result = self.classifier(relevant_text)

        analysis_result = {
            'timestamp': timestamp,
            'title': title,
            'summary': summary,
            'sentiment': sentiment_result
        }

        return analysis_result
    
    def plot_sentiment_graph(self, sentiment_analysis_result):
        """
        Plots a sentiment analysis graph 

        Args:
        - sentiment_analysis_result): (dict): Dictionary containing 'summary', 'headline', and 'created_at' keys.

        Returns:
        - dict: A dictionary containing sentiment analysis results.
        """
        df = pd.DataFrame(sentiment_analysis_result)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Date'] = df['Timestamp'].dt.date

        #Group by Date, sentiment value count
        grouped = df.groupby(by='Date')['Sentiment'].value_counts()
        grouped.plot.pie()
        

    def get_dominant_sentiment (self, sentiment_analysis_result):
        """
        Returns overall sentiment, negative or positive or neutral depending on the count of negative sentiment vs positive sentiment 

        Args:
        - sentiment_analysis_result): (dict): Dictionary containing 'summary', 'headline', and 'created_at' keys.

        Returns:
        - dict: A dictionary containing sentiment analysis results.
        """
        df = pd.DataFrame(sentiment_analysis_result)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Date'] = df['Timestamp'].dt.date

        #Group by Date, sentiment value count
        grouped = df.groupby(by='Date')['Sentiment'].value_counts()
        df = pd.DataFrame(list(grouped.items()), columns=['Sentiment', 'count'])
        df['date'] = df['Sentiment'].apply(lambda x: x[0])
        df['sentiment'] = df['Sentiment'].apply(lambda x: x[1])
        df.drop('Sentiment', axis=1, inplace=True)
        result = df.groupby('sentiment')['count'].sum().reset_index()
        
        # Determine the sentiment with the most count
        dominant_sentiment = result.loc[result['count'].idxmax()]

        return dominant_sentiment



#starting point of the program
if __name__ == '__main__':
    # Example Usage:
    # Initialize the AlpacaNewsFetcher object

    #Load Alpaca Key and Secret from environment.
    load_dotenv()
    api_key = os.environ["ALPACA_API_KEY"]
    api_secret = os.environ["ALPACA_SECRET"]

    #Initialize AlpacaNewsFetcher, a class for fetching news articles related to a specific stock from Alpaca API.
    news_fetcher = AlpacaNewsFetcher(api_key, api_secret)

    # Fetch news (contains - title of the news, timestamp and summary) for AAPL from 2021-01-01 to 2021-12-31
    news_data = news_fetcher.fetch_news("AAPL", "2021-01-01", "2021-12-31")

    # Initialize the NewsSentimentAnalysis object
    news_sentiment_analyzer = NewsSentimentAnalysis()
    analysis_result = []
    # Assume 'news_data' is a list of news articles (each as a dictionary)
    for article in news_data:
        sentiment_analysis_result = news_sentiment_analyzer.analyze_sentiment(article)

        # Display sentiment analysis results
        """ print(f'Timestamp: {sentiment_analysis_result["timestamp"]}, '
              f'Title: {sentiment_analysis_result["title"]}, '
              f'Summary: {sentiment_analysis_result["summary"]}')

        print(f'Sentiment: {sentiment_analysis_result["sentiment"]}', '\n') """

        #Extracting timestamp of article and sentiment of article for graphing
        result = {
                    'Timestamp': sentiment_analysis_result["timestamp"],
                    'News- Title:Summary': sentiment_analysis_result["title"] + sentiment_analysis_result["summary"],
                    'Sentiment': sentiment_analysis_result["sentiment"][0]['label']
                }
        
        analysis_result.append(result)

    #Graph dominant sentiment based on sentiment analysis data of news articles
    dominant_sentiment = news_sentiment_analyzer.get_dominant_sentiment(analysis_result)
    
    final_result = {
        'Sentiment-analysis-result' : analysis_result,
        'Dominant-sentiment' : dominant_sentiment['sentiment']
    }

    print(final_result)


