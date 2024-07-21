from transformers import pipeline
from alpaca_trade_api import REST
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from pydantic.v1 import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.tools import StructuredTool


def sentimental_analysis_tools():

    class AlpacaNewsFetcher:
        """
        A class for fetching news articles related to a specific stock from Alpaca API.

        Attributes:
        - api_key (str): Alpaca API key for authentication.
        - api_secret (str): Alpaca API secret for authentication.
        - rest_client (alpaca_trade_api.REST): Alpaca REST API client.
        """

        def __init__(self):
            """
            Initializes the AlpacaNewsFetcher object.

            Args:
            - api_key (str): Alpaca API key for authentication.
            - api_secret (str): Alpaca API secret for authentication.
            """
            load_dotenv()
            self.api_key = os.environ["ALPACA_API_KEY"]
            self.api_secret = os.environ["ALPACA_SECRET"]
            self.rest_client = REST(self.api_key, self.api_secret)

            #No of news articles to fetch for the input stock ticker.
            self.no_of_newsarticles_to_fetch = os.environ["NO_OF_NEWSARTICLES_TO_FETCH"]

            #No of days to fetch news articles for
            self.no_of_days = os.environ["NO_OF_DAYS_TO_FETCH_NEWS_ARTICLES"]


        def fetch_news(self, stockticker):
            """
            Fetches news articles for a given stock symbol within a specified date range.

            Args:
            - stockticker (str): Stock symbol for which news articles are to be fetched (e.g., "AAPL").

            Returns:
            - list: A list of dictionaries containing relevant information for each news article.
            """

            #Date range for which to get the news
            start_date = date.today()
            end_date = date.today() - timedelta(self.no_of_days)

            news_articles = self.rest_client.get_news(stockticker, start_date, end_date, limit=self.no_of_newsarticles_to_fetch )
            formatted_news = []

            for article in news_articles:
                summary = article.summary
                title = article.headline
                timestamp = article.created_at

                relevant_info = {
                    'timestamp': timestamp,
                    'title': title,
                    'summary': summary
                }

                formatted_news.append(relevant_info)

            return formatted_news
    
    
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


    #Function to get the stock sentiment
    def get_stock_sentiment(stockticker: str):
 
        #Initialize AlpacaNewsFetcher, a class for fetching news articles related to a specific stock from Alpaca API.
        news_fetcher = AlpacaNewsFetcher()


        # Fetch news (contains - title of the news, timestamp and summary) for specified stocksticker
        news_data = news_fetcher.fetch_news(stockticker)

        # Initialize the NewsSentimentAnalysis object
        news_sentiment_analyzer = NewsSentimentAnalysis()
        analysis_result = []
       
        # Assume 'news_data' is a list of news articles (each as a dictionary), analyze sentiment of each news
        for article in news_data:
            sentiment_analysis_result = news_sentiment_analyzer.analyze_sentiment(article)

            # Display sentiment analysis results
            print(f'Timestamp: {sentiment_analysis_result["timestamp"]}, '
                f'Title: {sentiment_analysis_result["title"]}, '
                f'Summary: {sentiment_analysis_result["summary"]}')

            print(f'Sentiment: {sentiment_analysis_result["sentiment"]}', '\n')

            result = {
                    'Timestamp': sentiment_analysis_result["timestamp"],
                    'News- Title:Summar': sentiment_analysis_result["title"] + sentiment_analysis_result["summary"],
                    'Sentiment': sentiment_analysis_result["sentiment"][0]['label']
                }
            analysis_result.append(result)

            #Extracting timestamp of article and sentiment of article for graphing
            """  result_for_graph = {
                    'Timestamp': sentiment_analysis_result["timestamp"],
                    'Sentiment': sentiment_analysis_result["sentiment"][0]['label']
                }
             
            analysis_result.append(result_for_graph)
            """

        #Get dominant sentiment
        dominant_sentiment = news_sentiment_analyzer.get_dominant_sentiment(sentiment_analysis_result)

        #Build response string for news sentiment
        output_string = ""
        for result in analysis_result:
            output_string = output_string + f'{result["Timestamp"]} : {result["News- Title:Summary"]} : {result["Sentiment"]}' + '\n'
        
        final_result = {
                'Sentiment-analysis-result' : output_string,
                'Dominant-sentiment' : dominant_sentiment['sentiment']
        }

        return final_result    


    class StockSentimentCheckInput(BaseModel):
        """Input for Stock price check."""
        stockticker: str = Field(..., description="Ticker symbol for stock or index")

    class StockSentimentAnalysisTool(BaseTool):
        name = "get_stock_sentiment"
        description = """Useful for finding sentiment of stock, based on published news articles. 
                        Fetches configured number of news items for the sentiment, 
                        determines sentiment of each news items and then returns 
                        List of sentiment analysit result & domainant sentiment of the news
                        """
        
        """Input for Stock sentiment analysis."""
        stockticker: str = Field(..., description="Ticker symbol for stock or index")
        def _run(self, stockticker: str):
            # print("i'm running")
            sentiment_response = get_stock_sentiment(stockticker)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(str(sentiment_response))
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            return sentiment_response

        def _arun(self, stockticker: str):
            raise NotImplementedError("This tool does not support async")
        
        args_schema: Optional[Type[BaseModel]] = StockSentimentCheckInput


    tools_sentiment_analyst = [StructuredTool.from_function(
            func=StockSentimentAnalysisTool,
            args_schema=StockSentimentCheckInput,
            description="Function to get stock sentiment.",
        )
    ]
    return tools_sentiment_analyst