from pydantic.v1 import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.tools import StructuredTool
import yfinance as yf
from typing import List
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import chainlit as cl
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from plotly.subplots import make_subplots

def chart_expert_tools(): 

    def historical_stock_prices(stockticker, days_ago):
        """Upload accurate data to accurate dates from yahoo finance.
        Receive data on the last week and give them to forecasting experts.
        Receive data on the last 90 days and give them to visualization expert."""
        ticker = yf.Ticker(stockticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_ago)
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
        historical_data = ticker.history(start=start_date, end=end_date)
        return historical_data

    class HistoricalStockPricesInput(BaseModel):
        """Input for Stock ticker check."""

        stockticker: str = Field(..., description="Ticker symbol for stock or index")
        days_ago: int = Field(..., description="Int number of days to look back")

    class HistoricalStockPricesTool(BaseTool):
        name = "historical_stock_prices"
        description = "Useful for when you need to find out the historical stock prices. Use Yahoo Finance API to find the correct stockticker."

        def _run(self, stockticker: str, days_ago: int):
            historical_prices = historical_stock_prices(stockticker, days_ago)

            return {"historical prices":  historical_prices}

        def _arun(self, stockticker: str, days_ago: int):
            raise NotImplementedError("This tool does not support async")

        args_schema: Optional[Type[BaseModel]] = HistoricalStockPricesInput

    def calculate_MACD(historical_data, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculates the MACD (Moving Average Convergence Divergence) and related indicators.

        Parameters:
            df (DataFrame): A pandas DataFrame containing at least a 'Close' column with closing prices.
            fast_period (int): The period for the fast EMA (default is 12).
            slow_period (int): The period for the slow EMA (default is 26).
            signal_period (int): The period for the signal line EMA (default is 9).

        Returns:
            DataFrame: A pandas DataFrame with the original data and added columns for MACD, Signal Line, and MACD Histogram.
        """
        df=historical_data[['Close','Open','High','Low']]
        df['EMA_fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']

        df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        return df


    class MACDCalculateInput(BaseModel):
        """Input for Stock ticker check."""
        stockticker: str = Field(..., description="Ticker symbol for stock or index")

    class MACDCalculateTool(BaseTool):
        name = "macd_calculate"
        description = "Useful for calculating MACD as input for MACD plot."

        def _run(self, stockticker: str, historical_data: float):
            df = calculate_MACD(historical_data)

            return df

        def _arun(self, stockticker: str, historical_data: float):
            raise NotImplementedError("This tool does not support async")

        args_schema: Optional[Type[BaseModel]] = MACDCalculateInput

    def plot_macd(df):

        # Create Figure
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                            vertical_spacing=0.15,  # Adjust vertical spacing between subplots
                            subplot_titles=("Candlestick Chart", "MACD"))  # Add subplot titles


        # Subplot 1: Plot candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='#00cc96',  # Green for increasing
            decreasing_line_color='#ff3e3e',  # Red for decreasing
            showlegend=False
        ), row=1, col=1)  # Specify row and column indices


        # Subplot 2: Plot MACD
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue')
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Signal_Line'],
                mode='lines',
                name='Signal Line',
                line=dict(color='red')
            ),
            row=2, col=1
        )

        # Plot MACD Histogram with different colors for positive and negative values
        histogram_colors = ['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACD_Histogram'],
                name='MACD Histogram',
                marker_color=histogram_colors
            ),
            row=2, col=1
        )

        # Update layout with zoom and pan tools enabled
        layout = go.Layout(
            title='MSFT Candlestick Chart and MACD Subplots',
            title_font=dict(size=25),  # Adjust title font size
            plot_bgcolor='#f2f2f2',  # Light gray background
            height=800,
            width=1500,
            xaxis_rangeslider=dict(visible=True, thickness=0.03),
        )

        # Update the layout of the entire figure
        fig.update_layout(layout)
        fig.update_yaxes(fixedrange=False, row=1, col=1)
        fig.update_yaxes(fixedrange=True, row=2, col=1)
        fig.update_xaxes(type='category', row=1, col=1)
        fig.update_xaxes(type='category', nticks=10, row=2, col=1)

        fig.show()

    class PlotMACDInput(BaseModel):
        """Input for Stock ticker check."""

        stockticker: str = Field(..., description="Ticker symbol for stock or index")
        df: List = Field(..., description="List of historical price values")
        days_ago: int = Field(..., description="Int number of days to look back")

    class PlotMACDTool(BaseTool):
        name = "plot_macd"
        description = "Useful for creating beautiful candle stick plot for MACD for a stock price."

        def _run(self, df: List[float]):
            historical_prices = plot_macd(df)

            return {"historical prices":  historical_prices}

        def _arun(self, df: List[float]):
            raise NotImplementedError("This tool does not support async")

        args_schema: Optional[Type[BaseModel]] = PlotMACDInput



    tools_chart_expert = [StructuredTool.from_function(
            func=HistoricalStockPricesTool,
            args_schema=HistoricalStockPricesInput,
            description="Function to get historical stock prices.",
        ),
        StructuredTool.from_function(
            func=MACDCalculateTool,
            args_schema=MACDCalculateInput,
            description="Calculate MACD as  input for MACD plot.",
        ),
        StructuredTool.from_function(
            func=PlotMACDTool,
            args_schema=PlotMACDInput,
            description="Plot MACD.",
        ),

    ]
    return tools_chart_expert