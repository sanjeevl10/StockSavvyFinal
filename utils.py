import matplotlib.pyplot as plt
import chainlit as cl
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from plotly.subplots import make_subplots

def get_stock_price(stockticker: str) -> str:
        ticker = yf.Ticker(stockticker)
        todays_data = ticker.history(period='1d')
        return str(round(todays_data['Close'][0], 2))

def plot_candlestick_stock_price(historical_data):
    """Useful for plotting candlestick plot for stock prices.
    Use historical stock price data from yahoo finance for the week and plot them."""
    df=historical_data[['Close','Open','High','Low']]
    df.index=pd.to_datetime(df.index)
    df.index.names=['Date']
    df=df.reset_index()

    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    fig.show()

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

def plot_macd2(df):
    try:
        # Debugging: Print the dataframe columns and a few rows
        print("DataFrame columns:", df.columns)
        print("DataFrame head:\n", df.head())

        # Convert DataFrame index and columns to numpy arrays
        index = df.index.to_numpy()
        close_prices = df['Close'].to_numpy()
        macd = df['MACD'].to_numpy()
        signal_line = df['Signal_Line'].to_numpy()
        macd_histogram = df['MACD_Histogram'].to_numpy()

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Subplot 1: Candlestick chart
        ax1.plot(index, close_prices, label='Close', color='black')
        ax1.set_title("Candlestick Chart")
        ax1.set_ylabel("Price")
        ax1.legend()

        # Subplot 2: MACD
        ax2.plot(index, macd, label='MACD', color='blue')
        ax2.plot(index, signal_line, label='Signal Line', color='red')

        histogram_colors = np.where(macd_histogram >= 0, 'green', 'red')
        ax2.bar(index, macd_histogram, color=histogram_colors, alpha=0.6)

        ax2.set_title("MACD")
        ax2.set_ylabel("MACD Value")
        ax2.legend()

        plt.xlabel("Date")
        plt.tight_layout()

        return fig
    except Exception as e:
        print(f"Error in plot_macd: {e}")
        return None

def plot_macd(df):

    # Create Figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.2, 0.1],
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
        title_font=dict(size=12),  # Adjust title font size
        plot_bgcolor='#f2f2f2',  # Light gray background
        height=600,
        width=1200,
        xaxis_rangeslider=dict(visible=True, thickness=0.03),
    )

    # Update the layout of the entire figure
    fig.update_layout(layout)
    fig.update_yaxes(fixedrange=False, row=1, col=1)
    fig.update_yaxes(fixedrange=True, row=2, col=1)
    fig.update_xaxes(type='category', row=1, col=1)
    fig.update_xaxes(type='category', nticks=10, row=2, col=1)
    
    fig.show()
    #return fig

def calculate_MACD(df, fast_period=12, slow_period=26, signal_period=9):
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

    df['EMA_fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']

    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

    return df