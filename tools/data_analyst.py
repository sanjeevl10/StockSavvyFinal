from pydantic.v1 import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.tools import StructuredTool
import yfinance as yf
from typing import List
from datetime import datetime,timedelta

def data_analyst_tools():
    def get_stock_price(stockticker: str) -> str:
        ticker = yf.Ticker(stockticker)
        todays_data = ticker.history(period='1d')
        return str(round(todays_data['Close'][0], 2))

    class StockPriceCheckInput(BaseModel):
        """Input for Stock price check."""
        stockticker: str = Field(..., description="Ticker symbol for stock or index")

    class StockPriceTool(BaseTool):
        name = "get_stock_ticker_price"
        description = "Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API"
        """Input for Stock price check."""
        stockticker: str = Field(..., description="Ticker symbol for stock or index")
        def _run(self, stockticker: str):
            # print("i'm running")
            price_response = get_stock_price(stockticker)

            return str(price_response)

        def _arun(self, stockticker: str):
            raise NotImplementedError("This tool does not support async")
        args_schema: Optional[Type[BaseModel]] = StockPriceCheckInput

    def historical_stock_prices(stockticker, days_ago):
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

    tools_data_analyst = [StructuredTool.from_function(
            func=StockPriceTool,
            args_schema=StockPriceCheckInput,
            description="Function to get current stock prices.",
        ),
        # StructuredTool.from_function(
        #     func=HistoricalStockPricesTool,
        #     args_schema=HistoricalStockPricesInput,
        #     description="Function to get historical stock prices.",
        # )
    ]
    return tools_data_analyst