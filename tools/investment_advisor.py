from pydantic.v1 import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.tools import StructuredTool
import yfinance as yf
from typing import List
from datetime import datetime,timedelta
import pandas as pd

def investment_advisor_tools():


    def news_summary(df_search):
        "Take df_search from the user input message. Summarize news on the selected stockticker and provide Sentiment: positive/negative/neutral to the user."
        return eval(df_search)
    
    class newsSummaryInput(BaseModel):
        """Input for summarizing articles."""
        df_search: str = Field(..., description="News articles.")

    class newsSummaryTool(BaseTool):
        name = "Summarize news on the stockticker"
        description = """Useful for summarizing the newest article on a selected stockticker."""

        def _run(self, df_search=str):
            position = news_summary(df_search)
            return {"position": position}

        def _arun(self,df_search=str):
            raise NotImplementedError("This tool does not support async")

        args_schema: Optional[Type[BaseModel]] = newsSummaryInput

<<<<<<< HEAD
    def analyze_prices():
        """Take historical prices, analyze them and answer user's questions."""
        df_prices=pd.read_csv('../df_history.csv')
        return df_prices
    
    class pricesInput(BaseModel):
        """Input for summarizing articles."""
        stockticker: str = Field(..., description="stockticker name")

    class pricesTool(BaseTool):
        name = "Get prices from csv file analyze them and answer questions"
        description = """Useful for analyzing historical stock prices."""

        def _run(self, stockticker=str):
            df_prices = analyze_prices()
            return {"prices": df_prices}

        def _arun(self, stockticker=str):
            raise NotImplementedError("This tool does not support async")

        args_schema: Optional[Type[BaseModel]] = pricesInput
=======

>>>>>>> 594c18622dd698f1854229dda280817492475d75

    tools_reccommend = [
        StructuredTool.from_function(
            func=newsSummaryTool,
            args_schema=newsSummaryInput,
            description="Summarize articles.",
<<<<<<< HEAD
        ),
        StructuredTool.from_function(
            func=pricesTool,
            args_schema=pricesInput,
            description="Analyze stock prices.",
=======
>>>>>>> 594c18622dd698f1854229dda280817492475d75
        )
    ]
    return tools_reccommend