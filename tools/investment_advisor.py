from pydantic.v1 import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.tools import StructuredTool
import yfinance as yf
from typing import List
from datetime import datetime,timedelta

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



    tools_reccommend = [
        StructuredTool.from_function(
            func=newsSummaryTool,
            args_schema=newsSummaryInput,
            description="Summarize articles.",
        )
    ]
    return tools_reccommend