# EVALUATOR
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from pydantic.v1 import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.tools import StructuredTool

def evaluator_tools():
    def compare_prediction(mae_rf, mae_arima,prediction_rf,prediction_arima):
        if mae_rf>mae_arima:
            result=prediction_arima
        else:
            result=prediction_rf
        return {"final_predicted_outcome": result}#,"mae_rf": mae_rf}

    class compare_predictionInput(BaseModel):
        """Input for printing final prediction number."""
        mae_rf: int = Field(..., description="Mean average error for random forest")
        mae_arima: int = Field(..., description="Mean average error for ARIMA")

        prediction_rf: int = Field(..., description="Price prediction using random forest")
        prediction_arima: int = Field(..., description="Price prediction using ARIMA")

    class compare_predictionTool(BaseTool):
        name = "Comparing rf and arima predictions"
        description = "Useful for showing which predicted outcome is the final result."

        def _run(self, mae_rf=int,mae_arima=int,prediction_rf=int,prediction_arima=int):
            result = compare_prediction(mae_rf,mae_arima,prediction_rf,prediction_arima)
            return {"final_predicted_outcome": result}

        def _arun(self, mae_rf=int,mae_arima=int,prediction_rf=int,prediction_arima=int):
            raise NotImplementedError("This tool does not support async")

        args_schema: Optional[Type[BaseModel]] = compare_predictionInput

    tools_evaluate = [
        StructuredTool.from_function(
            func=compare_predictionTool,
            args_schema=compare_predictionInput,
            description="Function to evaluate predicted stock prices and print final result.",
        )
    ]
    return tools_evaluate