# FORECASTING EXPERT ARIMA TOOLS

from datetime import datetime, timedelta
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from pydantic.v1 import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.tools import StructuredTool

def forecasting_expert_arima_tools():
    def ARIMA_forecast(symbol,historical_data, train_days_ago, forecast_days):
            """Useful for forecasting a variable using ARIMA model.
            Use historical 'Close' stock prices and get prediction.
            Give prediction output to the client.
            Give mae_arima from the model to  Evaluator.
            """

            df=historical_data[['Close']]
            df.index=pd.to_datetime(df.index)
            model = ARIMA(df.dropna(), order=(2,0,2))
            model_fit = model.fit()

            # Split the data into training and testing sets
            train_size = int(len(df) * 0.8)
            train, test = df.iloc[:train_size], df.iloc[train_size:]

            # Fit the ARIMA model on the training set
            model = ARIMA(train.dropna(), order=(2, 0, 2))
            model_fit = model.fit()

            # Make predictions
            predictions = model_fit.forecast(steps=len(test))
            #test['Predicted'] = predictions

            # Calculate the MAE
            mae_arima = mean_absolute_error(test['Close'], predictions)
            # plt.plot(y_test, label='Actual')
            # plt.plot(y_pred, label='Predicted')
            # plt.legend()
            # plt.show()
            forecast = model_fit.get_forecast(forecast_days).predicted_mean
            arima_prediction=forecast
            return {"arima_prediction": arima_prediction,"mae_arima": mae_arima}

    class PredictStocksARIMAInput(BaseModel):
        """Input for Stock ticker check."""

        stockticker: str = Field(..., description="Ticker symbol for stock or index")
        days_ago: int = Field(..., description="Int number of days to look back")

    class PredictStocksARIMATool(BaseTool):
        name = "ARIMA_forecast"
        description = "Useful for forecasting stock prices using ARIMA model."

        def _run(self, stockticker: str, days_ago: int,historical_data: float, train_days_ago=int, forecast_days=int):
            arima_prediction = ARIMA_forecast(stockticker,historical_data, train_days_ago, forecast_days).predicted_price
            mae_arima== ARIMA_forecast(stockticker,historical_data, train_days_ago, forecast_days).mae_arima

            return {"arima_prediction":arima_prediction,"mae_arima":mae_arima}

        def _arun(self, stockticker: str, days_ago: int,historical_data: float, train_days_ago=int, forecast_days=int):
            raise NotImplementedError("This tool does not support async")

        args_schema: Optional[Type[BaseModel]] = PredictStocksARIMAInput

    tools_forecasting_expert_arima = [
        StructuredTool.from_function(
            func=PredictStocksARIMATool,
            args_schema=PredictStocksARIMAInput,
            description="Function to predict stock prices with ARIMA model and to get mae_arima for the model.",
        ),
        StructuredTool.from_function(
            func=PredictStocksARIMATool,
            args_schema=PredictStocksARIMAInput,
            description="Function to predict stock prices with ARIMA model and to get mae_arima for the model.",
        )
    ]
    return tools_forecasting_expert_arima