# FORECASTING EXPERT RF TOOLS

from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pydantic.v1 import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.tools import StructuredTool

def forecasting_expert_rf_tools():
    def RF_forecast(symbol,historical_data, train_days_ago, forecast_days):
        """Useful for forecasting a variable using ARIMA model.
        Use historical 'Close' stock prices and get prediction.
        Give prediction output.
        Send mae_rf from the model to  Evaluator.
        """
        df=historical_data[['Close']]
        df.index=pd.to_datetime(df.index)
        df.index.names=['date']
        end_date = datetime.now()

        df=df.reset_index()
        # Feature Engineering
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['lag1'] = df['Close'].shift(1)
        df['lag2'] = df['Close'].shift(2)
        df = df.dropna()

        # Prepare the data
        features = ['day','month', 'year', 'lag1', 'lag2']
        X = df[features]
        y = df['Close']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Initialize and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mae_rf = mean_absolute_error(y_test, y_pred)
        print(f'Mean Absolute Error: {mae_rf}')

        # Forecast future values (next 12 months)
        future_dates = pd.date_range(start=pd.to_datetime(end_date), end=pd.to_datetime(end_date)+ timedelta(days=forecast_days), freq='D')
        future_df = pd.DataFrame(future_dates, columns=['date'])
        future_df['day'] = future_df['date'].dt.day
        future_df['month'] = future_df['date'].dt.month
        future_df['year'] = future_df['date'].dt.year
        future_df['lag1'] = df['Close'].iloc[-1]
        future_df['lag2'] = df['Close'].iloc[-2]

        # Use the last observed values for lag features
        for i in range(1, len(future_df)):
            future_df.loc[future_df.index[i], 'lag1'] = future_df.loc[future_df.index[i-1], 'Close'] if 'Close' in future_df.columns else future_df.loc[future_df.index[i-1], 'lag1']
            future_df.loc[future_df.index[i], 'lag2'] = future_df.loc[future_df.index[i-1], 'lag1']

        future_X = future_df[features]
        future_df['Close'] = model.predict(future_X)
        rf_prediction=future_df['Close']
        # Print the forecasted values
        return {"predicted_price": rf_prediction,"mae_rf": mae_rf}

    class PredictStocksRFInput(BaseModel):
        """Input for Stock ticker check."""

        stockticker: str = Field(..., description="Ticker symbol for stock or index")
        days_ago: int = Field(..., description="Int number of days to look back")

    class PredictStocksRFTool(BaseTool):
        name = "Random_forest_forecast"
        description = "Useful for forecasting stock prices using Random forest model."

        def _run(self, stockticker: str, days_ago: int,historical_data: float, train_days_ago=int, forecast_days=int):
            predicted_prices = RF_forecast(stockticker,historical_data, train_days_ago, forecast_days).predict_price
            mae_rf= RF_forecast(stockticker,historical_data, train_days_ago, forecast_days).mae_rf
            return {"rf_prediction":rf_prediction,"mae_rf":mae_rf}

        def _arun(self, stockticker: str, days_ago: int,historical_data: float, train_days_ago=int, forecast_days=int):
            raise NotImplementedError("This tool does not support async")

        args_schema: Optional[Type[BaseModel]] = PredictStocksRFInput

    tools_forecasting_expert_random_forest = [
        StructuredTool.from_function(
            func=PredictStocksRFTool,
            args_schema=PredictStocksRFInput,
            description="Function to predict stock prices with random forest model and to get mae_rf for the model.",
        ),
        StructuredTool.from_function(
            func=PredictStocksRFTool,
            args_schema=PredictStocksRFInput,
            description="Function to predict stock prices with random forest model and to get mae_rf for the model.",
        ),
    ]
    return tools_forecasting_expert_random_forest