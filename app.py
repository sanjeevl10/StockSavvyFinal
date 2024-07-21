from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import chainlit as cl
from plotly.subplots import make_subplots
import utils as u
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import data_analyst, stock_sentiment_evalutor, forecasting_expert_arima, forecasting_expert_rf, evaluator, investment_advisor
import functools
from typing import Annotated, Sequence, TypedDict
from langgraph.graph import END, StateGraph
import operator
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import yfinance as yf
from chainlit.input_widget import Select
import matplotlib.pyplot as plt
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig


load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

from GoogleNews import GoogleNews

def search_news(stockticker):
    googlenews = GoogleNews()
    googlenews.set_period('7d')
    googlenews.get_news(stockticker)
    return googlenews.get_texts()

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

llm = ChatOpenAI(model="gpt-3.5-turbo")

#======================== AGENTS ==================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Data Analyst
prompt_data_analyst = "You are a stock data analyst. Provide correct stock ticker from Yahoo Finance. Expected output: stockticker. Provide it in the following format: >>stockticker>> for example: >>AAPL>>"
tools_data_analyst = data_analyst.data_analyst_tools()
data_agent = create_agent(llm, tools_data_analyst, prompt_data_analyst)
get_historical_prices = functools.partial(agent_node, agent=data_agent, name="Data_analyst")

# ARIMA Forecasting Expert
prompt_forecasting_expert_arima = "system You are a stock prediction expert. Take historical stock data from message and train the ARIMA model from statsmodels Python library on the last week, then provide prediction for the 'Close' price for the next day. Give the value for mae_arima to Evaluator. Expected output: list of predicted prices with predicted dates for a selected stock ticker and mae_arima value. assistant"
tools_forecasting_expert_arima = forecasting_expert_arima.forecasting_expert_arima_tools()
code_forecasting_arima = create_agent(llm, tools_forecasting_expert_arima, prompt_forecasting_expert_arima)
predict_future_prices_arima = functools.partial(agent_node, agent=code_forecasting_arima, name="Forecasting_expert_ARIMA")

# RF Forecasting Expert
prompt_forecasting_expert_random_forest = "system You are a stock prediction expert. Take historical stock data from message and train the Random Forest model from statsmodels Python library on the last week, then provide prediction for the 'Close' price for the next day. Give the value for mae_rf to Evaluator. Expected output: list of predicted prices with predicted dates for a selected stock ticker and mae_rf value. assistant"
tools_forecasting_expert_random_forest = forecasting_expert_rf.forecasting_expert_rf_tools()
code_forecasting_random_forest = create_agent(llm, tools_forecasting_expert_random_forest, prompt_forecasting_expert_random_forest)
predict_future_prices_random_forest = functools.partial(agent_node, agent=code_forecasting_random_forest, name="Forecasting_expert_random_forest")

# Evaluator
prompt_evaluator = "system You are an evaluator. Retrieve arima_prediction and arima mean average error from forecasting expert arima and rf_prediction and mean average error for random forest from forecasting expert random forest. Print final prediction number. Next, compare prediction price and current price to provide recommendation if he should buy/sell/hold the stock. Expected output: one value for the prediction, explain why you have selected this value, recommendation buy or sell stock and why. assistant"
tools_evaluate = evaluator.evaluator_tools()
code_evaluate = create_agent(llm, tools_evaluate, prompt_evaluator)
evaluate = functools.partial(agent_node, agent=code_evaluate, name="Evaluator")

# Stock Sentiment Evaluator
prompt_sentiment_evaluator = "system You are a stock sentiment evaluator. Retrieve news for the stock based on the configured data range starting today and their corresponding sentiment, along with the most dominant sentiment for the stock. Expected output: List ALL stock news and their sentiment from the StockSentimentAnalysis tool response, and the dominant sentiment for the stock also in StockSentimentAnalysis tool response as is without change. assistant"
tools_sentiment_evaluator = stock_sentiment_evalutor.sentimental_analysis_tools()
sentiment_evaluator = create_agent(llm, tools_sentiment_evaluator, prompt_sentiment_evaluator)
evaluate_sentiment = functools.partial(agent_node, agent=sentiment_evaluator, name="Sentiment_Evaluator")

# Investment Advisor
prompt_inv_advisor = "system Provide personalized investment advice and recommendations. Consider user input message for the latest news on the stock. Provide overall sentiment of the news Positive/Negative/Neutral, and recommend if the user should invest in such stock. MUST finish the analysis with a summary on the latest news from the user input on the stock! assistant"
tools_reccommend = investment_advisor.investment_advisor_tools()
code_inv_advisor = create_agent(llm, tools_reccommend, prompt_inv_advisor)
reccommend = functools.partial(agent_node, agent=code_inv_advisor, name="Investment_advisor")

workflow_data = StateGraph(AgentState)
workflow_data.add_node("Data_analyst", get_historical_prices)
workflow_data.set_entry_point("Data_analyst")
graph_data = workflow_data.compile()

workflow = StateGraph(AgentState)
workflow.add_node("Forecasting_expert_random_forest", predict_future_prices_random_forest)
workflow.add_node("Forecasting_expert_ARIMA", predict_future_prices_arima)
workflow.add_node("Evaluator", evaluate)
workflow.set_entry_point("Forecasting_expert_random_forest")
workflow.add_edge("Forecasting_expert_random_forest", "Forecasting_expert_ARIMA")
workflow.add_edge("Forecasting_expert_ARIMA", "Evaluator")
workflow.add_edge("Evaluator", END)
graph = workflow.compile()

memory = MemorySaver()
workflow_news = StateGraph(AgentState)
workflow_news.add_node("Investment_advisor", reccommend)
workflow_news.set_entry_point("Investment_advisor")
workflow_news.add_edge("Investment_advisor", END)
graph_news = workflow_news.compile(checkpointer=memory)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("counter", 0)
    elements = [cl.Image(name="image1", display="inline", path="./stock_image1.png", size="large")]
    await cl.Message(content="Welcome to ##StockSavvy!", elements=elements).send()
    await cl.Message(content="Enter a Company Name for Stock Recommendations from Our Chatbot").send()

@cl.on_message
async def main(message: cl.Message):
    counter = cl.user_session.get("counter")
    counter += 1
    cl.user_session.set("counter", counter)
    await cl.Message(content=f"You sent {counter} message(s)!").send()

    if counter == 1:
        inputs = {"messages": [HumanMessage(content=message.content)]}
        res_data = graph_data.invoke(inputs, config=RunnableConfig(callbacks=[
            cl.LangchainCallbackHandler(
                to_ignore=["ChannelRead", "RunnableLambda", "ChannelWrite", "__start__", "_execute"]
            )]))
        await cl.Message(content=res_data["messages"][-1].content).send()

        stockticker = (str(res_data).split(">>")[1] if len(str(res_data).split(">>")[1]) < 10 else str(res_data).split(">>")[0])
        df = u.get_stock_price(stockticker)
        df_history = u.historical_stock_prices(stockticker, 90)
        df_history_to_msg1 = eval(str(list((pd.DataFrame(df_history['Close'].values.reshape(1, -1)[0]).T).iloc[0, :])))
        inputs_all = {"messages": [HumanMessage(content=(f"Predict {stockticker}, historical prices are: {df_history_to_msg1}."))]}
        df_history = pd.DataFrame(df_history)
        df_history['stockticker'] = np.repeat(stockticker, len(df_history))
        df_history.to_csv('df_history.csv')

        res = graph.invoke(inputs_all, config=RunnableConfig(callbacks=[
            cl.LangchainCallbackHandler(
                to_ignore=["ChannelRead", "RunnableLambda", "ChannelWrite", "__start__", "_execute"]
            )]))
        await cl.Message(content=res["messages"][-2].content + '\n\n' + res["messages"][-1].content).send()

    df_history = pd.read_csv('df_history.csv')
    stockticker = str(df_history['stockticker'][0])
    df_search = search_news(stockticker)
    with open('search_news.txt', 'w') as a:
        a.write(str(df_search[0:10]))
    with open("search_news.txt", "r") as file:
        df_search = file.read()

    inputs_news = {"messages": [HumanMessage(content=(f"Summarize articles for {stockticker} to write 2 sentences about following articles: {df_search}."))]}
    for event in graph_news.stream(inputs_news, {"configurable": {"thread_id": "1"}}, stream_mode="values"):
        await cl.Message(content=event["messages"][-1].content).send()

    if counter == 1:
        df = u.historical_stock_prices(stockticker, 90)
        df = u.calculate_MACD(df, fast_period=12, slow_period=26, signal_period=9)
        fig = u.plot_macd2(df)
        elements = [cl.Pyplot(name="plot", figure=fig, display="inline", size="large")] if fig else []
        await cl.Message(content="Here is the MACD plot" if fig else "Failed to generate the MACD plot.", elements=elements).send()