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
from tools import data_analyst
from tools import stock_sentiment_analysis_util
import functools
from typing import Annotated
import operator
from typing import Sequence, TypedDict
from langchain.agents import initialize_agent,  Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import yfinance as yf
import functools
from typing import Annotated
import operator
from typing import Sequence, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from tools import data_analyst, forecasting_expert_arima, forecasting_expert_rf, evaluator, investment_advisor
from chainlit.input_widget import Select
import matplotlib.pyplot as plt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai.embeddings import OpenAIEmbeddings
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

load_dotenv()
HF_ACCESS_TOKEN = os.environ["HF_ACCESS_TOKEN"]
DAYS_TO_FETCH_NEWS = os.environ["DAYS_TO_FETCH_NEWS"]
NO_OF_NEWS_ARTICLES_TO_FETCH = os.environ["NO_OF_NEWS_ARTICLES_TO_FETCH"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


from GoogleNews import GoogleNews

def search_news(stockticker):
    """Useful to search the internet for news about a given topic and return relevant results."""
    # Set the number of top news results to return
    googlenews = GoogleNews()
    googlenews.set_period('7d')
    googlenews.get_news(stockticker)
    result_string=googlenews.get_texts()

    return result_string


def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

llm = ChatOpenAI(model="gpt-3.5-turbo")

#======================== AGENTS ==================================
# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

# DATA ANALYST
prompt_data_analyst="You are a stock data analyst.\
                Provide correct stock ticker from Yahoo Finance.\
                Expected output: stocticker.\
                Provide it in the following format: >>stockticker>> \
                for example: >>AAPL>>"
                
tools_data_analyst=data_analyst.data_analyst_tools()
data_agent = create_agent(
    llm,
    tools_data_analyst,
    prompt_data_analyst)
get_historical_prices = functools.partial(agent_node, agent=data_agent, name="Data_analyst")

#ARIMA Forecasting expert
prompt_forecasting_expert_arima="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are stock prediction expert, \
               take historical stock data from message and train the ARIMA model from statsmodels Python library on the last week,then provide prediction for the 'Close' price for the next day.\
               Give the value for mae_arima to Evaluator.\
               Expected output:list of predicted prices with predicted dates for a selected stock ticker and mae_arima value.\n
               <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

tools_forecasting_expert_arima=forecasting_expert_arima.forecasting_expert_arima_tools()
code_forecasting_arima = create_agent(
    llm,
    tools_forecasting_expert_arima,
    prompt_forecasting_expert_arima,
)
predict_future_prices_arima = functools.partial(agent_node, agent=code_forecasting_arima, name="Forecasting_expert_ARIMA")

# RF  Forecasting expert
prompt_forecasting_expert_random_forest="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are stock prediction expert, \
               take historical stock data from message and train the Random forest model from statsmodels Python library on the last week,then provide prediction for the 'Close' price for the next day.\
               Give the value for mae_rf to Evaluator.\
               Expected output:list of predicted prices with predicted dates for a selected stock ticker and mae_rf value.\n
               <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

tools_forecasting_expert_random_forest=forecasting_expert_rf.forecasting_expert_rf_tools()
code_forecasting_random_forest = create_agent(
    llm,
    tools_forecasting_expert_random_forest,
    prompt_forecasting_expert_random_forest,
)
predict_future_prices_random_forest = functools.partial(agent_node, agent=code_forecasting_random_forest, name="Forecasting_expert_random_forest")

# EVALUATOR
prompt_evaluator="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are an evaluator retrieve arima_prediction and arima mean average error from forecasting expert arima and rf_prediction and mean average error for random forest from forecasting expert random forest\
                print final prediction number. 
                Next, compare prediction price and current price to provide reccommendation if he should buy/sell/hold the stock. \
                 Expected output: one value for the prediction, explain why you have selected this value, reccommendation  buy or sell stock and why.\
                  <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

tools_evaluate=evaluator.evaluator_tools()
code_evaluate = create_agent(
    llm,
    tools_evaluate,
    prompt_evaluator,
)
evaluate = functools.partial(agent_node, agent=code_evaluate, name="Evaluator")

# Investment advisor
prompt_inv_advisor="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                Provide personalized investment advice and recommendations and analyze historical stock prices if asked.\
                Consider user input message for the latest news on the stock.\
                Provide overall sentiment of the news Positive/Negative/Neutral, and recommend if  the user should invest in such stock.\
                  <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

tools_reccommend=investment_advisor.investment_advisor_tools()

code_inv_advisor = create_agent(
    llm,
    tools_reccommend,
    prompt_inv_advisor,
)

reccommend = functools.partial(agent_node, agent=code_inv_advisor, name="Investment_advisor")

workflow_data = StateGraph(AgentState)
workflow_data.add_node("Data_analyst", get_historical_prices)
workflow_data.set_entry_point("Data_analyst")
graph_data=workflow_data.compile()

workflow = StateGraph(AgentState)
#workflow.add_node("Data_analyst", get_historical_prices)
workflow.add_node("Forecasting_expert_random_forest", predict_future_prices_random_forest)
workflow.add_node("Forecasting_expert_ARIMA", predict_future_prices_arima)
workflow.add_node("Evaluator", evaluate)


# Finally, add entrypoint
workflow.set_entry_point("Forecasting_expert_random_forest")
workflow.add_edge("Forecasting_expert_random_forest","Forecasting_expert_ARIMA")
workflow.add_edge("Forecasting_expert_ARIMA","Evaluator")
workflow.add_edge("Evaluator",END)
graph = workflow.compile()

""" memory = MemorySaver()
workflow_news = StateGraph(AgentState)
workflow_news.add_node("Investment_advisor", reccommend)
workflow_news.set_entry_point("Investment_advisor")
workflow_news.add_edge("Investment_advisor",END)
graph_news = workflow_news.compile(checkpointer=memory) """
#Print graph
#graph.get_graph().print_ascii()

from langchain_core.runnables import RunnableConfig
from chainlit import AskUserMessage
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("counter", 0)

    # Sending an image with the local file path
    elements = [
    cl.Image(name="image1", display="inline", path="./good_day.png",size="large")
    ]
    await cl.Message(content="Hello there, Welcome to ##StockSavyy!", elements=elements).send()
    await cl.Message(content="Tell me the stockticker you want me to analyze.").send()
    question_array='These are my questions.'
    cl.user_session.set("question_array",question_array)
    question=''
    response=''
    cl.user_session.set("question",question)

@cl.on_message
async def main(message: cl.Message):
    #"what is the weather in sf"
    counter = cl.user_session.get("counter")
    counter += 1
    cl.user_session.set("counter", counter)
    question_array=cl.user_session.get("question_array")
    question_array += (f"Question: {message.content}")
    cl.user_session.set("question_array", question_array)
    #question=cl.user_session.get("question")
    
    await cl.Message(content=f"You sent {counter} message(s)!").send()
    #if counter==1:
    inputs = {"messages": [HumanMessage(content=message.content)]}
    #print(str(message.content))
    
    #Checking if input message is a stock search, assumption here is that if user types a stockticker explicity or
    #inputs the name of the company for app to find stockticker the lenght of input won't be greater than 15
    if len(str(message.content)) <= 15:

        res_data = graph_data.invoke(inputs, config=RunnableConfig(callbacks=[
            cl.LangchainCallbackHandler(
                to_ignore=["ChannelRead", "RunnableLambda", "ChannelWrite", "__start__", "_execute"]
                # can add more into the to_ignore: "agent:edges", "call_model"
                # to_keep=

            )]))
        #print(res_data)
        await cl.Message(content=res_data["messages"][-1].content).send()
        print('ticker',str(res_data).split(">>")[0])
        if len(str(res_data).split(">>")[1])<10:
            stockticker=(str(res_data).split(">>")[1])
        else:
            stockticker=(str(res_data).split(">>")[0])


        #print('ticker1',stockticker)
        print('here')
        df=u.get_stock_price(stockticker)
        df_history=u.historical_stock_prices(stockticker,90)
        df_history_to_msg1=eval(str(list((pd.DataFrame(df_history['Close'].values.reshape(1, -1)[0]).T).iloc[0,:])))

        inputs_all = {"messages": [HumanMessage(content=(f"Predict {stockticker}, historical prices are: {df_history_to_msg1}."))]}
        df_history=pd.DataFrame(df_history)
        df_history['stockticker']=np.repeat(stockticker,len(df_history))
        df_history.to_csv('df_history.csv')
        #df_history.to_csv('./tools/df_history.csv')

        print ("Running forecasting models on historical prices")
        res = graph.invoke(inputs_all, config=RunnableConfig(callbacks=[
            cl.LangchainCallbackHandler(
                to_ignore=["ChannelRead", "RunnableLambda", "ChannelWrite", "__start__", "_execute"]
                # can add more into the to_ignore: "agent:edges", "call_model"
                # to_keep=

            )]))
        await cl.Message(content= res["messages"][-2].content + '\n\n' + res["messages"][-1].content).send()

        #Storing recommendation
        recommendation = "Recommendation for " + stockticker + '\n' + res["messages"][-2].content + '\n\n' + res["messages"][-1].content


        #Plotting the graph
        df=u.historical_stock_prices(stockticker,90)
        df=u.calculate_MACD(df, fast_period=12, slow_period=26, signal_period=9)
        #df values
        #Index(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits','EMA_fast', 'EMA_slow', 'MACD', 'Signal_Line', 'MACD_Histogram']
        fig = u.plot_macd2(df)
        
        if fig:
            elements = [cl.Pyplot(name="plot", figure=fig, display="inline",size="large"),
            ]
            await cl.Message(
                content="Here is the MACD plot",
                elements=elements,
            ).send()
        else:
            await cl.Message(
                content="Failed to generate the MACD plot."
            ).send()


        #Perform sentiment analysis on the stock news & predict dominant sentiment along with plotting the sentiment breakdown chart
        news_articles = stock_sentiment_analysis_util.fetch_news(stockticker)

        analysis_results = []
        
        #Perform sentiment analysis for each product review
        for article in news_articles:
            sentiment_analysis_result = stock_sentiment_analysis_util.analyze_sentiment(article['News_Article'])

            # Display sentiment analysis results
            #print(f'News Article: {sentiment_analysis_result["News_Article"]} : Sentiment: {sentiment_analysis_result["Sentiment"]}', '\n')

            result = {
                        'News_Article': sentiment_analysis_result["News_Article"],
                        'Sentiment': sentiment_analysis_result["Sentiment"][0]['label']
                    }
            
            analysis_results.append(result)

        
        #Retrieve dominant sentiment based on sentiment analysis data of reviews
        dominant_sentiment = stock_sentiment_analysis_util.get_dominant_sentiment(analysis_results)
        await cl.Message(
                content="Dominant sentiment of the stock based on last 7 days of news is : " + dominant_sentiment
            ).send()
        
        #Plot sentiment breakdown chart
        fig = stock_sentiment_analysis_util.plot_sentiment_graph(analysis_results)
        if fig:
            elements = [cl.Pyplot(name="plot", figure=fig, display="inline",size="large"),
            ]
            await cl.Message(
                content="Sentiment breakdown plot",
                elements=elements,
            ).send()
        else:
            await cl.Message(
                content="Failed to generate the MACD plot."
            ).send()

        #Generate summarized message rationalize dominant sentiment
        summary = stock_sentiment_analysis_util.generate_summary_of_sentiment(analysis_results, dominant_sentiment)
        await cl.Message(
                content= summary
            ).send()
        

        #Storing sentiment summary
        recommendation = recommendation + '\n' + "Stock sentiment summary for " + stockticker + ' is,  \n' + summary + '\n and dominant sentiment for stock is ' + dominant_sentiment
        print("******************************************************")
        print(recommendation)
        print("******************************************************")
        answers=np.append(res["messages"][-1].content,summary)
        with open('answers.txt', 'w') as a:
            a.write(str(answers))


        #Adding messages to Qdrant in memory store, to provide users ability to ask questions based on the recommmendation and sentiment summarization
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 250,
            chunk_overlap = 20
        )
        recommendation_chunks = text_splitter.split_text(recommendation)
        # Convert the chunks into Document objects
        from langchain.schema import Document
        documents = [Document(page_content=chunk) for chunk in recommendation_chunks]

        #4 Store embeddings in QDrant vector store in memory
        from langchain_community.vectorstores import Qdrant
        qdrant_vector_store = Qdrant.from_documents(
            documents,
            OpenAIEmbeddings(model="text-embedding-3-small"),
            location=":memory:",
            collection_name="Stock Analysis",
        )
        qdrant_retriever = qdrant_vector_store.as_retriever()

        #Setting up RAG Prompt Template
        from langchain_core.prompts import PromptTemplate
        RAG_PROMPT_TEMPLATE = """\
        <|start_header_id|>system<|end_header_id|>
        You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

        <|start_header_id|>user<|end_header_id|>
        User Query:
        {question}

        Context:
        {context}<|eot_id|>

        <|start_header_id|>assistant<|end_header_id|>
        """
        rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        from langchain.memory import ConversationBufferMemory
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        # Instantiate ConversationBufferMemory
        memory = ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question"
        )
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        # First, load the memory to access chat history
        loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
        )
        retrieval_augmented_qa_chain = (loaded_memory|
        {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | llm, "context": itemgetter("context")}
        )
        cl.user_session.set("lcel_rag_chain", retrieval_augmented_qa_chain)
        

    else:
        #question_array=question_array+message.content
        print('questions', question_array)
        file = open("answers.txt", "r")
        answers = file.read()

        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 250,
            chunk_overlap = 20
        )
        recommendation_chunks = text_splitter.split_text(answers)
        question_chunks = text_splitter.split_text(question_array)
        all_chunks=recommendation_chunks+question_chunks
        # Convert the chunks into Document objects
        from langchain.schema import Document
        documents = [Document(page_content=chunk) for chunk in all_chunks] #recommendation_chunks]

        #4 Store embeddings in QDrant vector store in memory
        from langchain_community.vectorstores import Qdrant
        qdrant_vector_store = Qdrant.from_documents(
            documents,
            OpenAIEmbeddings(model="text-embedding-3-small"),
            location=":memory:",
            collection_name="Stock Analysis",
        )
        qdrant_retriever = qdrant_vector_store.as_retriever()

        #Setting up RAG Prompt Template
        from langchain_core.prompts import PromptTemplate
        RAG_PROMPT_TEMPLATE = """\
        <|start_header_id|>system<|end_header_id|>
        You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

        <|start_header_id|>user<|end_header_id|>
        User Query:
        {question}

        Context:
        {context}<|eot_id|>

        <|start_header_id|>assistant<|end_header_id|>
        """
        rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        from langchain.memory import ConversationBufferMemory
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        # Instantiate ConversationBufferMemory
        memory = ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question"
        )
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        # First, load the memory to access chat history
        loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
        )
        retrieval_augmented_qa_chain = (loaded_memory|
        {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | llm, "context": itemgetter("context")}
        )
        #retrieve lcel chain
        cl.user_session.set("lcel_rag_chain", retrieval_augmented_qa_chain)

        #retrieve lcel chain
        lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

        question = message.content
        print("Query : " + question)

        result =  lcel_rag_chain.invoke({"question" : question})
        await cl.Message(
                content= result["response"].content
            ).send()
        response=result["response"].content
        question_array += (f"Answer: {response}")
        print(response)
        print(question_array)

    
    