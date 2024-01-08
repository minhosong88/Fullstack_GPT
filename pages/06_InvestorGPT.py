from langchain.schema import SystemMessage
import streamlit as st
import requests
import os
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

llm = ChatOpenAI(temperature=0.1)
alpha_vantage_api_key = os.environ.get("ALPHAVANTAGE_API")

class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for. Example query: Stock Market Symbol for Apple Company")
    
class CompanyOverviewArgsSchema(BaseModel):
    symbol: str = Field(description="Stock symbol of the company. Example: APPL, TSLA")

class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = """
    Use this tool to find the stock market symbol for a company.
    It takes a query as an argument.
    """
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = StockMarketSymbolSearchToolArgsSchema
    
    def _run(self,query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)

class CompanyOverviewTool(BaseTool):
    name="CompanyOverview"
    description="""
    Use this to get an overview of the financials of the company.
    You sould enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema
    def _run(self, symbol):
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}"
        r = requests.get(url)
        return r.json()
    
class CompanyIncomeStatementTool(BaseTool):
    name="CompanyIncomeStatement"
    description="""
    Use this to get the income statement of the company.
    You sould enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema
    def _run(self, symbol):
        url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}"
        r = requests.get(url)
        return r.json()["annualReports"]
    
class CompanyStockPerformanceTool(BaseTool):
    name="CompanyStockPerformance"
    description="""
    Use this to get the weekly performance of the company.
    You sould enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema
    def _run(self, symbol):
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}"
        r = requests.get(url)
        responses = r.json()
        # Getting a list of the first 200 key-value pairs for the results and return
        return list(responses["Weekly Time Series"].items())[:200]

     
agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    # handle errors automatically
    handle_parsing_errors=True,
    # Add a list of tools
    tools=[
       StockMarketSymbolSearchTool(),
       CompanyOverviewTool(),
       CompanyIncomeStatementTool(),
       CompanyStockPerformanceTool(),
    ],  
    agent_kwargs={
        #Update the system message, which is  "You are a helpful assistant" in default
        "system_message": SystemMessage(content="""
        You are a hedge fund manager.
            You evaluate a company and provider your opinion and reasons why the stock is a buy or not.
            Consider the performance of a stock, the company overview and the income statement.
            Be assertive in your judgement and recommend the stock or advise the user against it.
            """)
    }
    
)
prompt = "Give me financial information on Cloudflare's stock, considering its financials, income statement and stock performance, and help me analyze if it's potenstial good investment."

st.set_page_config(
    page_title="InvestorGPT",
    page_icon="📈",
)
st.markdown(
    """
    # InvestorGPT
    
    Welcome to InvestorGPT.
    
    Write down the name of a company and our Agent will do the research for you.
    """
)

company = st.text_input("Write the name of the company you are interested in.")

if company:
    result = agent.invoke(company)
    st.write(result["output"].replace("$","\$"))