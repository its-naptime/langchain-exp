from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers.pydantic import PydanticOutputParser

import getpass
import os
from dotenv import load_dotenv

import asyncio

class Recommendation(BaseModel):
    name: str = Field(description="Name of wine recommendation")
    reason: str = Field(description="Reason for recommendation")
    score: float = Field(description="Matching score with user query")

class RecommendationList(BaseModel):
    recommendations: List[Recommendation] = Field(description="List of matching wine recommendations")

async def handle_query(query):
    load_dotenv(dotenv_path='.env')

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass()
        
    db = SQLDatabase.from_uri("sqlite:///wines.db")

    llm = ChatOpenAI(model="gpt-4o-mini")

    parser = PydanticOutputParser(pydantic_object=RecommendationList)
    format_instructions = parser.get_format_instructions()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    tools = toolkit.get_tools()

    SQL_PREFIX = f"""You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables.
    Here is the user query: {query}
    
    When collating results, do not repeat the same wine. Some wines have duplicate rows due to different packaging sizes. Only give unique wines, very very important. I do not want any recommendations that sound the same.
    After querying and analyzing the results, provide your recommendations in the following format:
    {format_instructions}
    """

    system_message = SystemMessage(content=SQL_PREFIX)
    
    agent_executor = create_react_agent(
        llm, toolkit.get_tools(), state_modifier=system_message
    )
    
    loop = asyncio.get_event_loop()
    messages = await loop.run_in_executor(None, agent_executor.invoke, {"messages": [("human", query)]})
        
    return (query, parser.parse(messages["messages"][-1].content))

if __name__ == '__main__':
    response = asyncio.run(handle_query("recommend me some nice sweet red wine"))
    print(response)
