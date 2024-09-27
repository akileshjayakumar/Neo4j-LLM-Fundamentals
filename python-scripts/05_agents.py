import os
from uuid import uuid4
from dotenv import load_dotenv
from langchain_community.tools import YouTubeSearchTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import Neo4jGraph

# Load environment variables from .env file
load_dotenv()

# Initialize YouTube search tool
youtube = YouTubeSearchTool()

# Generate a unique session ID
SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

# Initialize the language model with the OpenAI API key from environment variables
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Function to call YouTube trailer search tool


def call_trailer_search(input):
    input = input.replace(",", " ")
    return youtube.run(input)


# Initialize Neo4j graph with credentials from environment variables
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert. You find movies from a genre or plot."),
        ("human", "{input}"),
    ]
)

# Create the movie chat pipeline
movie_chat = prompt | llm | StrOutputParser()

# Function to get chat message history from Neo4j


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


# Define the tools for the agent
tools = [
    Tool.from_function(
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=movie_chat.invoke,
    ),
    Tool.from_function(
        name="Movie Trailer Search",
        description="Use when needing to find a movie trailer. The question will include the word trailer. Return a link to a YouTube video.",
        func=call_trailer_search,
    ),
]

# Pull the agent prompt from the hub
agent_prompt = hub.pull("hwchase17/react-chat")

# Create the agent and agent executor
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Create the chat agent with message history
chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
    handle_parsing_errors=True,
)

# Main loop to interact with the chat agent
while True:
    q = input("> ")
    response = chat_agent.invoke(
        {"input": q},
        {"configurable": {"session_id": SESSION_ID}},
    )
    print(response["output"])
