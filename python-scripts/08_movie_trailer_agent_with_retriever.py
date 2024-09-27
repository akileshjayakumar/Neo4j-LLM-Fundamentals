from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import StrOutputParser
from langchain_community.tools import YouTubeSearchTool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from uuid import uuid4

# Load environment variables
load_dotenv()

# Get environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Validate environment variables
if not all([OPENAI_API_KEY, NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise ValueError(
        "Missing one or more environment variables: OPENAI_API_KEY, NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD")

# Generate a session ID
SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

# Initialize LLM and embedding provider
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
embedding_provider = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Neo4j graph
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert. You find movies from a genre or plot."),
        ("human", "{input}"),
    ]
)

# Create the movie chat pipeline
movie_chat = prompt | llm | StrOutputParser()

# Initialize YouTube search tool
youtube = YouTubeSearchTool()

# Initialize Neo4j vector for movie plots
movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)

# Create the plot retriever
plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector.as_retriever()
)

# Function to get chat message history


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Function to call trailer search


def call_trailer_search(input):
    input = input.replace(",", " ")
    return youtube.run(input)


# Define tools
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
    Tool.from_function(
        name="Movie Plot Search",
        description="For when you need to compare a plot to a movie. The question will be a string. Return a string.",
        func=plot_retriever.invoke,
    ),
]


# Create the agent and agent executor
agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Create the chat agent with message history
chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Main loop to interact with the chat agent
while True:
    q = input("> ")
    response = chat_agent.invoke(
        {"input": q},
        {"configurable": {"session_id": SESSION_ID}},
    )
    print(response["output"])
