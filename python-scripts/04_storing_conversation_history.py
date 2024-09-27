import os
from uuid import uuid4
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_message_histories import Neo4jChatMessageHistory

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Neo4j graph connection
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

print(graph)

# Generate a unique session ID
SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


# Initialize ChatOpenAI with API key
chat_llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY
)

# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
        ),
        ("system", "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Create the chat chain with message history
chat_chain = prompt | chat_llm | StrOutputParser()

chat_with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Current weather conditions
current_weather = """
{
    "surf": [
        {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
        {"beach": "Bells", "conditions": "Flat and calm"},
        {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
    ]
}"""

# Main loop to interact with the user
while True:
    question = input("> ")

    response = chat_with_message_history.invoke(
        {
            "context": current_weather,
            "question": question,
        },
        config={
            "configurable": {"session_id": SESSION_ID}
        }
    )

    print(response)
