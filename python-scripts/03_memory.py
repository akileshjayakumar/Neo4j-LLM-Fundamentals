from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI chat model with API key
chat_llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY")
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

# Initialize chat message history
memory = ChatMessageHistory()


def get_memory(session_id):
    return memory


# Create the chat chain
chat_chain = prompt | chat_llm | StrOutputParser()

# Create the chat with message history runnable
chat_with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Define the current weather conditions
current_weather = """
{
    "surf": [
        {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
        {"beach": "Polzeath", "conditions": "Flat and calm"},
        {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
    ]
}
"""


def main():
    while True:
        try:
            question = input("> ")
            if question.lower() in ["exit", "quit"]:
                print("Exiting the chat. Goodbye!")
                break

            response = chat_with_message_history.invoke(
                {
                    "context": current_weather,
                    "question": question,
                },
                config={
                    "configurable": {"session_id": "none"}
                }
            )

            print(response)
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
