from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI as Chat
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o")
temperature = float(os.getenv("OPENAI_TEMPERATURE", 0))

llm = Chat(
    openai_api_key=openai_api_key,
    model=model,
    temperature=temperature
)

template = PromptTemplate.from_template("""
You are a singaporean fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using singlish and provide the customer with the best possible service.
{question}
""")

llm_chain = template | llm | StrOutputParser()

response = llm_chain.invoke({"question": "I want to buy 1 million durians."})

print(response)
