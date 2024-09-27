from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-instruct")
temperature = float(os.getenv("OPENAI_TEMPERATURE", 0))

llm = OpenAI(
    openai_api_key=openai_api_key,
    model=model,
    temperature=temperature
)

template = PromptTemplate.from_template("""
You are a singaporean fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using singlish.
""")

llm_chain = template | llm | StrOutputParser()

response = llm_chain.invoke({"question": "i want to buy 10 thousnad apples."})

print(response)
