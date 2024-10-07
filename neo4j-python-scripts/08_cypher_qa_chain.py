import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Initialize the language model with the OpenAI API key from environment variables
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the Neo4j graph connection using environment variables
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# Define the Cypher generation template
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Schema: {schema}
Question: {question}
"""

# Create a prompt template for Cypher generation
cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

# Initialize the GraphCypherQAChain with the language model, graph, and prompt template
cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True
)

# Invoke the chain with a sample query and print the result
response = cypher_chain.invoke(
    {"query": "How many movies has James Gunn directed?"}
)
print()
print()
print()
print(response["result"])
