import os
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI as Chat
from langchain.schema import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o")
temperature = float(os.getenv("OPENAI_TEMPERATURE", 0))

NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

llm = Chat(
    openai_api_key=openai_api_key,
    model=model,
    temperature=temperature
)

print(llm)

embeddings = OpenAIEmbeddings()

print(embeddings)

# Initialize Neo4j graph connection
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Initialize Neo4j vector store connection
vector_store = Neo4jVector(
    embeddings,
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

print(graph)
print()
print(vector_store)

documents = graph.query("MATCH (n) RETURN n LIMIT 25;")

# Convert the retrieved documents into Document objects
document_objects = [Document(page_content=str(doc)) for doc in documents]

db = Neo4jVector.from_documents(
    document_objects, embeddings, url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
)

print(db)

stack_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=db.as_retriever()
)

print(stack_retriever)

response = stack_retriever.invoke("what is in the db?")

print(response["result"])
