import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
neo4j_url = os.getenv("NEO4J_URL", "bolt://34.201.118.249:7687")
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "skirts-churns-terminologies")
index_name = os.getenv("NEO4J_INDEX_NAME", "moviePlots")
embedding_node_property = os.getenv(
    "NEO4J_EMBEDDING_NODE_PROPERTY", "plotEmbedding")
text_node_property = os.getenv("NEO4J_TEXT_NODE_PROPERTY", "plot")

# Initialize the language model
llm = ChatOpenAI(openai_api_key=openai_api_key)

# Initialize the embedding provider
embedding_provider = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize the Neo4j graph
graph = Neo4jGraph(
    url=neo4j_url,
    username=neo4j_username,
    password=neo4j_password
)

# Initialize the Neo4j vector store
movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name=index_name,
    embedding_node_property=embedding_node_property,
    text_node_property=text_node_property,
)

# Initialize the retriever
plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector.as_retriever(),
    verbose=False,
    return_source_documents=True
)

# Invoke the retriever with a query
response = plot_retriever.invoke("A movie where toys come to live.")

# Print the response
print(response["result"])
print(response)
