import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()

# Retrieve sensitive information from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
neo4j_url = os.getenv("NEO4J_URL")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Ensure all necessary environment variables are set
if not all([openai_api_key, neo4j_url, neo4j_username, neo4j_password]):
    raise ValueError(
        "Missing one or more environment variables: OPENAI_API_KEY, NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD")

# Initialize the embedding provider
embedding_provider = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize the Neo4j graph connection
graph = Neo4jGraph(
    url=neo4j_url,
    username=neo4j_username,
    password=neo4j_password
)

# Define the documents to be indexed
documents = [
    Document(
        page_content="Text to be indexed",
        metadata={"source": "local"}
    )
]

# Create a new vector index from the documents
new_vector = Neo4jVector.from_documents(
    documents,
    embedding_provider,
    graph=graph,
    index_name="myVectorIndex",
    node_label="Chunk",
    text_node_property="text",
    embedding_node_property="embedding",
    create_id_index=True,
)

# Perform a similarity search
query = "A movie where aliens land and attack earth."
result = new_vector.similarity_search(query=query, k=5)

# Print the search results
for doc in result:
    print(doc.metadata['source'], "======", doc.page_content)

# Print the new vector index
print(new_vector)
