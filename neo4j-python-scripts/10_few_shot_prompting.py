import os

# Fetch schema from environment variable
SCHEMA = os.getenv('NEO4J_SCHEMA', 'default_schema')

CYPHER_GENERATION_TEMPLATE = f"""
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For movie titles that begin with "The", move "the" to the end, For example "The 39 Steps" becomes "39 Steps, The" or "The Matrix" becomes "Matrix, The".

If no data is returned, do not attempt to answer the question.
Only respond to questions that require you to construct a Cypher statement.
Do not include any explanations or apologies in your responses.

Examples: 

Find movies and genres:
MATCH (m:Movie)-[:IN_GENRE]->(g)
RETURN m.title, g.name

Schema: {SCHEMA}
Question: {{question}}
"""

# Example usage
question = "Find all movies directed by Christopher Nolan"
formatted_query = CYPHER_GENERATION_TEMPLATE.format(question=question)
print(formatted_query)
