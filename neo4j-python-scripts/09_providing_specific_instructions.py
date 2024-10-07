import os

# Fetch schema and question from environment variables
schema = os.getenv('NEO4J_SCHEMA', 'default_schema')
question = os.getenv('USER_QUESTION', 'default_question')

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For movie titles that begin with "The", move "the" to the end, For example "The 39 Steps" becomes "39 Steps, The" or "The Matrix" becomes "Matrix, The".

Schema: {schema}
Question: {question}
"""

# Format the template with the schema and question
formatted_template = CYPHER_GENERATION_TEMPLATE.format(
    schema=schema, question=question)

# Print the formatted template
print(formatted_template)
