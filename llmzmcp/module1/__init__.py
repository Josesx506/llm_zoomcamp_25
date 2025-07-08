"""
This module is all about performing RAG using context with GPT-4o. First part uses
- MinSearch - a simple object class for indexing documents in memory that can be used
    to provide context for an llm.
- ElasticSearch - a type of NoSQL db for indexing documents and searching for results 
    across different fields/columns. ES persists data outside memory as is run as a 
    docker service in a dev container.
"""