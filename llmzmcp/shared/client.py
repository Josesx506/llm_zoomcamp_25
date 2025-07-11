from elasticsearch import Elasticsearch
from openai import OpenAI
from qdrant_client import QdrantClient

# Create an open ai client. Ensure the account is funded
oaiclient = OpenAI()

# Create an elastic search client
esclient = Elasticsearch(hosts=[{"host":"elasticsearch", "port":9200, "scheme":"http"}]) 

# Create a Qdrant client
# QdrantClient("http://localhost:6333") # connecting to local Qdrant instance
qdclient = QdrantClient(host="qdrant", port=6333) # devcontainer