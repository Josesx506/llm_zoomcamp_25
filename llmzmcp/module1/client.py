from openai import OpenAI
from elasticsearch import Elasticsearch

# Create an open ai client. Ensure the account is funded
oaiclient = OpenAI()

# Create an elastic search client
esclient = Elasticsearch(hosts=[{"host": "elasticsearch", "port": 9200, "scheme": "http"}]) 