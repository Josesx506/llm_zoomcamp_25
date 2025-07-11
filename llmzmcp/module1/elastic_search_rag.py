from elasticsearch import Elasticsearch
from tqdm import tqdm

from llmzmcp.data import load_rag_documents
from llmzmcp.shared import esclient as es_client
from llmzmcp.module1.utils import build_prompt, llm

documents = load_rag_documents()

###########################################################################################
# Define the ES index settings, similar to a SQL db table schema
###########################################################################################
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": { # column types
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}
index_name = "course-questions"     # Name

# Create the index if it doesn't exist
if not es_client.indices.exists(index=index_name):
    print(f"Creating index: {index_name}")
    es_client.indices.create(index=index_name, body=index_settings)

    ###########################################################################################
    # Insert entries/documents into the index
    ###########################################################################################
    print("Populating index...")
    for doc in tqdm(documents, position=0):
        es_client.index(index=index_name, document=doc)
else:
    print(f"Index '{index_name}' already exists. Skipping creation and population.")


###########################################################################################
# utils
###########################################################################################
def elastic_search_query(query, course:str="data-engineering-zoomcamp"):
    """
    Query the index
    """
    search_query = {
        "size": 5, # Return 5 documents from index
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        # Weight the question field to be 3x as important
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": course
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    
    result_docs = []
    
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    
    return result_docs


def rag_elasticsearch(query):
    search_results = elastic_search_query(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


###########################################################################################
# Answer a question with context from the documents json
###########################################################################################
print("Using document context with GPT-4o and elastic search db.\n\n")

query = 'how do I run kafka?'
answer = rag_elasticsearch(query)
print(f"Query: {query}")
print(answer,"\n\n")

query = 'the course has already started, can I still enroll?'
answer = rag_elasticsearch(query)
print(f"Query: {query}")
print(answer)