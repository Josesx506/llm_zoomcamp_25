"""
Show how to evaluate MinSearch vs. ElasticSearch retrieval performance 
on the ground truth dataset.
"""

import json

import minsearch
import numpy as np
from tqdm import tqdm

from llmzmcp.data import load_eval_documents, load_ground_truth_questions
from llmzmcp.module3.functions import evaluate_search
from llmzmcp.shared import esclient as es_client

# Load the eval documents that contain the document id
documents = load_eval_documents()
ground_truth = load_ground_truth_questions().to_dict(orient="records")

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
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
        }
    }
}
index_name = "course-questions-with-ids"     # DB Name

# Create the index if it doesn't exist
if not es_client.indices.exists(index=index_name):
    print(f"Creating index: {index_name}")
    # es_client.indices.delete(index=index_name, ignore_unavailable=True) # Delete old index if you need to recreate
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
# Define the index properties for minsearch
###########################################################################################
index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course", "id"]
)

index.fit(documents)


###########################################################################################
# Define the search function for elastic search to use the updated index name
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

def minsearch_query(query:str, course:str='data-engineering-zoomcamp'):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': course},
        boost_dict=boost,
        num_results=5
    )

    return results


# Evaluate both search functions
es_res = evaluate_search(ground_truth, lambda q: elastic_search_query(q['question'], q['course']))
ms_res = evaluate_search(ground_truth, lambda q: minsearch_query(q['question'], q['course']))

print("\n\n","Elastic search results:\n", json.dumps(es_res, indent=2))
print("\n\n","Min. search results:\n", json.dumps(ms_res, indent=2))