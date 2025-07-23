import numpy as np
from tqdm import tqdm
from llmzmcp.shared import esclient
from llmzmcp.shared import qdclient
from qdrant_client import models


###########################################################################################
# Define the metrics functions
###########################################################################################
def hit_rate(relevance_embedded_list):
    cnt = 0

    for line in relevance_embedded_list:
        # line is an array of booleans that tells you whether 
        # the correct document id was retrieved e.g. [False, False, True, False, False]
        query_total = np.array(line).sum()
        cnt += query_total

    return cnt / len(relevance_embedded_list)


def mrr(relevance_embedded_list):
    total_score = 0.0

    for line in relevance_embedded_list:
        doc_relevance = np.array(line).astype(int)
        doc_ranking = np.arange(len(doc_relevance))+1
        doc_total = (doc_relevance/doc_ranking).sum()
        total_score += doc_total

    return total_score / len(relevance_embedded_list)


def evaluate_search(ground_truth, search_function):
    """
    Evaluate either the Elastic search or min search algo.
    """
    relevance_list = []

    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_list.append(relevance)

    return {
        'hit_rate': round(hit_rate(relevance_list),3),
        'mrr': round(mrr(relevance_list),3),
    }


def elastic_search_query(index_name, query, course:str="data-engineering-zoomcamp"):
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

    response = esclient.search(index=index_name, body=search_query)
    
    result_docs = []
    
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    
    return result_docs


def minsearch_query(index, query:str, course:str='data-engineering-zoomcamp', 
                    boost = {'question': 3.0, 'section': 0.5}):
    results = index.search(
        query=query,
        filter_dict={'course': course},
        boost_dict=boost,
        num_results=5
    )

    return results


def minsearch_vector_query(vindex, vector, course, limit=5):
    """
    MinSearch with vector embeddings
    Returns top 5 results similar to input vector
    """
    return vindex.search(
        vector,
        filter_dict={'course': course},
        num_results=limit
    )


def qdrant_vector_query(query, collection_name, model_handle,
                        course="mlops-zoomcamp", limit=5):
    """Perform query with filter applied"""
    vector_points = qdclient.query_points(
        collection_name=collection_name,
        query=models.Document(text=query, model=model_handle),
        query_filter=models.Filter( 
            must=[models.FieldCondition(key="course",
                    match=models.MatchValue(value=course))
            ]
        ),
        limit=limit, with_payload=True
    )

    results = [point.payload for point in vector_points.points]

    return results