import numpy as np
from tqdm import tqdm
import json
from utils import load_and_index_documents, minsearch_query

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
        doc_id = q['id']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_list.append(relevance)

    return {
        'hit_rate': round(hit_rate(relevance_list),3),
        'mrr': round(mrr(relevance_list),3),
    }


if __name__=="__main__":
    docs = load_and_index_documents("mini_wiki.csv")
    ms_res = evaluate_search(docs, lambda q: minsearch_query(q["Question"]))
    print(json.dumps(ms_res, default=str, indent=2))