from pprint import pprint

import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


"""
Embeddings is reducing the feature space of the bag of words to make the search 
more efficient. We'll use
- Singular Value Decomposition
- Non-Negative Matrix Factorization (Similar to svd without negative values)
"""

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []
for course in documents_raw:
    course_name = course['course']
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

df = pd.DataFrame(documents, columns=['course', 'section', 'question', 'text'])

############################### Generate embeddings with sklearn ######################################
fields = ['section', 'question', 'text']
transformers = {}
matrices = {}
svd_transformers = {}
svd_embeddings = {}
nmf_transformers = {}
nmf_embeddings = {}

for field in fields:
    cv = TfidfVectorizer(stop_words='english', min_df=3)
    X = cv.fit_transform(df[field])

    transformers[field] = cv
    matrices[field] = X

    # Reduce search space to 16 column embeddings
    svd = TruncatedSVD(n_components=16)
    svd_transformers[field] = svd
    svd_embeddings[field] = svd.fit_transform(X)

    # Extract nmf embeddings
    nmf = NMF(n_components=16)
    nmf_transformers[field] = nmf
    nmf_embeddings[field] = nmf.fit_transform(X)


### Word search for query
query = 'I just signed up. Is it too late to join the course?'

svd_scores = np.zeros(len(df))
nmf_scores = np.zeros(len(df))

# Restrict entries to the one type of zoomcamp course
mask = (df.course == 'data-engineering-zoomcamp').values

for f in fields:
    q = transformers[f].transform([query])

    svd_q_embd = svd_transformers[f].transform(q)
    svd_sim = cosine_similarity(svd_embeddings[f], svd_q_embd).flatten()
    svd_scores = svd_scores + svd_sim * mask

    nmf_q_embd = nmf_transformers[f].transform(q)
    nmf_sim = cosine_similarity(nmf_embeddings[f], nmf_q_embd).flatten()
    nmf_scores = nmf_scores + nmf_sim * mask


svd_rank = np.argsort(svd_scores)
svd_top3 = svd_rank[-3:]
svd_search_results = df.iloc[svd_top3].to_dict(orient='records')

nmf_rank = np.argsort(nmf_scores)
nmf_top3 = nmf_rank[-3:]
nmf_search_results = df.iloc[nmf_top3].to_dict(orient='records')

print(f"\nQuery: `{query}`\n")
print("\n################################ SVD ################################\n")
pprint(svd_search_results,indent=2,width=120)
print("\n################################ NMF ################################\n")
pprint(nmf_search_results,indent=2,width=120)