import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


"""
The more frequently a word appears, the less important it is e.g. words like 
`the`, `yes`, etc. You can return an array for search results that are related 
to the query.
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

# Weight words to increase importance of infrequent words
# Also check for words that appear in at least 5 documents
cv = TfidfVectorizer(stop_words="english",min_df=5)
X = cv.fit_transform(df.text)

word_names = cv.get_feature_names_out()

# Creates a matrix of word importance
df_docs = pd.DataFrame(X.toarray(), columns=word_names)
df_docs = df_docs.round(2)


# Given an input query, transform it into a vector
query = "Do I need to know python to sign up for the January course?"
q = cv.transform([query])
q.toarray()

# Map the word names in the term document to the query to see the weight of each word
query_dict = dict(zip(word_names, q.toarray()[0]))

# Map the document ids to the query to see whihc documents are weighted to the query
doc_dict = dict(zip(word_names, X.toarray()[1]))

# The more words in common - the better the matching score. 
# Compute cosine similarity between bag of words and transformed query array
df_qd = pd.DataFrame([query_dict, doc_dict], index=['query', 'doc']).T
cos_sim_pd = (df_qd['query'] * df_qd['doc']).sum()
cos_sim_np = X.dot(q.T).toarray().flatten()    # numpy
cos_sim = cosine_similarity(X, q).flatten()    # Sklearn

# Rank the documents by importance by sorting the cosine similarity score
# Sorts in ascending order from lowest to highest
rank = np.argsort(cos_sim)
highest_doc_index = rank[-1]

print(f"Query: `{query}`")
print("\n#### Most related document to query is: \n\n",df.iloc[highest_doc_index].text,"\n")

############################ Perform search across all fields ############################
# Sum of scores for each column type
fields = ['section', 'question', 'text']
transformers = {}
matrices = {}

for field in fields:
    cv = TfidfVectorizer(stop_words='english', min_df=3)
    X = cv.fit_transform(df[field])

    transformers[field] = cv
    matrices[field] = X

# Search for most relatable documents across multiple fields
query2 = "I just signed up. Is it too late to join the course?"

# Initialize an array of zero scores
scores = np.zeros(len(df))

# Increase the weight for the question field
boost = {'question': 3.0} 
# Restrict entries to the one type of zoomcamp course
mask = (df.course == 'data-engineering-zoomcamp').values

for f in fields:
    b = boost.get(f, 1.0) # Default field weight is 1
    q = transformers[f].transform([query2])
    s = cosine_similarity(matrices[f], q).flatten()
    scores = scores + b * s * mask

rank = np.argsort(scores)
top3 = rank[-3:]

print(f"Query: `{query2}`")
print("\n#### Most related document to query is: \n\n",df.iloc[top3[0]].text,"\n")