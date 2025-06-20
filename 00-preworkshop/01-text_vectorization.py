import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


### Example test for word vectorizer. 
# Each string is like a dummy document so there are 5 docs in total
# Each word is turned into a vector to create the term-document matrix
cv = CountVectorizer()
docs_example = [
    "January course details, register now",
    "Course prerequisites listed in January catalog",
    "Submit January course homework by end of month",
    "Register for January course, no prerequisites",
    "January course setup: Python and Google Cloud"
]
cv.fit(docs_example)
word_names = cv.get_feature_names_out()
X = cv.transform(docs_example)
# Originally rows are documents and columns are words/token
# Transpose it so that the columns are documents to create a bag of words
# Order is not preserved and we only care about word occurrence
df_docs = pd.DataFrame(X.toarray(), columns=word_names).T

print(df_docs.head())
