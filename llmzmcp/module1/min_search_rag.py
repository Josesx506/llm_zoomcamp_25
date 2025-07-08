import json
from pprint import pprint

import minsearch

from llmzmcp.data import load_rag_documents
from llmzmcp.module1.client import oaiclient as client
from llmzmcp.module1.utils import build_prompt, llm

documents = load_rag_documents()

# Index the documents - create a bag of key words dataframe
index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)
index.fit(documents)


###########################################################################################
# Answer a question without providing context
###########################################################################################
q = 'the course has already started, can I still enroll?'

response = client.chat.completions.create(
    model='gpt-4o',
    messages=[{"role": "user", "content": q}]
)

print("No context provided\n\n")
print(f"Query: {q}")
print(response.choices[0].message.content)


###########################################################################################
# utils
###########################################################################################
def search_course(query:str, course:str='data-engineering-zoomcamp'):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': course},
        boost_dict=boost,
        num_results=5
    )

    return results

def rag_minsearch(query):
    search_results = search_course(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer




###########################################################################################
# Answer a question with context from the documents json
###########################################################################################
print("\n\nUsing document context with GPT-4o.\n\n")

query = 'how do I run kafka?'
answer = rag_minsearch(query)
print(f"Query: {query}")
print(answer,"\n\n")

query = 'the course has already started, can I still enroll?'
answer = rag_minsearch(query)
print(f"Query: {query}")
print(answer)
