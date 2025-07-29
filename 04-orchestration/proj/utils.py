import requests
import pandas as pd
import minsearch
from prompts import wiki_prompt_template
from llmzmcp.shared import oaiclient as client

mindex = minsearch.Index(
    text_fields=["Question", "Answer"],
    keyword_fields=["ArticleTitle", "ArticleFile"]
)


def load_and_index_documents(file_path:str):
    file = pd.read_csv(file_path)
    documents = file.to_dict(orient="records")
    mindex.fit(documents)
    return True


def minsearch_query(query:str):
    """
    Search for the most similar documents to generate context
    """
    results = mindex.search(
        query=query,
        num_results=5
    )

    return results


def build_prompt(query:str, search_results:list, prompt_template:str):
    context = ""
    
    for doc in search_results:
        context = context + f"question: {doc['Question']}\nanswer: {doc['Answer']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


def llm(prompt):
    response = requests.post("http://ollama:11434/api/chat", json={
        "model": "gemma3:1b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    })
    return response.json()["message"]["content"]

    # response = client.chat.completions.create(
    #     model='gpt-4o-mini',
    #     messages=[{"role": "user", "content": prompt}],
    # )
    
    # return response.choices[0].message.content


def rag(query):
    search = minsearch_query(query)
    prompt = build_prompt(query, search, wiki_prompt_template)
    result = llm(prompt)
    return result