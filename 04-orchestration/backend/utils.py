from time import time

import minsearch
import pandas as pd
import numpy as np
import requests
import tiktoken
from models import Messages
from prompts import wiki_prompt_template
from sqlmodel import Session

from llmzmcp.shared import oaiclient as client

mindex = minsearch.Index(
    text_fields=["Question", "Answer"],
    keyword_fields=["ArticleTitle", "ArticleFile"]
)


def load_and_index_documents(file_path:str):
    file = pd.read_csv(file_path)
    file["id"] = np.arange(len(file))+1
    documents = file.to_dict(orient="records")
    mindex.fit(documents)
    return documents


def minsearch_query(query:str):
    """
    Search for the most similar documents to generate context
    """
    results = mindex.search(
        query=query,
        num_results=5
    )

    return results


def get_recent_messages(session: Session, conv_id: int, limit: int = 3):
    """
    Get recent messages for conversation history context
    """
    return (
        session.query(Messages)
        .with_entities(Messages.question, Messages.answer)
        .filter(Messages.conv_id == conv_id)
        .order_by(Messages.timestamp.desc())
        .limit(limit)
        .all()[::-1]# reverse to chronological order
    )


def build_prompt(query:str, history:list, search_results:list, prompt_template:str): 
    # Format chat history
    chat_history = "\n".join(
        f"User: {msg.question}\nAssistant: {msg.answer}" for msg in history
    )

    # Format RAG search context
    rag_context = "\n\n".join(
        f"Question: {doc['Question']}\nAnswer: {doc['Answer']}" for doc in search_results
    )

    # Final formatted prompt
    prompt = prompt_template.format(
        question=query,
        history=chat_history,
        context=rag_context
    ).strip()

    return prompt


def token_counter(string:str, encoding_name:str="cl100k_base")->int:
    """
    Token estimator for chatgpt
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def calculate_openai_cost(model_choice, tokens):
    openai_cost = 0

    if model_choice == 'gpt-3.5-turbo':
        openai_cost = (tokens['prompt_tokens'] * 0.0015 + tokens['completion_tokens'] * 0.002) / 1000
    elif model_choice in ['gpt-4o', 'gpt-4o-mini']:
        openai_cost = (tokens['prompt_tokens'] * 0.03 + tokens['completion_tokens'] * 0.06) / 1000

    return openai_cost


def llm(prompt, model="gemma3:1b"):
    start_time = time()

    match model:
        case "gemma3:1b":
            response = requests.post("http://ollama:11434/api/chat", json={
                "model": "gemma3:1b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            })
            output = response.json()["message"]["content"]
            prompt_tokens = token_counter(prompt)
            completion_tokens = token_counter(output)
            tokens = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            }
        case _ if model.startswith("gpt"):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            tokens = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            output = response.choices[0].message.content
        case _:
            output = "Invalid model name, no response provided"
    
    end_time = time()
    resp_time = end_time - start_time

    return output, tokens, resp_time


def rag(query:str, history:list=[], model:str="gemma3:1b"):
    search = minsearch_query(query)
    prompt = build_prompt(query, history, search, wiki_prompt_template)
    result, tokens, resp_time = llm(prompt, model)
    return result, tokens, resp_time
