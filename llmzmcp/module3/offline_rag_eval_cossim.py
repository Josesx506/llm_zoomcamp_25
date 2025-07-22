"""
Compare answers to ground truth questions from an LLM to an actual human answer. Compute the cosine similarity 
of the embedding vectors for both answers to evaluate how well a model performed.
The zoomcamp provides responses from the gpt-4o and gpt-3.5-turbo. This costs quite a number of tokens
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from minsearch import VectorSearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from llmzmcp.data import (load_eval_documents, load_ground_truth_questions,
                          load_llm_eval_dataframes)
from llmzmcp.shared import multithread_func
from llmzmcp.shared import oaiclient as client

###########################################################################################
# Load the evaluation data and create a simple lookup index
###########################################################################################
documents = load_eval_documents()
ground_truth = load_ground_truth_questions()
doc_idx = {d['id']: d for d in documents}


###########################################################################################
# Generate vector embeddings of documents using sentence transformer for MinSearch
###########################################################################################
model_name = 'multi-qa-MiniLM-L6-cos-v1'
model = SentenceTransformer(model_name)
vectors = []

for doc in tqdm(documents, position=0):
    question = doc['question']
    text = doc['text']
    vector = model.encode(question + ' ' + text)
    vectors.append(vector)
vectors = np.array(vectors)

vindex = VectorSearch(keyword_fields=['course'])
vindex.fit(vectors, documents)


###########################################################################################
# Helper functions to perform vector search with MinSearch
###########################################################################################
def minsearch_vector_search(vector, course):
    """
    Returns top 5 results similar to input vector
    """
    return vindex.search(
        vector,
        filter_dict={'course': course},
        num_results=5
    )

def question_text_vector(q):
    question = q['question']
    course = q['course']

    v_q = model.encode(question)

    return minsearch_vector_search(v_q, course)


###########################################################################################
# Helper functions for RAG prompt and llm api call
###########################################################################################
def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""
    
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


def llm(prompt, model='gpt-4o'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


def rag(query: dict, model='gpt-4o') -> str:
    search_results = question_text_vector(query)
    prompt = build_prompt(query['question'], search_results)
    answer = llm(prompt, model=model)
    return answer


def process_record(rec, model='gpt-3.5-turbo'):
    """
    Fetch llm answers from openai api and save it into a dict 
    that can be converted into a dataframe.

    Improved models like gpt-4o can be more expensive
    """
    answer_llm = rag(rec, model=model)
    
    doc_id = rec['document']
    original_doc = doc_idx[doc_id]
    answer_orig = original_doc['text']

    # The llm and original answer can be compared using cosine similarity
    return {
        'answer_llm': answer_llm,
        'answer_orig': answer_orig,
        'document': doc_id,
        'question': rec['question'],
        'course': rec['course'],
    }


###########################################################################################
# Perform api calls to generate the llm answers. You can change the default model in 
# `process_record()` to compare multiple models
###########################################################################################
# Select only 10 samples to minimize api costs
# ground_truth_list = ground_truth[:10].to_dict(orient="records")
# results_gpt35 = multithread_func(ground_truth_list, process_record)
# df_gpt35 = pd.DataFrame(results_gpt35)
# df_gpt35.to_csv("gpt-3.5-turbo.csv", index=False)


###########################################################################################
# Cosine similarity helper function
###########################################################################################
def compute_cosine_similarity(record):
    """
    Given a dataframe record compute the cosine similarity between the 
    llm answer and the ground truth answer
    """
    answer_orig = record['answer_orig']
    answer_llm = record['answer_llm']
    
    v_llm = model.encode(answer_llm)
    v_orig = model.encode(answer_orig)
    
    return v_llm.dot(v_orig)


if __name__=="__main__":
    # Load the completed results for the 3 models provided by datatalks
    gpt4o_mini = load_llm_eval_dataframes("gpt4o-mini")
    gpt4o = load_llm_eval_dataframes("gpt4o")
    gpt35 = load_llm_eval_dataframes("gpt35")

    # Compute the cosine similarity
    cos_sim_gpt4o_mini = multithread_func(gpt4o_mini.to_dict(orient="records"),compute_cosine_similarity)
    cos_sim_gpt4o = multithread_func(gpt4o.to_dict(orient="records"),compute_cosine_similarity)
    cos_sim_gpt35 = multithread_func(gpt35.to_dict(orient="records"),compute_cosine_similarity)

    # Plot the results of the 3 models
    sns.histplot(cos_sim_gpt4o_mini, kde=True, bins=20, linewidth=0, stat="density", label='4o-mini')
    sns.histplot(cos_sim_gpt4o, kde=True, bins=20, linewidth=0, stat="density", label='4o')
    sns.histplot(cos_sim_gpt35, kde=True, bins=20, linewidth=0, stat="density", label='3.5-turbo')

    plt.title("RAG LLM performance")
    plt.xlabel("A->Q->A' Cosine Similarity")
    plt.legend()
    plt.savefig("OpenAI_model_comparison.png",dpi=200)
    plt.close()

    mean_cos_sim = {
        "gpt-4o-mini": np.around(np.mean(cos_sim_gpt4o_mini),3),
        "gpt-4o": np.around(np.mean(cos_sim_gpt4o),3),
        "gpt-3.5-turbo": np.around(np.mean(cos_sim_gpt35),3)
    }
    print("\n\nMean cosine similarity for the 3 models")
    print(json.dumps(mean_cos_sim,indent=2, default=str))