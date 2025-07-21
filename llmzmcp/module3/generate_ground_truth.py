import hashlib
import json
import pickle

import pandas as pd
from tqdm import tqdm

from llmzmcp.data import load_rag_documents
from llmzmcp.shared import oaiclient as client


def generate_document_id(doc):
    """
    Create a document id using the hash of the course, 
    question, and part of the response.
    """
    combined = f"{doc['course']}-{doc['question']}-{doc['text'][:10]}"
    hash_object = hashlib.md5(combined.encode())
    hash_hex = hash_object.hexdigest()
    document_id = hash_hex[:8]
    return document_id


prompt_template = """
You emulate a student who's taking our course.
Formulate 5 questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record. 

The record:

section: {section}
question: {question}
answer: {text}

Provide the output in parsable JSON without using code blocks:

["question1", "question2", ..., "question5"]
""".strip()


def generate_questions(doc):
    """
    Generate copycat questions for each question id
    """
    prompt = prompt_template.format(**doc)

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}]
    )

    json_response = response.choices[0].message.content
    return json_response


def convert_results_to_csv(api_results,docs_list):
    """
    Convert the api results of 5 additioal queries for each document 
    record into a csv
    """
    parsed_resulst = {}

    for doc_id, json_questions in api_results.items():
        parsed_resulst[doc_id] = json.loads(json_questions)

    doc_index = {d['id']: d for d in docs_list}

    # List of tuples to create dataframe
    final_results = []
    for doc_id, questions in parsed_resulst.items():
        course = doc_index[doc_id]['course']
        for q in questions:
            final_results.append((q, course, doc_id))
    
    df = pd.DataFrame(final_results, columns=['question', 'course', 'document'])
    df.to_csv('ground-truth-data.csv', index=False)
        

if __name__=="__main__":

    documents = load_rag_documents()
    # Create the doc ids using the hash function
    for doc in documents:
        doc['id'] = generate_document_id(doc)


    results = {}
    for doc in tqdm(documents[:2]): # Use only 2 docs to test the pipeline
        doc_id = doc['id']

        # Use local cache to avoid OpenAI rate limiting and duplicated api calls
        if doc_id in results:
            continue
        
        # Make api and append results.
        questions = generate_questions(doc)
        results[doc_id] = questions
    
    # Save the file as a pickled binary file
    # with open('results.bin', 'wb') as f_out:
    #     pickle.dump(results, f_out)

    # Convert to csv
    # convert_results_to_csv(results, documents)