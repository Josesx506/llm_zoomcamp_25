import os
from pathlib import Path
import json
from llmzmcp.utils import timed_lru_cache

def data_dir():
    directory = Path(os.path.dirname(os.path.realpath(__file__)))
    return directory

@timed_lru_cache(1800)
def load_rag_documents():
    with open(f'{data_dir()}/documents.json', 'rt') as f_in:
        docs_raw = json.load(f_in)
    
    documents = []
    for course_dict in docs_raw:
        for doc in course_dict['documents']:
            doc['course'] = course_dict['course']
            documents.append(doc)
    
    return documents


def load_llm_documents():
    with open(f'{data_dir()}/documents-llm.json', 'rt') as f_in:
        docs_raw = json.load(f_in)
    
    return docs_raw