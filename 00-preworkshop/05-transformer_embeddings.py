from pprint import pprint

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # Set the model to evaluation mode if not training

# Each list item is a document
texts = [
    "Yes, we will keep all the materials after the course finishes.",
    "You can follow the course at your own pace after it finishes"
]
encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model(**encoded_input)
    hidden_states = outputs.last_hidden_state

# Take the avg/sum of the hidden states across all columns to get the document embeddings
sentence_embeddings = hidden_states.mean(dim=1)

X_emb = sentence_embeddings.detach().numpy()


################## Generate Embeddings for the entire dataframe
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


def make_batches(seq, n):
    """
    Extract batches for generating embeddings
    """
    result = []
    for i in range(0, len(seq), n):
        batch = seq[i:i+n]
        result.append(batch)
    return result


def compute_embeddings(texts, batch_size=8):
    text_batches = make_batches(texts, 8)
    
    all_embeddings = []
    
    for batch in tqdm(text_batches):
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
    
        with torch.no_grad():
            outputs = model(**encoded_input)
            hidden_states = outputs.last_hidden_state
            
            batch_embeddings = hidden_states.mean(dim=1)
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings_np)
    
    final_embeddings = np.vstack(all_embeddings)
    return final_embeddings

dataset_embeddings = compute_embeddings(df['text'].tolist())


# Compute cosine similarity using BeRT embeddings
cos_sim = cosine_similarity(dataset_embeddings, X_emb)

query1_rank = np.argsort(cos_sim[:,0])
query1_idxs = query1_rank[-3:]
query2_rank = np.argsort(cos_sim[:,1])
query2_idxs = query2_rank[-3:]

print(f"\nQuery 1: `{texts[0]}`\n")
pprint(df.iloc[query1_idxs].text.to_dict(), indent=2, width=120)

print(f"\nQuery 2: `{texts[1]}`\n")
pprint(df.iloc[query2_idxs].text.to_dict(), indent=2, width=120)