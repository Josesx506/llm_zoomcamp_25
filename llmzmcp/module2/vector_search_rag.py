from qdrant_client import models
from tqdm import tqdm

from llmzmcp.data import load_rag_documents
from llmzmcp.module1.utils import build_prompt, llm
from llmzmcp.shared import qdclient as client

# Create a collection that generates embeddings from both the question and text
documents_raw = load_rag_documents()
EMBEDDING_DIMENSIONALITY = 512

model_handle = "jinaai/jina-embeddings-v2-small-en"
collection_name = "zoomcamp-faq"

# Check if the collection exists
existing_collections = client.get_collections().collections
existing_names = [c.name for c in existing_collections]

# Create the collection with specified vector parameters
if collection_name not in existing_names:
    print(f"Creating collection '{collection_name}'")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSIONALITY,  # Dimensionality of the vectors
            distance=models.Distance.COSINE  # Distance metric for similarity search
        )
    )
else:
    print(f"Collection '{collection_name}' already exists. Skipping creation.")

# Create a list of points 
points = []
for id,doc in tqdm(enumerate(documents_raw), total=len(documents_raw), position=0):
    # Manually specify a collection index
    text = doc['question'] + " " + doc['text']
    point = models.PointStruct(
        id=id+1,
        vector=models.Document(text=text, model=model_handle),
        payload=doc
    )
    points.append(point)

# Upsert the entries into the collection i.e. create if it doesn't exist or update if it exists
resp = client.count(collection_name=collection_name, exact=True)
if resp.count == 0:
    print(f"Collection '{collection_name}' is empty. Proceeding with upsert.")
    client.upsert(collection_name=collection_name,points=points)
else:
    print(f"Collection '{collection_name}' already has {resp.count} points. Skipping upsert.")

# Create the index
client.create_payload_index(
    collection_name=collection_name, field_name="course",
    field_schema="keyword" # exact matching on string metadata fields
)

def vector_search_w_filter(query, course="mlops-zoomcamp", limit=5):
    """Perform query with filter applied"""
    vector_points = client.query_points(
        collection_name=collection_name,
        query=models.Document(text=query, model=model_handle),
        query_filter=models.Filter( 
            must=[models.FieldCondition(key="course",
                    match=models.MatchValue(value=course))
            ]
        ),
        limit=limit, with_payload=True
    )

    results = [point.payload for point in vector_points.points]

    return results


def rag_vectorsearch(query, course="data-engineering-zoomcamp"):
    search_results = vector_search_w_filter(query, course=course)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


query = 'how do I setup postgres?'
answer = rag_vectorsearch(query)
print(f"Query: {query}")
print(answer,"\n\n")