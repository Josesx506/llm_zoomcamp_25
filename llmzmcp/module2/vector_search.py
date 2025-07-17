from random import choice

from fastembed import TextEmbedding
from qdrant_client import models
from tqdm import tqdm

from llmzmcp.data import load_rag_documents
from llmzmcp.shared import qdclient as client

# Load the documents
documents_raw = load_rag_documents()

# List the quantized models in the FastEmbed package
options = TextEmbedding.list_supported_models()

# Filter models that can generate 512 embeddings
EMBEDDING_DIMENSIONALITY = 512
options512 = [mod for mod in options if mod["dim"] == EMBEDDING_DIMENSIONALITY]

# You need to know the model name and description e.g.
# Whether it's valid for a specific language, whether it's unimodal or multi-modal, etc. 
# For this example, we use a unimodal model for english text only. It uses cosine similarity
model_handle = "jinaai/jina-embeddings-v2-small-en"


# Define the collection name. Analogous to a db table
collection_name = "zoomcamp-rag"

# Check if the collection exists
existing_collections = client.get_collections().collections
existing_names = [c.name for c in existing_collections]

# Create the collection with specified vector parameters
if collection_name not in existing_names:
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

print("Seeding points....")
for id,doc in tqdm(enumerate(documents_raw), total=len(documents_raw), position=0):
    # Manually specify a collection index
    point = models.PointStruct(
        id=id+1,
        #embed text locally with "jinaai/jina-embeddings-v2-small-en" from FastEmbed
        vector=models.Document(text=doc['text'], model=model_handle),
        payload={
            "text": doc['text'],
            "section": doc['section'],
            "course": doc['course']
        } #save all needed metadata fields
    )
    points.append(point)

# Upsert the entries into the collection i.e. create if it doesn't exist or update if it exists
resp = client.count(collection_name=collection_name, exact=True)
if resp.count == 0:
    print(f"Collection '{collection_name}' is empty. Proceeding with upsert.")
    client.upsert(collection_name=collection_name,points=points)
else:
    print(f"Collection '{collection_name}' already has {resp.count} points. Skipping upsert.")


###########################################################################################
# Search the db collection using the dense embedding point vectors
###########################################################################################
def similarity_search(query, limit=1):
    """
    Query the vector embeddings to find the closest one without filtering by metadata
    """
    #embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
    results = client.query_points(
        collection_name=collection_name,
        query=models.Document(text=query,model=model_handle ),
        limit=limit,       # top closest matches
        with_payload=True  # to get metadata in the results
    )

    return results

# Randomly select a question from the original documents and query the collection for it
rand = choice(documents_raw)
rand_q = rand['question']
rand_a = rand['text']
result = similarity_search(rand_q)
result_txt = result.points[0].payload['text']

print(f"\n\nQuery: {rand_q}")
if rand_a == result_txt:
    print("  ->\tWe found the original question")
else:
    print("  ->\tOriginal question could not be retrieved")


# Manual question
print(f"\n\nQuery 2 no filtering")
print(similarity_search("What if I submit homeworks late?").points[0].payload['text'].strip())


###########################################################################################
# Index the collection for more efficient search results
###########################################################################################
client.create_payload_index(
    collection_name=collection_name, field_name="course",
    field_schema="keyword" # exact matching on string metadata fields
)

def search_in_course(query, course="mlops-zoomcamp", limit=1):
    """Perform query with filter applied"""
    results = client.query_points(
        collection_name=collection_name,
        query=models.Document(text=query, model=model_handle),
        # filter results by course name
        query_filter=models.Filter( 
            must=[models.FieldCondition(key="course",
                    match=models.MatchValue(value=course))
            ]
        ),
        limit=limit, with_payload=True
    )

    return results


# Manual question
print(f"\n\nQuery 2 with filtered results")
print(search_in_course("What if I submit homeworks late?", "mlops-zoomcamp").points[0].payload['text'])