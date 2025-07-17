from uuid import uuid4

from llmzmcp.data import load_rag_documents
from llmzmcp.shared import qdclient as client
from qdrant_client import models
from tqdm import tqdm

# Load the documents
documents_raw = load_rag_documents()

collection_name = "zoomcamp-sparse"
model_handle = "Qdrant/bm25"

# Check if the collection exists
existing_collections = client.get_collections().collections
existing_names = [c.name for c in existing_collections]

# Create the collection with sparse `bm25` statistical algo
if collection_name not in existing_names:
    client.create_collection(
        collection_name=collection_name,
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    )
else:
    print(f"Collection '{collection_name}' already exists. Skipping creation.")


# Create a list of points 
points = []

print("Seeding points....")
for id,doc in tqdm(enumerate(documents_raw), total=len(documents_raw), position=0):
    # Manually specify a collection index
    point = models.PointStruct(
            id=uuid4().hex, # Use uuid for index instead of int (optional)
            vector={"bm25": models.Document(text=doc["text"],model=model_handle)},
            payload={# metadata
                "text": doc["text"],
                "section": doc["section"],
                "course": doc["course"],
            }
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
# Search the db collection using the sparse embedding point vectors
###########################################################################################
def sparse_search(query: str, limit: int = 1) -> list[models.ScoredPoint]:
    "Return top result only from sparse search"
    results = client.query_points(
        collection_name=collection_name,
        query=models.Document(text=query,model=model_handle),
        using="bm25", limit=limit,
        with_payload=True,
    )

    return results.points


# Manually get results for three different words from the collection 
print(f"\n\nQuery 1: `polars`")
print(sparse_search("polars"))

print(f"\n\nQuery 2: `pandas`")
print("Results are:")
res = sparse_search("pandas")
print(">",res[0].payload['text'].strip())

print(f"\n\nQuery 3: `postgres`")
print("Results are:")
res = sparse_search("postgres")
print(">",res[0].payload['text'].strip())
# Scores returned by BM25 are not calculated with cosine similarity, but with BM25 formula. 
# They are not bounded to a specific range like cos. sim which uses (-1,1). They are virtually unbounded.
score = res[0].score
print(f"BM25 score: {score:.2f}")


print(f"\n\nQuery 4: `Even though the upload works using aws cli and boto3 in Jupyter notebook.`")
print("Results are:")
res = sparse_search("Even though the upload works using aws cli and boto3 in Jupyter notebook.")
print(">",res[0].payload['text'].strip())
