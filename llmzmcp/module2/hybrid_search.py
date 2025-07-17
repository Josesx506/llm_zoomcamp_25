from random import choice

from fastembed import TextEmbedding
from qdrant_client import models
from tqdm import tqdm

from llmzmcp.data import load_rag_documents
from llmzmcp.shared import qdclient as client

# Load the documents
documents_raw = load_rag_documents()

collection_name = "zoomcamp-hybrid"

# Check if the collection exists
existing_collections = client.get_collections().collections
existing_names = [c.name for c in existing_collections]

# Create the collection with specified vector parameters
if collection_name not in existing_names:
    client.create_collection(
        collection_name=collection_name,
        vectors_config={ # Named dense vector
            "jina-small": models.VectorParams(
                size=512, distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={ # Named sparse vector
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    )
else:
    print(f"Collection '{collection_name}' already exists. Skipping creation.")


# Upsert the entries into the collection i.e. create if it doesn't exist or update if it exists
resp = client.count(collection_name=collection_name, exact=True)
if resp.count == 0:
    print(f"Collection '{collection_name}' is empty. Proceeding with upsert.")

    points = []
    print("Seeding points....")
    for id,doc in tqdm(enumerate(documents_raw), total=len(documents_raw), position=0):
        # Manually specify a collection index
        point = models.PointStruct(
            id=id+1,
            vector={
                "jina-small": models.Document(
                    text=doc["text"], model="jinaai/jina-embeddings-v2-small-en",
                ),
                "bm25": models.Document(text=doc["text"], model="Qdrant/bm25"),
            },
            payload={
                "text": doc["text"],
                "section": doc["section"],
                "course": doc["course"],
            }
        )
        points.append(point)
    
    client.upsert(collection_name=collection_name,points=points)
else:
    print(f"Collection '{collection_name}' already has {resp.count} points. Skipping upsert.")


###########################################################################################
# Search the db collection using the hybrid embedding point vectors
###########################################################################################
def reranking_search(query: str, limit: int = 1) -> list[models.ScoredPoint]:
    results = client.query_points(
        collection_name=collection_name,
        # Prefetch excess results
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query,
                                      model="jinaai/jina-embeddings-v2-small-en",),
                using="jina-small", # vector name
                limit=(10 * limit), # ten times more results
            ),
        ],
        # Rerank step
        query=models.Document(text=query, model="Qdrant/bm25"),
        using="bm25", limit=limit, with_payload=True,
    )

    return results.points


def fusion_rrf_search(query: str, limit: int = 1) -> list[models.ScoredPoint]:
    results = client.query_points(
        collection_name=collection_name,
        # Prefetch results for both the dense and sparse algo
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="jinaai/jina-embeddings-v2-small-en",
                ),
                using="jina-small",
                limit=(5 * limit),
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="Qdrant/bm25",
                ),
                using="bm25",
                limit=(5 * limit),
            ),
        ],
        # Fusion query enables fusion on the prefetched results
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
    )

    return results.points


query = "Uploading to s3 fails with An error occurred (InvalidAccessKeyId) when calling the PutObject operation: "+\
    "The AWS Access Key Id you provided does not exist in our records."
print(f"\n\nQuery 1: {query}")
print("\nHybrid Rerank results\n")
print(reranking_search(query)[0].payload['text'])
print("\nHybrid Fusion results\n")
print(fusion_rrf_search(query)[0].payload['text'])