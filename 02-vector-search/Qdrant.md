### Intro
Qdrant is fully open-source, which means you can run it in multiple ways depending on your needs.You can self-host it on your own infrastructure, deploy it on Kubernetes, or run it in managed Cloud. I'm running it as a service in my dev container.

- 6333 – REST API port
- 6334 – gRPC API port

To help you explore your data visually, Qdrant provides a built-in **Web UI**, available in both Qdrant Cloud and local instances.
You can use it to inspect collections, check system health, and even run simple queries.

When you're running Qdrant in Docker, the Web UI is available at http://localhost:6333/dashboard

### Vector Search Processing Steps
1. Dataset needs to be first ***chunked*** into a series of documents with fields that can be used for semantic search (finding related items), and associated metadata.

    - which fields could be used for **semantic search** ;
    - which fields should be stored as **metadata**, e.g. useable for filtering conditions; 

    In the zoomcamp documents dataset, the *question* and *text* fields can be used for **search**, while the *course* and *section* fields can be used for **metadata**.

2. ***Generate embeddings from the documents*** -  In the elastic search in Module 1, we used BeRT to generate embeddings. Qdrant has a simpler and faster quantized sentence transformer models built as part of the `FastEmbed` package.
    - Determine an embedding size you want. Larger embeddings take up more storage.
    - Select a model for generating the embeddings out of available options in fastembed
        ```python
        # List the quantized models in the FastEmbed package
        options = TextEmbedding.list_supported_models()

        # Filter models that can generate 512 embeddings
        EMBEDDING_DIMENSIONALITY = 512
        options512 = [mod for mod in options if mod["dim"] == EMBEDDING_DIMENSIONALITY]
        ```

3. Create a ***collection***. A collection is analogous to a table. It has the generated embeddings, and their associated metadata. Each document becomes a *point* in a collection. 
    - A point will contain the vector coordinates of the document within the collection, and the associated metadata that can be used to filter the search.

    - When creating a qdrant [collection](https://qdrant.tech/documentation/concepts/collections/), we need to specify:
        - Name: A unique identifier for the collection.
        - Vector Configuration:
            - Size: The dimensionality of the vectors.
            - Distance Metric: The method used to measure similarity between vectors.
    - There are additional parameters you can explore in qdrant's [documentation](https://qdrant.tech/documentation/concepts/collections/#create-a-collection). Moreover, you can configure other vector types in Qdrant beyond typical dense embeddings (f.e., for hybrid search). However, for this example, the simplest default configuration is sufficient.

4. Populate / Seed the collection with document embeddings. You can view the vector representation with the Qdrant UI. Include this config to modify the color legend
    ```json
    {
    "limit": 948,
    "color_by": {
        "payload": "course"
        }
    }
    ```

5. To improve search efficiency, you can create an index on a metadata column
    ```python
    client.create_payload_index(
        collection_name=collection_name, field_name="course",
        field_schema="keyword" # exact matching on string metadata fields
    )
    ```

### Vector Search RAG
Once the search pipeline is setup, the search results can be passed as background context to an LLM for improving suggestions with RAG.


### Sparse Search
Qdrant allows us to generate sparse embeddings (mostly zeros) using statistical models like `bm25`. Unlike the dense embeddings used in vector search, sparse embeddings are useful for finding single words from documents e.g "pandas", "s3" etc. <br>
When results don't match, search using sparse embeddings can return an empty list of results unlike vector embeddings. When the input query consists of multiple words like a sentence, sparse search isn't very performant. To optimize the blend between sparse and dense embeddings, we can implement hybrid search. <br>
Qdrant's `.query_points` method allows building multi-step search pipelines which can incorporate various methods into a single call. For example, we can retrieve some candidates with dense vector search, and then rerank them with sparse search, or use a fast method for initial retrieval and precise, but slow, reranking.

```ascii
┌─────────────┐           ┌─────────────┐
│             │           │             │
│  Retrieval  │ ────────► │  Reranking  │
│             │           │             │
└─────────────┘           └─────────────┘
```


### Hybrid Search
One approach for hybrid search is _reranking_. You can use a dense embedding model to retrieve excess search results, then use a sparse embedding 
algorithm to rerank the returned results to get your final answer. e.g. Use dense embedding model to prefetch 10 results, use sparse model to rank and 
return the top result. <br>
Another approach to performing hybrid search is to do _fusion_. In fusion, both the dense, and sparse models, return equal number of results. e.g. 5 
documents for both algorithms. Then the scores are merged. There are various ways of how to achieve that, but `Reciprocal Rank Fusion` is the most 
popular technique. It is based on the rankings of the documents in each methods used, and these rankings are used to calculate the final scores. Qdrant 
has built in capabilties to implement this and raw equations can be found online. An example is shown below 

| Document | Dense ranking | Sparse ranking | RRF score | Final ranking |
| --- | --- | --- | --- | --- |
| D1 | **1** | 5 | 0.0318 | 2 |
| D2 | 2 | 4 | 0.0317 | 3 |
| D3 | 3 | 2 | 0.0320 | **1** |
| D4 | 4 | 3 | 0.0315 | 5 |
| D5 | 5 | **1** | 0.0318 | 2 |

Documents _D1_ and _D5_ are ranked inverse to each other by both algorithms. When the scores are fused using RRF, document _D3_ emerges as the most 
likely search result. <br>

Reranking is a broader term related to Hybrid Search. Fusion is one of the ways to rerank the results of multiple methods, but you can also apply a 
slower method that won't be effective enough to search over all the documents. But there is more to it. Business rules are often important for 
retrieval, as you prefer to show documents coming from the most recent news, for instance. Dense and sparse vector search methods might not be enough 
in some cases, but both are fast enough to be used as initial retrievers. Plenty of more accurate yet slower methods exist, such as cross-encoders or 
[multivector representations](https://qdrant.tech/documentation/advanced-tutorials/using-multivector-representations/). These topics are definitely 
more advanced, and we won't cover them right now. However, it's good to mention them so you are aware they exist.