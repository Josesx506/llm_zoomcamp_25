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