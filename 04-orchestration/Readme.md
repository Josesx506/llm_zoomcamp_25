### Problem description
This application provides RAG context for a mini-wikipedia problem. It uses a 
[Question-Answer Dataset](https://www.kaggle.com/datasets/rtatman/questionanswer-dataset/data) 
that was generated using cleaned text from Wikipedia articles. The dataset for different years 
(2008 - 2010) was downloaded as separate text files and cleaned and merged into a single csv.
Preprocessing steps included removing duplicates and simple formatting changes. Check 
[combine_data.py](./combine_data.py) to generate the main dataset. In production I used  `gpt-4o-mini` and in development, I used `gemma3:1b` with an ollama server running in a 
devcontainer with docker.

### Retrieval flow
Data is ingested into a minsearch in-memory index using a python script. For each query, the top 
5 search results from the knowledge base are provided as context to an LLM. The LLM reviews the 
index and provides a response to the query.

### Retrieval evaluation
The retrieval performance is evaluated using the `hit rate` and `MRR`. For each Q-A pair, we 
compare the document ids (`ArticleFile`) in the retrieval response. You can run the workflow in 
the [evaluate_search.py](./backend/evaluate_search.py) script with `poetry run python evaluate_search.py `


Launch server `poetry run uvicorn app:app --reload`

Question - Answer Mini-wiki RAG


### Offline Eval - (Cosine similarity between ground truth and llm answers).
- The mean cosine similarity for gemma3:1b is 0.283
- The mean cosine similarity for gpt-4o-mini is 0.248

>[!Note]
> Ground truth answers are overly simplified for the dataset including one word answers like yes/no without including additional context that the LLM provides which decreases performance metrics.