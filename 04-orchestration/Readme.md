### Orchestration
This involved building 
[Question-Answer Dataset](https://www.kaggle.com/datasets/rtatman/questionanswer-dataset/data)


Launch server `poetry run uvicorn app:app --reload`

Question - Answer Mini-wiki RAG


### Offline Eval - (Cosine similarity between ground truth and llm answers).
- The mean cosine similarity for gemma3:1b is 0.283
- The mean cosine similarity for gpt-4o-mini is 0.248

>[!Note]
> Ground truth answers are overly simplified for the dataset including one word answers like yes/no without including additional context that the LLM provides which decreases performance metrics.