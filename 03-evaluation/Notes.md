### Intro
Evaluating LLMs requires ground truth data. In evaluation, you comparing the performance metrics of the search algo. or the input prompt.
E.g. 
- If you have a query that is related to multiple documents (D1,D5,D8), and you have a 25 prompts you want to evaluate. 
- For each search algo., you evaluate how many times the related documents are selected after you give the prompt. 
- If each search algo. returns 10 results. A algorithm that includes all 3 documents is better than an algo. that returns one / two.
Typical metrics are Precision k, Recall k, Hit Rate (Recall), Mean Reciprocal Rank (MRR) etc. <br>

Because creating the labels for a good evaluation scenario is complex for the zoomcamp, we used a roundabout route to use ChatGPT to generate 
alternate questions for each document in the zoomcamp dataset. i.e. for each record, generate 5 queries that are related to the same result. 
Then we can evaluate how each search algo./prompt assesses which prompt is relevant for the target queries. Generating the copycat questions 
for all the documents cost ~$4 of chatgpt tokens, so I just used the saved file.

### Search Retrieval Evaluation
In search evaluation, we pass the 5 copycat queries to the search algo., and basically confirm whether the algorithm returned the ground truth 
document id. We can alter the search algorithm params in MinSearch or ElasticSearch by increasing weights for specific fields, or changing 
which fields should be eliminated or included in the search. As we change each metric, we'll typically want to track our changes with an experiment 
tracking tool like MLflow, and track how our metrics alternate for each parameter change.

### Prompt Evaluation
Prompt evaluation can be computed either _online_ or _offline_.
- **Online** - We evaluate metries like user feedback, A/B tests, experiments etc. 
- **Offline** - This is usually done in dev. or before the llm is deployed to production. We evaluate metrics like cosine similarity or we 
    can use the *`LLM as a judge`* etc. Given a query that we know the original answer for, we can use an LLM to generate a predicted answer 
    and compare the cosine similarity between original answer vs. predicted.

<br>

LLM as a judge is a special case of evaluating model responses using discrete groups like *relevant*, *non-relevant* etc. to evaluate rag 
performance. It can be performed for online or offline evaluation and can make sense if an advanced / superior model is used to evaluate 
a smaller / less performant model (almost like a teacher-student scenario). For this to work, you provide a 
- question, 
- an llm response, and 
- the human response (optional only for offline eval., unavailable for online eval.)
The bigger model then ranks the llm response into one of the 3 categories. If the judge llm is subpar to the llm used to generate a response, 
I think the evaluation results would be garbage. It can be useful in a scenario like evaluating `gpt-3.5-turbo` using `gpt-4o` but it sounds 
and feels weird to evaluate performance using this approach due to risk of hallucination.

>[!Important]
>The performance of the search retrieval algorithm affects the quality of the context in a RAG pipeline too.