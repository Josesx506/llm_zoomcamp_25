### Intro
Evaluating LLMs requires ground truth data. In evaluation, you comparing the performance metrics of the search algo. or the input prompt.
E.g. 
- If you have a query that is related to multiple documents (D1,D5,D8), and you have a 25 prompts you want to evaluate. 
- For each search algo., you evaluate how many times the related documents are selected after you give the prompt. 
- If each search algo. returns 10 results. A algorithm that includes all 3 documents is better than an algo. that returns one / two.
Typical metrics are Precision k, Recall k etc. <br>

Because creating the labels for a good evaluation scenario is complex for the zoomcamp, we used a roundabout route to use ChatGPT to generate 
alternate questions for each document in the zoomcamp dataset. i.e. for each record, generate 5 queries that are related to the same result. 
Then we can evaluate how each search algo./prompt assesses which prompt is relevant for the target queries. Generating the copycat questions 
for all the documents cost ~$4 of chatgpt tokens, so I just used the saved file.