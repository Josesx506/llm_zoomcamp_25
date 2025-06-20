### Learn how search engines work
Given a dataframe / db of results with multiple fields, find a good way to search / match a query to the closest results
1. Generate a vectorial representation of all the words (bag of words) in the db with scikit-learn
2. Weight the words within each document to balance out identification of uncommon words
    - Compute cosine similarity score between query vector and db vector. 
        - You can sum scores across multiple columns, and even weight specific columns as desired to improve performance.
    - Rank results by similarity score to get the top n search results
3. Create a compact class for the search engine
4. Use sklearn to extract embeddings (smaller representation of the vector matrix) to optimize search
    - Compute scores using the condensed embedding matrices. You can also calculate across multiple fields/use weighted results.
    - Rank by similarity score and return search results
5. Use BeRT or other transformer libraries to extract embeddings.