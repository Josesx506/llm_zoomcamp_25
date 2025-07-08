# llm_zoomcamp_25
Datatalks llmops zoomcamp for learning RAG

Create an openai api key at https://platform.openai.com/api-keys. Save a key in an .env variable as OPENAI_API_KEY.

- Ensure you have **docker desktop** running and you can run the repo as a devcontainer locally or on github codespaces.
    - Whenever you make changes to the `.devcontainer` files, rebuild the container by pressing `Cmd + Shift + P` and select ***Dev Containers: Rebuild and Reopen in Container*** from the list of options to trigger the changes. 
    - It's configured with postgres, redis, and elastic search by default.
    - Docker is installed by default for codespaces

- Vector search is a type of similarity search that converts, text, images, or any file formats into a vector. Search results are obtained by comparing and ranking similarity between vectors.