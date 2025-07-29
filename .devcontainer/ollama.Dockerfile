FROM ollama/ollama

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY .devcontainer/ollama_start.sh /ollama_start.sh
RUN chmod +x /ollama_start.sh

ENTRYPOINT ["/ollama_start.sh"]
