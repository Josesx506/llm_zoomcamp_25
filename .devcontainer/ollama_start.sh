#!/bin/bash

# Start the Ollama server in the background
ollama serve &

# Wait for the server to be ready
until curl -s http://localhost:11434 > /dev/null; do
  echo "Waiting for Ollama server..."
  sleep 1
done

# Pull the model
ollama pull gemma3:1b

# Keep the server running in the foreground
wait
