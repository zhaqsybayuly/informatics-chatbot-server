FROM ollama/ollama RUN ollama pull mistral:7b-instruct-q4_0 EXPOSE 11434 CMD \["ollama", "serve"\]
