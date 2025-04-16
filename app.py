from flask import Flask, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

app = Flask(__name__)

# Ollama Mistral конфигурациясы
llm = Ollama(model="mistral:7b-instruct-q4_0", base_url="http://ollama:11434", request_timeout=120.0)
Settings.llm = llm
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# Құжаттарды жүктеу
reader = SimpleDirectoryReader("/app/data")
documents = reader.load_data()
index = VectorStoreIndex.from_documents(documents)
chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Сұрақ жоқ"}), 400
    response = chat_engine.chat(f"Тек қазақ тілінде жауап бер: {question}")
    return jsonify({"response": response.response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
