from flask import Flask, request, jsonify
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import gdown
import tempfile
import logging

app = Flask(__name__)

# Логтауды орнату
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama Mistral конфигурациясы
llm = Ollama(model="mistral:7b-instruct-q4_0", base_url="http://ollama:11434", request_timeout=120.0)
Settings.llm = llm
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# Google Drive-тан файлдарды жүктеу
file_urls = [
 "https://drive.google.com/uc?id=1s88Xt5QZKNn5sxIBS-qGYTAmUWuyLNc-",
 "https://drive.google.com/uc?id=164gru04ThIRz8S31D5FfVhXtNmuwu8hX",
 "https://drive.google.com/uc?id=16OFzZn8AZRTJrzTf-9UoPy3hAOvR8935",
 "https://drive.google.com/uc?id=18ZcZSDjGeGev80mjs4A1Jmsqo2JyjpEB",
 "https://drive.google.com/uc?id=1vTbmYP4bdfuZb1MqVgpkCQXopgTTe4O5"
]
documents = []
with tempfile.TemporaryDirectory() as temp_dir:
 for idx, url in enumerate(file_urls):
     try:
         output = os.path.join(temp_dir, f"file_{idx}.txt")
         logger.info(f"Downloading {url} to {output}")
         gdown.download(url, output, quiet=False)
         with open(output, 'r', encoding='utf-8') as f:
             text = f.read()
             documents.append(Document(text=text, metadata={"source": f"file_{idx}.txt"}))
     except Exception as e:
         logger.error(f"Failed to process {url}: {str(e)}")

# Индекс құру
if documents:
 index = VectorStoreIndex.from_documents(documents)
 chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)
else:
 logger.error("No documents loaded, cannot create index")
 raise ValueError("No documents available for indexing")

@app.route("/chat", methods=["POST"])
def chat():
 data = request.json
 question = data.get("question", "")
 if not question:
     return jsonify({"error": "Сұрақ жоқ"}), 400
 try:
     response = chat_engine.chat(f"Тек қазақ тілінде жауап бер: {question}")
     return jsonify({"response": response.response})
 except Exception as e:
     logger.error(f"Chat error: {str(e)}")
     return jsonify({"error": "Серверде қате пайда болды"}), 500

if __name__ == "__main__":
 port = int(os.getenv("PORT", 5000))
 app.run(host="0.0.0.0", port=port)
