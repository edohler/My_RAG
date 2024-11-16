import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("API key is missing! Please set HF_API_KEY in your environment.")

# Initialize Hugging Face client
client = InferenceClient(api_key=HF_API_KEY)

# Load the vector index
INDEX_FOLDER = "data/indexes"
FAISS_INDEX_FILE = os.path.join(INDEX_FOLDER, "faiss_index")
METADATA_FILE = os.path.join(INDEX_FOLDER, "metadata.pkl")

# Initialize embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def query_faiss_index(question, top_k=3):
    # Load FAISS index
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    
    # Generate query embedding
    question_embedding = embedding_model.encode(question)

    # Search FAISS index
    distances, indices = faiss_index.search(np.array([question_embedding]), top_k)

    # Retrieve relevant chunks and sources
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx == -1:  # No result
            continue
        result = metadata[idx]
        result["distance"] = distance
        results.append(result)

    return results

def generate_answer_with_sources(question, context):
    messages = [
        {"role": "user",
         "content": f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}"}
    ]

    # Call LLaMA API
    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        max_tokens=500,
        stream=True
    )

    # Aggregate streamed chunks
    answer = "".join(chunk.choices[0].delta.content for chunk in stream)
    return answer


if __name__ == "__main__":
    question = input("Enter your question: ")

    # retrieve context from FAISS
    results = query_faiss_index(question)
    context = " ".join([res["text"] for res in results])

    # generate response using Llama
    answer = generate_answer_with_sources(question, context)

    print("\nAnswer: ", answer)
    print("\Sources: ")
    for res in results:
        print(f"  - Source: {res['source']}")
        print(f"    Text: {res['text']}")


