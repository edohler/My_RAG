import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq
from dotenv import load_dotenv

# load_dotenv()
# API_KEY = os.getenv("GROQ_API_KEY")
API_KEY = os.environ.get("GROQ_API_KEY") # set as environment variable
if not API_KEY:
    raise ValueError("API key is missing! Please set API_KEY in your environment.")

# Initialize Hugging Face client
client = Groq(api_key=API_KEY)

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
    
    # Debug: Verify retrieved results
    print(f"Retrieved Metadata: {results}")

    return results

def generate_answer_with_sources(question, context):
    messages = [
        {"role": "system",
         "content": "you are helpful teacher."},

        {"role": "user",
         "content": f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}"}
    ]

    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
    )

    # Aggregate streamed chunks
    answer = chat_completion.choices[0].message.content
    return answer


if __name__ == "__main__":
    question = input("Enter your question: ")

    # retrieve context from FAISS
    results = query_faiss_index(question)
    context = " ".join([res["text"] for res in results])
    print(f"Constructed Context: {context}")

    # generate response using Llama
    answer = generate_answer_with_sources(question, context)

    print("\nAnswer: ", answer)
    print("\nSources: ")
    for res in results:
        print(f"  - Source: {res['source']}")
        print(f"    Text: {res['text']}")


