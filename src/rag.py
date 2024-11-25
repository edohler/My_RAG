import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

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
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Initialize Chroma vector store
chroma_db_path = os.path.join(INDEX_FOLDER, "chroma")
vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embed_model)

def query_vectorstore(question, top_k=3):
    # Search Chroma for the most relevant chunks
    results = vectorstore.similarity_search_with_score(question, k=top_k)
    
    # Extract relevant chunks and metadata
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": score
        })
    
    # Debug: Verify retrieved results
    # print(f"Retrieved Results: {formatted_results}")

    return formatted_results

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
    # print(f"Retrieved Metadata: {results}")

    return results

def generate_chat_response(question, conversation_history):
    """
    Generate a response based on the user's question and conversation history.
    """
    # Add the user's question to the conversation history
    conversation_history.append({"role": "user", "content": question})

    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=conversation_history,
    )

    # Aggregate streamed chunks
    response = chat_completion.choices[0].message.content

    # Add the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response})
    
    return response


if __name__ == "__main__":
    conversation_history = [
        {"role": "system",
         "content": "You are a helpful teacher. Answer questions clearly and thoughtfully."}
    ]
    
    print("Welcome to the RAG-Chat! Type 'exit' to end the conversation.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye! See you soon my dear!")
            break

        # retrieve context from FAISS or Chroma vector store
        # results = query_faiss_index(question)
        results = query_vectorstore(user_input)
        context = " ".join([res["text"] for res in results])
        
        # Optionally add context into the conversation history for deeper responses
        if context:
            conversation_history.append({"role": "system", "content": f"Context: {context}"})
        
        # Generate response
        response = generate_chat_response(user_input, conversation_history)

        # Display the response
        print("\nAI Chatbox: ", response)

        # Optionally, show sources
        print("\nSources: ")
        for res in results:
            print(f"  - Source: {res['source']}")
        print("\n")
