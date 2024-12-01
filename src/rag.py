import os
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import tkinter as tk
from tkinter import scrolledtext, messagebox
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

# load_dotenv()
# API_KEY = os.getenv("GROQ_API_KEY")
API_KEY = os.environ.get("GROQ_API_KEY") # set as environment variable
if not API_KEY:
    raise ValueError("API key is missing! Please set API_KEY in your environment.")

# Initialize Hugging Face client
client = Groq(api_key=API_KEY)

# Load the vector index
INDEX_FOLDER = "data/indexes"
METADATA_FILE = os.path.join(INDEX_FOLDER, "metadata.pkl")

# Initialize embedding model
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Initialize Chroma vector store
chroma_db_path = os.path.join(INDEX_FOLDER, "chroma")
vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embed_model)

# Fetch both content and metadata from Chroma
all_data = vectorstore.get(include=["metadatas", "documents"])

# Combine content and metadata into a unified structure
documents = [
    {"content": doc, "metadata": meta}
    for doc, meta in zip(all_data["documents"], all_data["metadatas"])
]

# Tokenize corpus for BM25
tokenized_corpus = [word_tokenize(doc["content"].lower()) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

class BM25Retriever:
    def __init__(self, bm25, documents):
        self.bm25 = bm25
        self.documents = documents

    def get_relevant_documents(self, query, top_k=5):
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"page_content": self.documents[i]["content"], "metadata": self.documents[i].get("metadata", {})}
            for i in top_indices
        ]

bm25_retriever = BM25Retriever(bm25, documents)


def weighted_query_embedding(question, conversation_history, embed_model, weight_decay=0.5):
    """
    Combines weighted embeddings of the current question and conversation history.

    Parameters:
    - question (str): The user's current question.
    - conversation_history (list): List of conversation history entries with "role" and "content".
    - embed_model: Embedding model to generate vector representations.
    - weight_decay (float): Factor by which the weight of historical context decreases (default 0.5).

    Returns:
    - numpy.ndarray: A normalized, combined embedding vector.
    """
    # Generate embedding for the current question
    question_embedding = embed_model.embed_query(question)
    combined_embedding = np.array(question_embedding) * 1.0  # Current question has the highest weight

    # Add embeddings of the conversation history with decreasing weight
    weight = 1.0
    for entry in reversed(conversation_history):  # Start with the most recent entries
        if entry["role"] == "user":
            weight *= weight_decay  # Reduce weight for older entries
            history_embedding = embed_model.embed_query(entry["content"])
            combined_embedding += np.array(history_embedding) * weight

    # Normalize the combined embedding to unit length
    combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
    return combined_embedding


def hybrid_similarity_search(question, conversation_history, embed_model, bm25_retriever, top_k=5):
    """
    Performs a hybrid search combining vector-based similarity and BM25 lexical search.

    Parameters:
    - question (str): The user's current question.
    - conversation_history (list): List of conversation history entries with "role" and "content".
    - vectorstore: Vector-based retrieval system.
    - embed_model: Embedding model for semantic similarity.
    - bm25_retriever: BM25 retriever for lexical matching.
    - top_k (int): Number of top results to retrieve (default 5).

    Returns:
    - list: A list of dictionaries containing the top results, with text, source, and score.
    """
    # Generate a weighted embedding for the query
    query_embedding = weighted_query_embedding(question, conversation_history, embed_model)

    # 1. Vector-based similarity search
    vector_results = vectorstore.similarity_search_by_vector(query_embedding, k=top_k)

    # 2. BM25 lexical search
    bm25_results = bm25_retriever.get_relevant_documents(question)[:top_k]

    # Combine results from both methods
    combined_results = vector_results + bm25_results

    # Re-score combined results based on cosine similarity with the query embedding
    scored_results = []
    for res in combined_results:
        # Generate embedding for the result content
        # For vector results (document objects), access the 'page_content' attribute
        if hasattr(res, 'page_content'):
            page_content = res.page_content
        # For BM25 results (dictionaries), access the 'page_content' key
        else:
            page_content = res.get('page_content', '')  # Get the content from the dictionary, default to empty string if not found
        
        result_embedding = embed_model.embed_query(page_content)
        # Calculate cosine similarity between query and result embedding
        score = cosine_similarity([query_embedding], [result_embedding])[0][0]
        scored_results.append((res, score))

    # Sort results by score in descending order
    scored_results.sort(key=lambda x: x[1], reverse=True)

    # Format the results for output
    formatted_results = [
        {
            "text": res[0].page_content if hasattr(res[0], 'page_content') else res[0].get('page_content', ''),
            "source": res[0].metadata.get("source", "unknown") if hasattr(res[0], 'metadata') else res[0].get('metadata', {}).get("source", "unknown"),
            "score": res[1],
        }
        for res in scored_results[:top_k]
    ]

    # Debug: Verify retrieved results
    print(f"Retrieved Results: {formatted_results}")
    print(f"\nFinal Combined Results (Source, Score):")
    for res in scored_results:
        if hasattr(res[0], 'metadata'):
            source_str = res[0].metadata.get("source", "unknown")
            retrieval_type = "vector search"
        else:
            res[0].get('metadata', {}).get("source", "unknown")
            retrieval_type = "BM25 keyword search"
        print(f"Search type: {retrieval_type}, Source: {source_str}, Score: {res[1]}")

    return formatted_results


def generate_chat_response(question, context, conversation_history):
    """
    Generate a response based on the user's question and conversation history.
    """
    # Create a new conversation payload, emphasizing the context
    full_conversation = conversation_history + [
        {"role": "system", "content": f"Here is additional context for the question: {context}"},
        {"role": "user", "content": question},
    ]

    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=full_conversation,
    )

    # Aggregate streamed chunks
    response = chat_completion.choices[0].message.content

    # Update the conversation history with the user's question and assistant's response
    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": response})

    return response

class RAGChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAG Chat")
        self.geometry("900x600")
        self.configure(bg="#f5f5f5")  # background color

        self.conversation_history = [
            {"role": "system",
            "content": "You are a helpful teacher. Answer questions clearly and thoughtfully in the same language as the question."}
        ]

        self.create_widgets()


    def create_widgets(self):
        """Erstellt die Widgets der GUI."""
        # header
        header = tk.Label(
            self,
            text="RAG Chat - Fragebeantwortung",
            font=("Helvetica", 16, "bold"),
            bg="#f5f5f5",
            fg="#333",
        )
        header.pack(pady=10)

        # Chat-Fenster
        self.chat_box = tk.Text(
            self, wrap=tk.WORD, font=("Helvetica", 11), bg="#ffffff", state=tk.DISABLED
        )
        self.chat_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Eingabefeld
        input_frame = tk.Frame(self, bg="#f5f5f5")
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        self.input_entry = tk.Entry(input_frame, font=("Helvetica", 12))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_entry.bind("<Return>", self.on_enter_pressed)  # Enter-Taste binden

        send_button = tk.Button(
            input_frame,
            text="Senden",
            font=("Helvetica", 12),
            bg="#4caf50",
            fg="white",
            command=self.ask_question,
        )
        send_button.pack(side=tk.RIGHT)

    def ask_question(self):
        """Verarbeitet die Frage und zeigt die Antwort sowie Quellen an."""
        question = self.input_entry.get().strip()
        if not question:
            messagebox.showwarning("Leere Frage", "Bitte geben Sie eine Frage ein.")
            return

        # Eingabe leeren
        self.input_entry.delete(0, tk.END)

        # Retrieve context from the vector store
        results = hybrid_similarity_search(question, self.conversation_history, embed_model, bm25_retriever)
        context = " ".join([res["text"] for res in results])

        if context:
            self.append_to_chat("INFO", "Kontext aus der Vektordatenbank hinzugefügt.")
        else:
            self.append_to_chat("INFO", "Kein relevanter Kontext gefunden.")

        # Generate response using the user input, context, and chat history
        try:
            response = generate_chat_response(question, context, self.conversation_history)
        except Exception as e:
            messagebox.showerror(
                "Fehler bei der Antwortgenerierung",
                f"Ein Fehler ist aufgetreten: {str(e)}",
            )
            return

        # Antwort und Quellen anzeigen
        self.display_response(question, response, results)

    def on_enter_pressed(self, event):
        """Handler für die Enter-Taste."""
        self.ask_question()

    def display_response(self, question, response, sources):
        """Zeigt die Frage, die Antwort und die Quellen im Chatfenster an."""
        self.append_to_chat("USER", question)
        self.append_to_chat("AI", response)

        if sources:
            self.append_to_chat("SOURCES", "Quellen:")
            for res in sources:
                self.append_to_chat("SOURCES", f"  - {res['source']} (Score: {res['score']:.2f})")
        self.append_to_chat("SEPARATOR", "-" * 50)

    def append_to_chat(self, tag, text):
        """Fügt Text mit Formatierung in das Chatfenster ein."""
        self.chat_box.config(state=tk.NORMAL)
        if tag == "USER":
            self.chat_box.insert(tk.END, f"\nYou: {text}\n\n", ("bold",))
        elif tag == "AI":
            self.chat_box.insert(tk.END, f"AI: {text}\n\n")
        # elif tag == "INFO":
        #     self.chat_box.insert(tk.END, f"[INFO] {text}\n", ("italic",))
        elif tag == "SOURCES":
            self.chat_box.insert(tk.END, f"{text}\n")
        elif tag == "SEPARATOR":
            self.chat_box.insert(tk.END, f"{text}\n")
        self.chat_box.config(state=tk.DISABLED)  # Chatfeld nur lesbar machen
        self.chat_box.see(tk.END)  # Scrollen ans Ende


def main():
    """Startet die Hauptanwendung."""
    app = RAGChatApp()
    app.mainloop()


if __name__ == "__main__":
    main()
