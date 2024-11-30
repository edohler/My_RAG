import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import tkinter as tk
from tkinter import scrolledtext, messagebox

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
    print(f"Retrieved Results: {formatted_results}")

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
        results = query_vectorstore(question)
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


    # def create_widgets(self):
    #     # header
    #     header = tk.Label(
    #         self,
    #         text="RAG Chat - Fragebeantwortung",
    #         font=("Helvetica", 16, "bold"),
    #         bg="#f5f5f5",
    #         fg="#333",
    #     )
    #     header.pack(pady=10)

    #     # input
    #     input_frame = tk.Frame(self, bg="#f5f5f5")
    #     input_frame.pack(pady=5)

    #     question_label = tk.Label(
    #         input_frame, text="Frage eingeben:", font=("Helvetica", 12), bg="#f5f5f5"
    #     )
    #     question_label.pack(side=tk.LEFT, padx=5)

    #     self.question_entry = tk.Entry(input_frame, font=("Helvetica", 12), width=40)
    #     self.question_entry.pack(side=tk.LEFT, padx=5)

    #     ask_button = tk.Button(
    #         input_frame,
    #         text="Frage stellen",
    #         font=("Helvetica", 12),
    #         bg="#4caf50",
    #         fg="white",
    #         command=self.ask_question,
    #     )
    #     ask_button.pack(side=tk.LEFT, padx=5)

    #     # output textbox
    #     self.answer_box = scrolledtext.ScrolledText(
    #         self, width=70, height=15, wrap=tk.WORD, font=("Helvetica", 11)
    #     )
    #     self.answer_box.pack(pady=10, padx=10)
    #     self.answer_box.config(state=tk.DISABLED)

    # def ask_question(self):
    #     question = self.question_entry.get().strip()
    #     if not question:
    #         messagebox.showwarning("Leere Frage", "Bitte geben Sie eine Frage ein.")
    #         return

    #     # Retrieve context from the vector store
    #     results = query_vectorstore(question)
    #     context = " ".join([res["text"] for res in results])

    #     if context:
    #         self.append_to_answer_box("INFO", "Kontext aus der Vektordatenbank hinzugefügt.")
    #     else:
    #         self.append_to_answer_box("INFO", "Kein relevanter Kontext gefunden.")

    #     # Generate response using the user input, context, and chat history
    #     try:
    #         response = generate_chat_response(question, context, self.conversation_history)
    #     except Exception as e:
    #         messagebox.showerror(
    #             "Fehler bei der Antwortgenerierung",
    #             f"Ein Fehler ist aufgetreten: {str(e)}",
    #         )
    #         return

    #     # Antwort und Quellen anzeigen
    #     self.display_response(question, response, results)

    # def display_response(self, question, response, sources):
    #     """Zeigt die Frage, die Antwort und die Quellen in der Ausgabe-Textbox an."""
    #     self.append_to_answer_box("USER", f"Frage: {question}")
    #     self.append_to_answer_box("AI", f"Antwort: {response}")

    #     if sources:
    #         self.append_to_answer_box("SOURCES", "Quellen:")
    #         for res in sources:
    #             self.append_to_answer_box("SOURCES", f"  - {res['source']} (Score: {res['score']:.2f})")
    #     self.append_to_answer_box("SEPARATOR", "-" * 50)

    # def append_to_answer_box(self, tag, text):
    #     """Fügt Text mit Formatierung in die Antwort-Textbox ein."""
    #     self.answer_box.config(state=tk.NORMAL)
    #     if tag == "USER":
    #         self.answer_box.insert(tk.END, f"\nYou: {text}\n", ("bold",))
    #     elif tag == "AI":
    #         self.answer_box.insert(tk.END, f"AI Chatbox: {text}\n\n")
    #     elif tag == "INFO":
    #         self.answer_box.insert(tk.END, f"[INFO] {text}\n", ("italic",))
    #     elif tag == "SOURCES":
    #         self.answer_box.insert(tk.END, f"{text}\n")
    #     elif tag == "SEPARATOR":
    #         self.answer_box.insert(tk.END, f"{text}\n")
    #     self.answer_box.config(state=tk.DISABLED)  # Antwortfeld nur lesbar machen
    #     self.answer_box.see(tk.END)  # Scrollen ans Ende


def main():
    """Startet die Hauptanwendung."""
    app = RAGChatApp()
    app.mainloop()


if __name__ == "__main__":
    main()
