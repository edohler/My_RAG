import tkinter as tk
from tkinter import messagebox
from retrievers import hybrid_similarity_search
from retrievers import semantic_search, refine_question_with_llm
from langdetect import detect


class RAGChatApp(tk.Tk):
    def __init__(self, embed_model, vectorstore, llm_client, bm25_retriever=None):
        super().__init__()
        self.title("RAG Chat")
        self.geometry("900x600")
        self.conversation_history = [
            {"role": "system",
             "content": "You are a helpful teacher. Answer questions clearly and thoughtfully in the same language as the question."}
        ]
        self.embed_model = embed_model
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.llm_client = llm_client

        # Determine retrieval mode
        self.use_hybrid = bm25_retriever is not None
        
        self.create_widgets()

    def create_widgets(self):
        header = tk.Label(self, text="RAG Chat", font=("Helvetica", 16, "bold"))
        header.pack(pady=10)

        self.chat_box = tk.Text(self, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        input_frame = tk.Frame(self)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        self.input_entry = tk.Entry(input_frame, font=("Helvetica", 12))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_entry.bind("<Return>", self.on_enter_pressed)

        send_button = tk.Button(input_frame, text="Send", command=self.ask_question)
        send_button.pack(side=tk.RIGHT)

    def ask_question(self):
        """
        Handles the question input by the user, retrieves context, generates a response,
        and updates the conversation history.
        """
        question = self.input_entry.get().strip()
        if not question:
            messagebox.showwarning("Empty Question", "Please enter a question.")
            return

        self.input_entry.delete(0, tk.END)
        self.append_to_chat("USER", question)

        if self.use_hybrid:
            # Retrieve relevant documents using hybrid search
            results = hybrid_similarity_search(
                question, self.conversation_history, self.embed_model, self.vectorstore, self.bm25_retriever
            )
        else:
            # Refine the question using the LLM
            refined_question = refine_question_with_llm(question, self.conversation_history, self.llm_client)

            # Perform semantic search using the refined question
            results = semantic_search(refined_question, self.vectorstore, self.embed_model)

        context = " ".join([res["text"] for res in results])

        # Generate a response using the context
        response = self.generate_response(question, context)

        # Append response to conversation history and chat box
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": response})
        self.append_to_chat("ASSISTANT", response)

        self.append_to_chat("SOURCES", "Sources:")
        for res in results:
            self.append_to_chat("SOURCES", f"  - {res['source']} (Score: {res['score']:.2f})")
        self.append_to_chat("SEPARATOR", "-" * 50)

    def generate_response(self, question, context):
        """
        :param question: The user question.
        :param context: Retrieved context from the hybrid search.
        :return: Generated response as a string.
        """
        # Detect the language of the question
        question_language = detect(question)

        # System prompt to instruct the LLM
        system_prompt = (
            f"You are an assistant that answers questions based on additional context. "
            f"Respond in the same language as the user's question, which is {question_language}. "
            f"Please answer the question and state whether your response is based on the provided context or your own knowledge. "
            f"Here is the additional context, to help you answering the user question: {context}"
        )

        # Conversation to send to the LLM
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        chat_completion = self.llm_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=conversation,
        )

        # Aggregate streamed chunks
        response = chat_completion.choices[0].message.content

        if not response:
            return "I'm sorry, but I couldn't find relevant information. Could you provide more details?"
        return response

    def on_enter_pressed(self, event):
        self.ask_question()

    def append_to_chat(self, tag, text):
        """
        Appends a message to the chat box.
        """
        self.chat_box.config(state=tk.NORMAL)
        # self.chat_box.insert(tk.END, f"{tag}: {text}\n\n")
        if tag == "USER":
            self.chat_box.insert(tk.END, f"\nYou: {text}\n\n", ("bold",))
        elif tag == "ASSISTANT":
            self.chat_box.insert(tk.END, f"AI assistant: {text}\n\n")
        elif tag == "SOURCES":
            self.chat_box.insert(tk.END, f"{text}\n")
        elif tag == "SEPARATOR":
            self.chat_box.insert(tk.END, f"{text}\n")
        self.chat_box.config(state=tk.DISABLED)
        self.chat_box.see(tk.END)