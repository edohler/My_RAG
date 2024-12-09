from models import initialize_models
from chatbot import RAGChatApp


def main():
    """
    Main function to initialize models and start the RAG Chat app.
    Allows dynamic switching between semantic and BM25 retrieval methods.
    """
    embed_model, vectorstore, bm25_retriever, llm_client = initialize_models()

    # Prompt user for preferred retrieval method (or use command-line args or configs)
    use_bm25 = input("Use BM25-based retrieval? (yes/no): ").strip().lower() == "yes"

    if use_bm25:
        app = RAGChatApp(embed_model, vectorstore, llm_client, bm25_retriever=bm25_retriever)
    else:
        app = RAGChatApp(embed_model, vectorstore, llm_client)

    app.mainloop()


if __name__ == "__main__":
    main()