import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from retrievers import BM25Retriever
from groq import Groq


def initialize_models(index_folder="data/indexes"):
    """
    Initializes the models and vectorstores required for the RAG Chat app.
    :param index_folder: Path to the folder where index files are stored.
    :return: embed_model, vectorstore, bm25_retriever, llm_client
    """
    load_dotenv()
    API_KEY = os.getenv("GROQ_API_KEY")
    if not API_KEY:
        raise ValueError("API key is missing! Please set GROQ_API_KEY in your environment.")

    # Initialize embedding model
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Initialize Chroma vector store
    chroma_db_path = os.path.join(index_folder, "chroma")
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embed_model)

    # Load metadata and documents
    all_data = vectorstore.get(include=["metadatas", "documents"])
    documents = [
        {"content": doc, "metadata": meta}
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]

    # Initialize BM25 retriever
    tokenized_corpus = [word_tokenize(doc["content"].lower()) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_retriever = BM25Retriever(bm25, documents)

    llm_client = Groq(api_key=API_KEY)

    return embed_model, vectorstore, bm25_retriever, llm_client