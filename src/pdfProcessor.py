import os
import faiss
import pickle
import hashlib
from sentence_transformers import SentenceTransformer
from llama_index.core import SimpleDirectoryReader
import numpy as np
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# paths
INPUT_PDF_FOLDER = "data/input_pdf"
INDEX_FOLDER = "data/indexes"
PROCESSED_FILES_FILE = os.path.join(INDEX_FOLDER, "processed_files.pkl")
FAISS_INDEX_FILE = os.path.join(INDEX_FOLDER, "faiss_index")
METADATA_FILE = os.path.join(INDEX_FOLDER, "metadata.pkl")

# Initialize embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Small and fast model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
text_splitter = SemanticChunker(OpenAIEmbeddings(api_key=OPENAI_API_KEY))
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def process_pdfs_and_create_index():
    # Load existing index, metadata, and processed files
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(METADATA_FILE):
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
    else:
        faiss_index = None
        metadata = []

    if os.path.exists(PROCESSED_FILES_FILE):
        with open(PROCESSED_FILES_FILE, "rb") as f:
            processed_files = pickle.load(f)
    else:
        processed_files = {}
    
    new_files = []
    for file_name in os.listdir(INPUT_PDF_FOLDER):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(INPUT_PDF_FOLDER, file_name)
            file_hash = get_file_hash(file_path)
            if file_name not in processed_files or processed_files[file_name] != file_hash:
                new_files.append(file_path)
                processed_files[file_name] = file_hash

    if not new_files:
        print("No new PDFs to process.")
        return

    print(f"Processing {len(new_files)} new/modified PDFs...")

    new_embeddings = []
    new_metadata = []
        
    reader = SimpleDirectoryReader(input_files=new_files)
    documents = reader.load_data()

    # Process each document
    for doc in documents:
        # Break the document into smaller chunks if necessary
        text_chunks = doc.text.split("\n\n")  # Simple chunking by paragraph (can be improved)
        for chunk in text_chunks:
            chunk_embedding = embedding_model.encode(chunk)
            new_embeddings.append(chunk_embedding)
            new_metadata.append({
                "text": chunk,
                "source": doc.extra_info.get("file_name", "unknown")
            })

    """
    combined_documents = []
    for file_path in new_files:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        combined_text = "\n".join([doc.page_content for doc in documents])  # Combine all pages
        combined_documents.append(Document(page_content=combined_text, metadata={"source": file_path}))
    
    idx = 0
    for doc in combined_documents:
        print(idx)
        idx = idx + 1
        # chunks = smart_chunk_text(doc)
        chunks = text_splitter.create_documents(doc.page_content)

        for chunk in chunks:
            chunk_embedding = embedding_model.encode(chunk.page_content)
            new_embeddings.append(chunk_embedding)
            new_metadata.append({
                "text": chunk.page_content,
                # "source": chunks.extra_info.get("file_name", "unknown")
                "source": os.path.basename(file_path)
            })
        # semantic_chunk_vectorstore = Chroma.from_documents(chunks, embedding=embed_model)
    """

    # Update FAISS index
    if new_embeddings:
        new_embeddings = np.array(new_embeddings)
        if faiss_index is None:
            embedding_dim = new_embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(embedding_dim)
        faiss_index.add(new_embeddings)
        metadata.extend(new_metadata)

        # Save updated FAISS index and metadata
        os.makedirs(INDEX_FOLDER, exist_ok=True)
        faiss.write_index(faiss_index, FAISS_INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(metadata, f)

        # Update processed files list
        with open(PROCESSED_FILES_FILE, "wb") as f:
            pickle.dump(processed_files, f)

        print(f"Updated FAISS index with {len(new_files)} new PDFs.")


if __name__ == "__main__":
    process_pdfs_and_create_index()