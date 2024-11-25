import os
import faiss
import pickle
import hashlib
import numpy as np
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# paths
INPUT_PDF_FOLDER = "data/input_pdf"
INDEX_FOLDER = "data/indexes"
PROCESSED_FILES_FILE = os.path.join(INDEX_FOLDER, "processed_files.pkl")

# Initialize embedding model
text_splitterSem = SemanticChunker(OpenAIEmbeddings(api_key=OPENAI_API_KEY))
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
text_splitterRec = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=True,
    separators=["\n", " ", ".", ","]
)

# Chroma storage path
CHROMA_DB_PATH = os.path.join(INDEX_FOLDER, "chroma")
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embed_model)

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def process_batch(batch):
    try:
        vectorstore.add_documents(batch)
    except Exception as e:
        print(f"Error processing batch: {e}")

def process_pdfs_and_create_index():
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

    # Process new files
    new_chunks = []
    for file_path in new_files:
        # Load and combine PDF content
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        combined_text = "\n".join([doc.page_content for doc in documents])  # Combine all pages

        # Split into chunks
        chunks = text_splitterRec.create_documents([combined_text])
        for chunk in chunks:
            chunk.metadata["source"] = file_path  # Add source metadata
        
        new_chunks.extend(chunks)

    # Add new chunks to the Chroma vector store
    batch_size = 1000
    if new_chunks:
        batches = [new_chunks[i:i+batch_size] for i in range(0, len(new_chunks), batch_size)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            for future in as_completed(futures):
                future.result()
                # print(f"{i} chunks added of total {len(new_chunks)} chunks ...")
        
        print(f"Added {len(new_chunks)} chunks to the vector store.")

    # Save the updated processed files list
    with open(PROCESSED_FILES_FILE, "wb") as f:
        pickle.dump(processed_files, f)

    print(f"Updated processed files list with {len(new_files)} PDFs.")


if __name__ == "__main__":
    process_pdfs_and_create_index()