import os
import faiss
import pickle
import hashlib
from sentence_transformers import SentenceTransformer
from llama_index import SimpleDirectoryReader
import numpy as np

# paths
INPUT_PDF_FOLDER = "data/input_pdfs"
INDEX_FOLDER = "data/indexes"
PROCESSED_FILES_FILE = os.path.join(INDEX_FOLDER, "processed_files.pkl")
FAISS_INDEX_FILE = os.path.join(INDEX_FOLDER, "faiss_index")
METADATA_FILE = os.path.join(INDEX_FOLDER, "metadata.pkl")

# Initialize embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Small and fast model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

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

    # Load PDFs and parse text
    reader = SimpleDirectoryReader(new_files)
    documents = reader.load_data()
    new_embeddings = []
    new_metadata = []
    for doc in documents:
        for node in doc.get_nodes():
            new_embeddings.append(embedding_model.encode(node.get_text()))
            new_metadata.append({
                "text": node.get_text(),
                "source": doc.extra_info.get("file_name", "unknown")
            })

    # Update FAISS index
    if new_embeddings:
        if faiss_index is None:
            embedding_dim = len(new_embeddings[0])
            faiss_index = faiss.IndexFlatL2(embedding_dim)
        faiss_index.add(np.array(new_embeddings))
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