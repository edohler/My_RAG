# **My RAG System**
created by Erik Döhler, 16.11.2024 Bern

This project implements a Retrieval-Augmented Generation (RAG) pipeline using a pre-trained LLM (Meta LLaMA) and a FAISS-based vector database for document retrieval. The system allows querying scientific papers or other documents in PDF format, returning context-aware answers and their sources.

## **Features**
- **Document Ingestion**: Automatically processes and indexes new PDFs added to the project.
- **Vector Database**: Uses FAISS for efficient similarity search.
- **Question Answering**: Combines retrieved document chunks with a powerful LLM for accurate and context-aware responses.
- **Source Tracking**: Includes the source document(s) in the response for transparency.

---

## **Setup**

### Prerequisites
1. **Python** (>=3.8)
2. **Git**
3. An **API key** for the Hugging Face LLM.

### Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>

2. Install dependencies:
   pip install -r requirements.txt

3. Create a .env file to store your Hugging Face API key:
   HF_API_KEY=your-hugging-face-api-key

## **Usage**

1. Ingest PDFs
To process and index PDFs:
    ```bash
    python src\pdfProcessor.py

2. Ask Questions
Run the query script and enter your question when prompted:
    ```bash
    python src\rag.py