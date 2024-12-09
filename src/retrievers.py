from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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


def weighted_query_embedding(question, conversation_history, embed_model, weight_decay=0.5):
    question_embedding = embed_model.embed_query(question)
    combined_embedding = np.array(question_embedding) * 1.0

    weight = 1.0
    for entry in reversed(conversation_history):
        if entry["role"] == "user":
            weight *= weight_decay
            history_embedding = embed_model.embed_query(entry["content"])
            combined_embedding += np.array(history_embedding) * weight

    combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
    return combined_embedding


def hybrid_similarity_search(question, conversation_history, embed_model, vectorstore, bm25_retriever, top_k=5):
    query_embedding = weighted_query_embedding(question, conversation_history, embed_model)

    vector_results = vectorstore.similarity_search_by_vector(query_embedding, k=top_k)
    bm25_results = bm25_retriever.get_relevant_documents(question)[:top_k]

    combined_results = vector_results + bm25_results
    scored_results = []
    for res in combined_results:
        page_content = res.page_content if hasattr(res, 'page_content') else res.get('page_content', '')
        result_embedding = embed_model.embed_query(page_content)
        score = cosine_similarity([query_embedding], [result_embedding])[0][0]
        scored_results.append((res, score))

    scored_results.sort(key=lambda x: x[1], reverse=True)
    formatted_results = [
        {
            "text": res[0].page_content if hasattr(res[0], 'page_content') else res[0].get('page_content', ''),
            "source": res[0].metadata.get("source", "unknown") if hasattr(res[0], 'metadata') else res[0].get('metadata', {}).get("source", "unknown"),
            "score": res[1],
        }
        for res in scored_results[:top_k]
    ]

    return formatted_results


def refine_question_with_llm(question, conversation_history, llm_client):
    """
    Uses an LLM to refine the user's question based on the last two exchanges in the conversation history.

    :param question: The current user question.
    :param conversation_history: List of conversation history entries.
    :param llm_client: The LLM client instance for generating the refined question.
    :return: Refined question as a string.
    """
    # Select the last two exchanges (if available)
    last_two_exchanges = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history

    # Prepare the LLM conversation messages
    messages = []

    # Add the task prompt for refining the question
    messages.append({
        "role": "system",
        "content": (
            "You are a helpful assistant tasked with refining questions for semantic search. "
            "Your task is to rewrite the user's question to make it more precise and focused, "
            "based on the provided conversation context. "
            "Do not answer the question. Provide only the refined question as your output."
        ),
    })

    # Add a clear marker for the conversation context
    if last_two_exchanges:
        context_content = "\n".join([f"{exchange['role']}: {exchange['content']}" for exchange in last_two_exchanges])
        messages.append({
            "role": "system",
            "content": f"The following is the relevant conversation context:\n{context_content}"
        })

    # Add the user's question
    messages.append({"role": "user", "content": f"The following is the question you shall refine:\n{question}"})

    # Debugging: Print the messages to ensure correctness
    # print("Prepared Messages for LLM:", messages)

    # Generate the refined question using the LLM
    chat_completion = llm_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
    )
    refined_question = chat_completion.choices[0].message.content
    
    # Debugging: Print refined question
    print(refined_question.strip())
    
    return refined_question.strip()


def semantic_search(question, vectorstore, embed_model, top_k=5):
    """
    Performs semantic search using similarity_search_by_vector and cosine similarity.

    :param question: The question to search for.
    :param vectorstore: The vectorstore instance.
    :param embed_model: The embedding model for generating query vectors.
    :param top_k: Number of top results to return (default: 5).
    :return: List of relevant documents with their similarity scores.
    """
    # Generate the query embedding
    query_embedding = embed_model.embed_query(question)

    # Perform similarity search using vectorstore
    search_results = vectorstore.similarity_search_by_vector(query_embedding, k=top_k)

    # Re-score results with cosine similarity for additional precision
    results_with_scores = []
    for result in search_results:
        result_embedding = embed_model.embed_query(result.page_content)
        score = cosine_similarity([query_embedding], [result_embedding])[0][0]
        results_with_scores.append({"text": result.page_content, "source": result.metadata["source"], "score": score})

    # Sort results by similarity score in descending order
    results_with_scores.sort(key=lambda x: x["score"], reverse=True)
    return results_with_scores