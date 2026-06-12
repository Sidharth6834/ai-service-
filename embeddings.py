from sentence_transformers import SentenceTransformer
import chromadb
import os

# Load model once when service starts (not on every request)
# This model runs locally — completely free, no API needed
model = SentenceTransformer('all-MiniLM-L6-v2')

# ChromaDB client — stores vectors on disk
chroma_client = chromadb.PersistentClient(path="./chromadb_storage")


def embed_and_store(document_id, chunks):
    """
    Convert text chunks to vectors and store in ChromaDB.
    
    What is an embedding?
    Text converted to a list of numbers (vector) that captures
    its meaning. Similar meaning = similar numbers = close in
    vector space. This lets us search by meaning, not keywords.
    
    Example:
    "evict tenant" and "remove occupant" have different words
    but similar embeddings — ChromaDB finds both when you
    search for either.
    """
    
    # Get or create a collection for this document
    # Each document gets its own collection in ChromaDB
    collection_name = f"doc_{document_id}"
    
    # Delete old collection if exists (re-analysis case)
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"document_id": document_id}
    )
    
    # Extract text from chunks
    texts = [chunk['text'] for chunk in chunks]
    
    # Convert all texts to vectors in one batch (faster)
    embeddings = model.encode(texts).tolist()
    
    # Store in ChromaDB with IDs and metadata
    collection.add(
        embeddings=embeddings,
        documents=texts,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"chunk_index": chunk['chunk_index']} 
                   for chunk in chunks]
    )
    
    return collection


def search_relevant_chunks(document_id, query, top_k=5):
    """
    Find the most relevant chunks for a given query.
    
    How it works:
    1. Convert query to vector
    2. Find top_k chunks whose vectors are closest to query vector
    3. Return those chunks as context for the LLM
    """
    
    collection_name = f"doc_{document_id}"
    
    try:
        collection = chroma_client.get_collection(collection_name)
    except:
        return []
    
    # Convert query to vector
    query_embedding = model.encode([query]).tolist()
    
    # Find similar chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count())
    )
    
    return results['documents'][0] if results['documents'] else []
