from groq import Groq
from chunker import chunk_by_clause
from embeddings import embed_and_store, search_relevant_chunks
from prompts import get_analysis_prompt, get_chat_prompt
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def analyze_document(document_id, extracted_text, document_type, language='english'):
    """
    Full RAG analysis pipeline for a legal document.
    This is what runs when user uploads a document.
    """
    
    print(f"Starting analysis for document: {document_id} in language: {language}")
    
    # STEP 1: Chunk the document
    print("Chunking document...")
    chunks = chunk_by_clause(extracted_text)
    print(f"Created {len(chunks)} chunks")
    
    # STEP 2: Embed and store in ChromaDB
    print("Embedding chunks into ChromaDB...")
    embed_and_store(document_id, chunks)
    print("Chunks stored in vector DB")
    
    # STEP 3: Send to LLM for analysis
    print("Sending to LLM for clause analysis...")
    prompt = get_analysis_prompt(extracted_text, document_type, language)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # Low temp = more consistent output
        max_tokens=4000
    )
    
    raw_response = response.choices[0].message.content
    
    # STEP 4: Parse JSON response
    print("Parsing LLM response...")
    try:
        # Clean response in case LLM adds extra text
        clean = raw_response.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0]
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0]
        
        result = json.loads(clean)
        print(f"Analysis complete. Found {len(result['clauses'])} clauses")
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        # Return safe fallback
        return {
            "summary": "Document analysis completed. Please review clauses below.",
            "overallRisk": "caution",
            "clauses": []
        }


def chat_with_document(document_id, message, extracted_text, chat_history, language='english'):
    """
    RAG-based chat — answers questions about a specific document.
    """
    
    # STEP 1: Find relevant chunks using vector search
    relevant_chunks = search_relevant_chunks(
        document_id, 
        message, 
        top_k=5
    )
    
    # STEP 2: If no chunks found (first chat), re-embed document
    if not relevant_chunks:
        chunks = chunk_by_clause(extracted_text)
        embed_and_store(document_id, chunks)
        relevant_chunks = search_relevant_chunks(document_id, message)
    
    # STEP 3: Build prompt with context
    # Detect document type from chat history or default
    document_type = "legal document"
    
    prompt = get_chat_prompt(
        message,
        relevant_chunks,
        chat_history,
        document_type,
        language
    )
    
    # STEP 4: Get answer from LLM
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    
    return response.choices[0].message.content
