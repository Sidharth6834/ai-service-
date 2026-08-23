from groq import Groq
from chunker import chunk_by_clause
from embeddings import embed_and_store, search_relevant_chunks
from prompts import get_analysis_prompt, get_chat_prompt
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_groq_with_retry(messages, temperature=0.1, max_tokens=4000, retries=2):
    """
    Call Groq API with automatic retry and model fallback on failures.
    Primary: llama-3.3-70b-versatile
    Fallbacks: llama-3.1-8b-instant, gemma2-9b-it
    """
    # Updated 2026-08-23: llama/gemma models removed from Groq, using current available models
    fallback_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b", "groq/compound"]

    for model_name in fallback_models:
        for attempt in range(retries):
            try:
                print(f"Calling Groq model: {model_name} (attempt {attempt + 1})")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                print(f"Groq call succeeded with model: {model_name}")
                return response.choices[0].message.content

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"Groq error [{model_name}] attempt {attempt + 1}: {error_type}: {error_msg}")

                # Rate limit — wait before retry
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    wait_time = 25 * (attempt + 1)
                    print(f"Rate limit detected. Waiting {wait_time}s...")
                    if attempt < retries - 1:
                        time.sleep(wait_time)
                    else:
                        print(f"Rate limit exhausted for {model_name}, trying fallback...")
                        break  # Try next model

                # Bad request — no point retrying this model
                elif "400" in error_msg or "invalid" in error_msg.lower():
                    print(f"Bad request for {model_name}, trying fallback model...")
                    break

                # Other errors — short wait then retry
                else:
                    if attempt < retries - 1:
                        time.sleep(5)
                    else:
                        break

    raise RuntimeError("All Groq models and retries exhausted. Could not complete LLM analysis.")


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
    print(f"Prompt length: {len(prompt)} chars")

    raw_response = call_groq_with_retry(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4000
    )

    # STEP 4: Parse JSON response
    print("Parsing LLM response...")
    print(f"Response preview: {raw_response[:300]}")

    try:
        # Clean response in case LLM adds extra text
        clean = raw_response.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0]
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0]

        result = json.loads(clean)

        # Ensure required keys exist
        result.setdefault("summary", "Document analysis completed.")
        result.setdefault("overallRisk", "caution")
        result.setdefault("clauses", [])

        print(f"Analysis complete. Found {len(result['clauses'])} clauses")
        return result

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Full raw response: {raw_response}")
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
    document_type = "legal document"

    prompt = get_chat_prompt(
        message,
        relevant_chunks,
        chat_history,
        document_type,
        language
    )

    # STEP 4: Get answer from LLM
    return call_groq_with_retry(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
