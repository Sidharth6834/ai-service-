def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split document text into overlapping chunks.
    
    Why chunking? LLMs have token limits. A 10-page legal
    document cannot fit in one prompt. We split it into
    small pieces and only send relevant pieces to the LLM.
    
    Why overlap? So context at chunk boundaries is not lost.
    Example: If clause starts at end of chunk 3 and continues
    in chunk 4, overlap ensures both chunks have that context.
    """
    
    # Clean the text first
    text = text.replace('\n\n\n', '\n\n')
    text = text.strip()
    
    words = text.split()
    chunks = []
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append({
            'text': chunk,
            'start_word': start,
            'end_word': min(end, len(words)),
            'chunk_index': len(chunks)
        })
        # Move forward by chunk_size minus overlap
        start += chunk_size - overlap
    
    return chunks


def chunk_by_clause(text):
    """
    Try to split by natural legal clause boundaries first.
    Falls back to word-based chunking if no clear structure.
    """
    import re
    
    # Common Indian legal document clause patterns
    clause_patterns = [
        r'\n\s*\d+\.\s+[A-Z]',      # "1. CLAUSE NAME"
        r'\n\s*CLAUSE\s+\d+',        # "CLAUSE 1"
        r'\n\s*Article\s+\d+',       # "Article 1"
        r'\n\s*[A-Z]{3,}\s*\n',      # "TERMINATION" as standalone heading
    ]
    
    combined_pattern = '|'.join(clause_patterns)
    splits = re.split(combined_pattern, text)
    
    # If we found meaningful splits, use them
    if len(splits) > 3:
        return [{'text': s.strip(), 'chunk_index': i} 
                for i, s in enumerate(splits) if s.strip()]
    
    # Otherwise fall back to word-based chunking
    return chunk_text(text)
