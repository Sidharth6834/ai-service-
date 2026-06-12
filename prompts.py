def get_analysis_prompt(extracted_text, document_type, language='english'):
    prompt = f"""
You are an expert Indian legal document analyzer. 
Analyze the following {document_type} and respond ONLY in valid JSON.

DOCUMENT TEXT:
{extracted_text[:6000]}

Your task:
1. Write a 3-4 sentence plain English summary of this document
2. Identify ALL clauses/sections in the document
3. For each clause assess risk level:
   - "safe" = standard, fair, legally normal
   - "caution" = unusual but not illegal, tenant/signer should know
   - "risky" = unfair, potentially illegal, strongly disadvantageous

Respond ONLY with this exact JSON structure, no other text:
{{
  "summary": "2-3 sentence summary here",
  "overallRisk": "safe|caution|risky",
  "clauses": [
    {{
      "title": "Clause name",
      "originalText": "exact text from document",
      "explanation": "plain English explanation in simple language",
      "riskLevel": "safe|caution|risky",
      "suggestion": "what the person should do or negotiate (only if caution or risky)"
    }}
  ]
}}

Important rules:
- Write explanations as if talking to a person with no legal knowledge
- Flag non-refundable deposits as risky (illegal in many Indian states)
- Flag notice periods under 30 days as caution
- Flag rent increases above 10% annually as caution  
- Flag inspection without notice rights as risky
- Be specific about Indian law where relevant
"""
    if language == 'hindi':
        prompt += """
- Respond with all 'explanation' and 'suggestion' fields in simple Hindi language (Devanagari script). Keep clause 'title' and 'originalText' in English. Write Hindi as a common person would speak it, not formal legal Hindi.
- Also respond with the 'summary' field in simple Hindi language (Devanagari script) reflecting a clear contract summary.
"""
    return prompt


def get_chat_prompt(question, context_chunks, chat_history, document_type, language='english'):
    history_text = ""
    for msg in chat_history[-6:]:  # Last 6 messages for context
        role = "User" if msg['role'] == 'user' else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    
    context = "\n\n".join(context_chunks)
    
    prompt = f"""
You are a helpful legal assistant helping someone understand 
their {document_type}. Answer based ONLY on the document content 
provided. If something is not in the document, say so clearly.

RELEVANT DOCUMENT SECTIONS:
{context}

PREVIOUS CONVERSATION:
{history_text}

USER QUESTION: {question}

Answer in simple, clear language. If the clause is risky or 
unfair to the user, say so directly. Keep answer under 150 words.
"""
    if language == 'hindi':
        prompt += "\nRespond in simple Hindi (Devanagari script).\n"
        
    return prompt
