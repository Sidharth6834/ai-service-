from pydantic import BaseModel, Field
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class ClauseAnalysis(BaseModel):
    title: str = Field(description="The title of the clause (e.g., Late Fee, Unilateral Entry, Non-Compete)")
    originalText: str = Field(description="The exact text snippet of the clause from the document")
    explanation: str = Field(description="A plain-English explanation of what this clause means")
    riskLevel: str = Field(description="The risk level of the clause: 'safe', 'caution', or 'risky'")
    suggestion: str = Field(description="Actionable suggestion or negotiation advice for the user")

class DocumentAudit(BaseModel):
    summary: str = Field(description="An executive summary of the document")
    riskLevel: str = Field(description="The overall risk level of the document: 'safe', 'caution', or 'risky'")
    clauses: List[ClauseAnalysis] = Field(description="List of key analyzed clauses inside the document")

def audit_document(text: str, doc_type: str, api_key: str) -> DocumentAudit:
    """
    Use ChatGroq with structured outputs to audit a legal document's text.
    """
    llm = ChatGroq(
        temperature=0.1,
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert legal auditor. Analyze the following document text and classify the key covenants/clauses, determine their risk level, provide a plain-English explanation and suggestions. Audit this document of type: {doc_type}."),
        ("human", "Here is the document text:\n\n{text}")
    ])
    
    # Bind structured output
    structured_llm = llm.with_structured_output(DocumentAudit)
    chain = prompt | structured_llm
    
    # Truncate text to fit within standard model context window comfortably
    truncated_text = text[:25000]
    return chain.invoke({"text": truncated_text, "doc_type": doc_type})
