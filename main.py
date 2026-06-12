from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import analyze_document, chat_with_document
import uvicorn

app = FastAPI(title="LexAI - AI Service")

class AnalyzeRequest(BaseModel):
    documentId: str
    extractedText: str
    documentType: str
    language: str = 'english'

class ChatRequest(BaseModel):
    documentId: str
    message: str
    extractedText: str
    chatHistory: list
    language: str = 'english'

@app.get("/health")
def health():
    return {"status": "AI service running"}

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        result = analyze_document(
            req.documentId,
            req.extractedText,
            req.documentType,
            req.language
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        response = chat_with_document(
            req.documentId,
            req.message,
            req.extractedText,
            req.chatHistory,
            req.language
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
