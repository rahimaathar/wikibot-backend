from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import traceback

from .services.query_processor import process_query
from .services.wiki_search import search_wikipedia
from .services.content_analyzer import analyze_content
from .services.response_generator import generate_response

app = FastAPI(title="Wikipedia Chatbot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://wikibot-frontend-3pel.vercel.app",  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    conversation_history: Optional[List[dict]] = []

class QueryResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]

@app.post("/api/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        # Stage 1: Input Processing
        processed_query = await process_query(request.query)
        if not processed_query:
            raise HTTPException(status_code=400, detail="Failed to process query")
        
        # Stage 2: Knowledge Retrieval
        search_results = await search_wikipedia(processed_query)
        if not search_results:
            raise HTTPException(status_code=404, detail="No relevant information found")
        
        # Stage 3: Content Analysis
        analyzed_content = await analyze_content(search_results, processed_query)
        if not analyzed_content:
            raise HTTPException(status_code=500, detail="Failed to analyze content")
        
        # Stage 4: Answer Formation
        response_data = await generate_response(analyzed_content, processed_query)
        if not response_data:
            raise HTTPException(status_code=500, detail="Failed to generate response")
        
        return QueryResponse(
            response=response_data["response"],
            confidence=response_data["confidence"],
            sources=response_data["sources"]
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        error_detail = f"Error processing query: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/")
async def root():
    return {"message": "Wikipedia Chatbot API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 