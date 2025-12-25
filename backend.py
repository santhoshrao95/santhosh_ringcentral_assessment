import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
from groq import Groq

import yaml 

from models import (
    SearchRequest,
    SearchResponse,
    Citation,
    StrategyInfo,
    StrategiesResponse,
)

from utils import CHUNKING_STRATEGIES, query_parser, generate_answer, search_weaviate, collection_exists, query_parser_rewriter

load_dotenv()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
GROQ_MODEL = config["generator_model_params"]["model_name"]
EMBEDDING_MODEL = config["embedding_model"]
CHUNKING_STRATEGIES = config["chunking_strategies"]
DEFAULT_STRATEGY = "basic_recursive"

params = config["generator_model_params"]
system_prompt = params["system_prompt"]
temperature = params["temperature"]
max_tokens = params["max_tokens"]

app = FastAPI(
    title="Car Manual RAG API",
    description="Query car manuals using RAG with multiple chunking strategies",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

weaviate_client = None
embedding_model = None
groq_client = None

@app.on_event("startup")
async def startup_event():
    global weaviate_client, embedding_model, groq_client

    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    if not weaviate_url or not weaviate_api_key:
        raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set")
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )
    if weaviate_client.is_ready():
        print("Connected to Weaviate")
    else:
        raise ConnectionError("Failed to connect to Weaviate")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY must be set")
    groq_client = Groq(api_key=groq_api_key)


@app.on_event("shutdown")
async def shutdown_event():
    global weaviate_client
    if weaviate_client:
        weaviate_client.close()
        print("Weaviate connection closed")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Car Manual RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "strategies": "/strategies",
            "search": "/search (POST)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "weaviate_connected": weaviate_client.is_ready() if weaviate_client else False,
        "embedding_model_loaded": embedding_model is not None,
        "groq_client_ready": groq_client is not None
    }


@app.get("/strategies", response_model=StrategiesResponse)
async def get_strategies():
    """Get list of available chunking strategies"""
    strategies = []
    
    for key, config in CHUNKING_STRATEGIES.items():
        strategies.append(StrategyInfo(
            key=key,
            name=config["display_name"],
            description=config["description"],
            collection_name=config["collection_name"],
            is_ready=collection_exists(key,weaviate_client)
        ))
    
    return StrategiesResponse(strategies=strategies)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """rag pipeline"""
    start_time = time.time()
    strategy = request.strategy or DEFAULT_STRATEGY
    if strategy not in CHUNKING_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy: {strategy}. Available: {list(CHUNKING_STRATEGIES.keys())}"
        )
    if not collection_exists(strategy, weaviate_client):
        raise HTTPException(
            status_code=503,
            detail=f"Collection not ready for strategy: {strategy}. Run ingestion script first."
        )
    
    parser_result = query_parser_rewriter(request.query,groq_client)
    car_model = parser_result["car_model"]
    query = parser_result["rewritten_query"]
    # print(query)
    
    if not car_model:
        return SearchResponse(
            status="not_available",
            strategy_used=strategy,
            message="Manual is not available for this car/model.",
            metadata={
                "available_models": ["MG Astor", "Tata Tiago"],
                "detected_query": request.query
            }
        )
    
    try:
        chunks = search_weaviate(query, embedding_model, weaviate_client, car_model, strategy, top_k=request.top_k, alpha=0.8, search_type=request.search_type)
        
        if not chunks:
            return SearchResponse(
                status="no_results",
                car_model=car_model,
                strategy_used=strategy,
                message="No relevant information found in the manual.",
                metadata={"chunks_retrieved": 0}
            )

        citations = [
            Citation(
                text=chunk["text"],
                page_number=chunk["page_number"],
                source_file=chunk["source_file"],
                relevance_score=round(chunk["relevance_score"], 2)
            )
            for chunk in chunks[:request.top_k]
        ]
        # print(chunks)
        if request.retrieve_only:
            print(chunks)
            processing_time = time.time() - start_time
            return SearchResponse(
                status="success",
                car_model=car_model,
                strategy_used=strategy,
                answer="[RETRIEVE_ONLY_MODE] Answer generation skipped for evaluation purposes.",
                citations=citations,
                metadata={
                    "collection": CHUNKING_STRATEGIES[strategy]["collection_name"],
                    "chunks_retrieved": len(chunks),
                    "chunks_used": len(chunks),
                    "processing_time_ms": round(processing_time * 1000, 2),
                    "retrieve_only": True
                }
            )
        
        answer = generate_answer(groq_client, query, chunks, car_model)
        # print(answer)
        
        processing_time = time.time() - start_time
        
        return SearchResponse(
            status="success",
            car_model=car_model,
            strategy_used=strategy,
            answer=answer,
            citations=citations,
            metadata={
                "collection": CHUNKING_STRATEGIES[strategy]["collection_name"],
                "chunks_retrieved": len(chunks),
                "chunks_used": len(chunks),
                "processing_time_ms": round(processing_time * 1000, 2)
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )