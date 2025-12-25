from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SearchRequest(BaseModel):
    query: str
    strategy: str
    # This is used during experimentation and evalauation of retrieval engine
    retrieve_only: bool = False  
    top_k: int = 5
    search_type: str = "hybrid"


class Citation(BaseModel):
    text: str
    page_number: Optional[int]
    source_file: str
    relevance_score: float


class SearchResponse(BaseModel):
    status: str
    car_model: Optional[str] = None
    strategy_used: str
    answer: Optional[str] = None
    citations: Optional[List[Citation]] = None
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StrategyInfo(BaseModel):
    key: str
    name: str
    description: str
    collection_name: str
    is_ready: bool


class StrategiesResponse(BaseModel):
    strategies: List[StrategyInfo]