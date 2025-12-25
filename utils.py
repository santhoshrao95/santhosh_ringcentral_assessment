import re
import yaml
from typing import List, Dict, Any
import weaviate
from weaviate.classes.query import MetadataQuery

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

params = config["generator_model_params"]
GROQ_MODEL = params['model_name']
EMBEDDING_MODEL = config["embedding_model"]
CHUNKING_STRATEGIES = config["chunking_strategies"]


def detect_car_model(query):
    car_patterns = {
    "MG_Astor": [r"\bMG\s+Astor\b",r"\bAstor\b",r"\bMG\b"],
    "Tata_Tiago": [r"\bTata\s+Tiago\b",r"\bTiago\b",r"\bTata\b"]
}
    
    for car_model, patterns in car_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return car_model
    
    return None


def query_parser(query,groq_client):

    query_type = None
    car_model = detect_car_model(query)

    return {"car_model":car_model,"type":query_type}

def query_parser_rewriter(query, groq_client):
    car_model = detect_car_model(query)
    
    if car_model:
        rewritten_query = rewrite_query(query, groq_client)
        return {"car_model": car_model, "rewritten_query": rewritten_query}
    else:
        result = detect_and_rewrite(query, groq_client)
        return result

def rewrite_query(query, groq_client):
    prompt = config['query_rewriter_prompt'].format(query=query)
    
    response = groq_client.chat.completions.create(
        model=config['generator_model_params']['model_name'],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )
    
    return response.choices[0].message.content.strip()

def detect_and_rewrite(query, groq_client):
    prompt = config['car_detection_rewriter_prompt'].format(query=query)
    
    response = groq_client.chat.completions.create(
        model=config['generator_model_params']['model_name'],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )
    
    result_text = response.choices[0].message.content.strip()
    
    lines = result_text.split('\n')
    car_model = None
    rewritten_query = query
    
    for line in lines:
        if line.startswith('CAR_MODEL:'):
            model_value = line.replace('CAR_MODEL:', '').strip()
            if model_value in ['MG_Astor', 'Tata_Tiago']:
                car_model = model_value
        elif line.startswith('QUERY:'):
            rewritten_query = line.replace('QUERY:', '').strip()
    
    return {"car_model": car_model, "rewritten_query": rewritten_query}

def collection_exists(strategy, weaviate_client) -> bool:
    if strategy not in CHUNKING_STRATEGIES:
        return False
    
    collection_name = CHUNKING_STRATEGIES[strategy]["collection_name"]
    return weaviate_client.collections.exists(collection_name)


def search_weaviate(
    query: str,
    embedding_model, 
    weaviate_client,
    car_model: str,
    strategy: str,
    search_type: str = "hybrid",
    alpha: float = 0.85,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    
    collection_name = CHUNKING_STRATEGIES[strategy]["collection_name"]
    collection = weaviate_client.collections.get(collection_name)
    
    query_embedding = embedding_model.encode(query).tolist()
    
    if search_type == "hybrid":
        response = collection.query.hybrid(
            query=query,
            vector=query_embedding,
            alpha=alpha,
            limit=top_k,
            filters=weaviate.classes.query.Filter.by_property("car_model").equal(car_model),
            return_properties=["text", "page_number", "source_file", "chunk_index"],
            return_metadata=MetadataQuery(score=True)
        )
        
        chunks = []
        for obj in response.objects:
            chunks.append({
                "text": obj.properties["text"],
                "page_number": obj.properties["page_number"],
                "source_file": obj.properties["source_file"],
                "chunk_index": obj.properties["chunk_index"],
                "relevance_score": obj.metadata.score
            })
    
    else:  # semantic
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            filters=weaviate.classes.query.Filter.by_property("car_model").equal(car_model),
            return_properties=["text", "page_number", "source_file", "chunk_index"],
            return_metadata=MetadataQuery(distance=True)
        )
        
        chunks = []
        for obj in response.objects:
            chunks.append({
                "text": obj.properties["text"],
                "page_number": obj.properties["page_number"],
                "source_file": obj.properties["source_file"],
                "chunk_index": obj.properties["chunk_index"],
                "relevance_score": 1 - obj.metadata.distance
            })
    
    return chunks
def generate_answer(groq_client, query, chunks, car_model):
    context = "\n\n".join([
        f"[Page {chunk['page_number']}] : Chuck number {idx+1}:{chunk['text']}"
        for idx, chunk in enumerate(chunks)
    ])
    
    prompt_template = params['user_prompt_template']
    system_prompt = params["system_prompt"]
    temperature = params["temperature"]
    max_tokens = params["max_tokens"]

    prompt = prompt_template.format(
        car_model=car_model.replace('_', ' '),
        context=context,
        query=query
    )
    
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()
