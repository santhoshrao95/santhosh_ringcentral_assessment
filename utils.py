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

# def is_query_factual_or_procedural(query,groq_client,groq_model):
#     prompt = f"""You are an intelligent query parser who is expert in detecting car model mentioned in the query as it is.

#         User Question: {query}

#         Instructions:
#         1. Answer the question clearly and concisely based on the manual excerpts above
#         2. If the excerpts don't contain enough information, say so
#         3. Reference specific page numbers when possible
#         4. Be helpful and practical

#         Answer:"""
    
#     # Call Groq API
#     response = groq_client.chat.completions.create(
#         model=GROQ_MODEL,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a helpful car manual assistant. Provide clear, accurate answers based on the manual excerpts provided."
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         temperature=0,
#         max_tokens=500
#     )
    
#     return response.choices[0].message.content.strip()

def query_parser(query,groq_client):

    query_type = None
    car_model = detect_car_model(query)

    return {"car_model":car_model,"type":query_type}

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
