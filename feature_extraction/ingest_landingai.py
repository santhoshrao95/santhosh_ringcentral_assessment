"""
Data Ingestion Script - Strategy 4: LandingAI-Based Chunking

This script ingests car manual chunks extracted by LandingAI into Weaviate.
- Source: Pre-parsed JSON files from LandingAI API
- Filtering: Keep only text, table, figure types
- Cleaning: Remove ID attributes from markdown
- Min length: 8 characters after cleaning

Collection: CarManual_BasicLandingai
"""

import os
import time
import json
import re
from pathlib import Path
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

STRATEGY_NAME = "landingai_based"
COLLECTION_NAME = "CarManual_BasicLandingai"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

ASTOR_MANUAL_FILES = [
    '../feature_extraction/pdfs_for_landing_ai/astor_manual_1_90.json',
    '../feature_extraction/pdfs_for_landing_ai/astor_manual_90_180.json',
    '../feature_extraction/pdfs_for_landing_ai/astor_manual_181_266.json'
]

TIAGO_MANUAL_FILES = [
    '../feature_extraction/pdfs_for_landing_ai/tiago_manual_1_90.json',
    '../feature_extraction/pdfs_for_landing_ai/tiago_manual_91_153.json'
]


def connect_to_weaviate():
    print("Connecting to Weaviate Cloud...")
    
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not weaviate_url or not weaviate_api_key:
        raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in .env file")
    
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )
    
    if client.is_ready():
        print("Connected to Weaviate")
        return client
    else:
        raise ConnectionError("Failed to connect to Weaviate")


def create_collection(client):
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
    
    client.collections.create(
        name=COLLECTION_NAME,
        description="Car manual chunks using LandingAI document parsing",
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(
                name="text",
                data_type=DataType.TEXT,
                description="The actual text content of the chunk"
            ),
            Property(
                name="car_model",
                data_type=DataType.TEXT,
                description="Car model name (e.g., 'MG_Astor', 'Tata_Tiago')"
            ),
            Property(
                name="page_number",
                data_type=DataType.INT,
                description="Page number in the PDF"
            ),
            Property(
                name="chunk_index",
                data_type=DataType.INT,
                description="Sequential chunk number within the document"
            ),
            Property(
                name="source_file",
                data_type=DataType.TEXT,
                description="Original PDF filename"
            ),
            Property(
                name="chunking_strategy",
                data_type=DataType.TEXT,
                description="Chunking strategy used"
            ),
            Property(
                name="element_type",
                data_type=DataType.TEXT,
                description="Type of document element from LandingAI (text, table, figure)"
            )
        ]
    )
    
    print(f"Created collection: {COLLECTION_NAME}")


def load_embedding_model():
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded")
    return model


def clean_markdown(markdown_text):
    text = re.sub(r'<a id=\'[^\']+\'></a>\s*', '', markdown_text)
    
    text = re.sub(r'\s+id="[^"]*"', '', text)
    text = re.sub(r'\s+id=\'[^\']*\'', '', text)
    
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = text.strip()
    
    return text


def load_and_merge_chunks(json_files, car_model):
    chunks = []
    last_page = 0
    
    for file_path in json_files:
        print(f"   Loading {file_path}")
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        for chunk in data['chunks']:
            chunk['grounding']['page'] = chunk['grounding']['page'] + last_page
            chunks.append(chunk)
        
        last_page = data['chunks'][-1]['grounding']['page']
        print(f"   Last page in this file: {last_page}")
    
    print(f"   Total chunks loaded: {len(chunks)}")
    return chunks


def filter_and_clean_chunks(chunks):
    filtered = []
    stats = {
        'total': len(chunks),
        'filtered_by_type': 0,
        'filtered_by_length': 0,
        'kept': 0
    }
    
    for chunk in chunks:
        if chunk['type'] not in ['text', 'table', 'figure']:
            stats['filtered_by_type'] += 1
            continue
        
        cleaned_text = clean_markdown(chunk['markdown'])
        
        if len(cleaned_text) < 8:
            stats['filtered_by_length'] += 1
            continue
        
        chunk['cleaned_text'] = cleaned_text
        filtered.append(chunk)
        stats['kept'] += 1
    
    print(f"   Total chunks: {stats['total']}")
    print(f"   Filtered by type (not text/table/figure): {stats['filtered_by_type']}")
    print(f"   Filtered by length (<8 chars): {stats['filtered_by_length']}")
    print(f"   Kept for ingestion: {stats['kept']}")
    
    return filtered


def ingest_chunks(client, model, chunks, car_model, source_file):
    collection = client.collections.get(COLLECTION_NAME)
    
    total_chunks = 0
    
    with collection.batch.dynamic() as batch:
        for idx, chunk in enumerate(tqdm(chunks, desc=f"   Embedding & inserting {car_model}")):
            embedding = model.encode(chunk['cleaned_text']).tolist()
            
            data_object = {
                "text": chunk['cleaned_text'],
                "car_model": car_model,
                "page_number": chunk['grounding']['page'],
                "chunk_index": idx,
                "source_file": source_file,
                "chunking_strategy": "landingai_based",
                "element_type": chunk['type']
            }
            
            batch.add_object(
                properties=data_object,
                vector=embedding
            )
            
            total_chunks += 1
    
    print(f"Inserted {total_chunks} chunks for {car_model}")
    return total_chunks


def print_summary(total_chunks, total_time, client):
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    print(f"Strategy:          {STRATEGY_NAME}")
    print(f"Collection:        {COLLECTION_NAME}")
    print(f"Total Chunks:      {total_chunks}")
    print(f"Time Taken:        {total_time:.2f} seconds")
    print(f"Chunks/Second:     {total_chunks/total_time:.2f}")
    
    collection = client.collections.get(COLLECTION_NAME)
    stats = collection.aggregate.over_all(total_count=True)
    print(f"Verified Count:    {stats.total_count}")
    print("="*60)


def main():
    start_time = time.time()
    
    print("\n" + "="*60)
    print("STARTING DATA INGESTION - STRATEGY 4")
    print("="*60)
    print(f"Strategy:     LandingAI-Based Chunking")
    print(f"Collection:   {COLLECTION_NAME}")
    print("="*60)
    
    client = connect_to_weaviate()
    
    try:
        create_collection(client)
        
        model = load_embedding_model()
        
        print("\nLoading Astor manual chunks...")
        astor_chunks = load_and_merge_chunks(ASTOR_MANUAL_FILES, "MG_Astor")
        
        print("\nLoading Tiago manual chunks...")
        tiago_chunks = load_and_merge_chunks(TIAGO_MANUAL_FILES, "Tata_Tiago")
        
        print("\nFiltering and cleaning Astor chunks...")
        astor_filtered = filter_and_clean_chunks(astor_chunks)
        
        print("\nFiltering and cleaning Tiago chunks...")
        tiago_filtered = filter_and_clean_chunks(tiago_chunks)
        
        total_chunks = 0
        
        print("\nIngesting Astor chunks...")
        astor_count = ingest_chunks(client, model, astor_filtered, "MG_Astor", "astor_manual.pdf")
        total_chunks += astor_count
        
        print("\nIngesting Tiago chunks...")
        tiago_count = ingest_chunks(client, model, tiago_filtered, "Tata_Tiago", "tiago_manual.pdf")
        total_chunks += tiago_count
        
        end_time = time.time()
        total_time = end_time - start_time
        print_summary(total_chunks, total_time, client)
        
        print("\nIngestion completed successfully")
        
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        raise
    
    finally:
        client.close()
        print("\nDisconnected from Weaviate")


if __name__ == "__main__":
    main()