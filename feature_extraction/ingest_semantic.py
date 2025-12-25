"""
Data Ingestion Script - Strategy 3: Semantic Chunking

This script ingests car manual PDFs into Weaviate using semantic similarity chunking.
Chunks are created based on semantic similarity between sentences, ensuring
that related content stays together.

Collection: CarManual_SemanticBased
"""

import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from sentence_transformers import SentenceTransformer
import fitz
import numpy as np
from tqdm import tqdm
import nltk
from typing import List, Tuple

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

load_dotenv()

STRATEGY_NAME = "semantic_based"
COLLECTION_NAME = "CarManual_SemanticBased_50"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

PDFS = {
    "MG_Astor": "/Users/santhosh/Documents/study_projects/ringcentral_assessment/pdfs/astor_manual.pdf",
    "Tata_Tiago": "/Users/santhosh/Documents/study_projects/ringcentral_assessment/pdfs/tiago_manual.pdf"
}


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
        description="Car manual chunks using semantic similarity chunking",
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
                description="Car model name"
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


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        if text.strip():
            pages.append({
                "page_number": page_num + 1,
                "text": text.strip()
            })
    
    doc.close()
    return pages


def semantic_chunk_text(text: str, model: SentenceTransformer, 
                       similarity_threshold: float = 0.5,
                       max_chunk_size: int = 800) -> List[str]:
    
    sentences = nltk.sent_tokenize(text)
    
    if len(sentences) == 0:
        return []
    
    if len(sentences) == 1:
        return [sentences[0]]
    
    sentence_embeddings = model.encode(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    current_chunk_embedding = sentence_embeddings[0]
    
    for i in range(1, len(sentences)):
        sentence = sentences[i]
        sentence_embedding = sentence_embeddings[i]
        
        similarity = np.dot(current_chunk_embedding, sentence_embedding) / (
            np.linalg.norm(current_chunk_embedding) * np.linalg.norm(sentence_embedding)
        )
        
        current_chunk_text = " ".join(current_chunk)
        projected_length = len(current_chunk_text) + len(sentence)
        
        if similarity >= similarity_threshold and projected_length <= max_chunk_size:
            current_chunk.append(sentence)
            current_chunk_embedding = np.mean(
                [current_chunk_embedding, sentence_embedding], axis=0
            )
        else:
            chunks.append(current_chunk_text)
            current_chunk = [sentence]
            current_chunk_embedding = sentence_embedding
    
    final_chunk_text = " ".join(current_chunk)
    chunks.append(final_chunk_text)
    
    return chunks


def ingest_pdf(client, model, car_model, pdf_filename, 
               similarity_threshold, max_chunk_size):
    
    pdf_path = pdf_filename
    
    pages = extract_text_from_pdf(pdf_path)
    print(f"   Extracted {len(pages)} pages")
    
    collection = client.collections.get(COLLECTION_NAME)
    
    total_chunks = 0
    chunk_index = 0
    
    with collection.batch.dynamic() as batch:
        for page in tqdm(pages, desc=f"   Chunking & embedding"):
            page_text = page["text"]
            page_num = page["page_number"]
            
            chunks = semantic_chunk_text(
                page_text, 
                model,
                similarity_threshold=similarity_threshold,
                max_chunk_size=max_chunk_size
            )
            
            for chunk in chunks:
                if len(chunk.strip()) < 50:
                    continue
                
                embedding = model.encode(chunk).tolist()
                
                data_object = {
                    "text": chunk,
                    "car_model": car_model,
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "source_file": pdf_filename,
                    "chunking_strategy": "semantic_based",
                    "element_type": "text"
                }
                
                batch.add_object(
                    properties=data_object,
                    vector=embedding
                )
                
                chunk_index += 1
                total_chunks += 1
    
    print(f"Inserted {total_chunks} chunks for {car_model}")
    return total_chunks


def print_summary(total_chunks, total_time, client, similarity_threshold, 
                 max_chunk_size):
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    print(f"Strategy:              {STRATEGY_NAME}")
    print(f"Collection:            {COLLECTION_NAME}")
    print(f"Similarity Threshold:  {similarity_threshold}")
    print(f"Max Chunk Size:        {max_chunk_size} characters")
    print(f"Total Chunks:          {total_chunks}")
    print(f"Time Taken:            {total_time:.2f} seconds")
    print(f"Chunks/Second:         {total_chunks/total_time:.2f}")
    
    collection = client.collections.get(COLLECTION_NAME)
    stats = collection.aggregate.over_all(total_count=True)
    print(f"Verified Count:        {stats.total_count}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Ingest car manuals using semantic chunking")
    parser.add_argument("--similarity-threshold", type=float, default=0.5,
                       help="Cosine similarity threshold for grouping sentences (0.0-1.0)")
    parser.add_argument("--max-chunk-size", type=int, default=800,
                       help="Maximum chunk size in characters")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("\n" + "="*60)
    print("STARTING DATA INGESTION - STRATEGY 3")
    print("="*60)
    print(f"Strategy:              Semantic Chunking")
    print(f"Collection:            {COLLECTION_NAME}")
    print(f"Similarity Threshold:  {args.similarity_threshold}")
    print(f"Max Chunk Size:        {args.max_chunk_size} characters")
    print("="*60)
    
    client = connect_to_weaviate()
    
    try:
        create_collection(client)
        
        model = load_embedding_model()
        
        total_chunks = 0
        for car_model, pdf_filename in PDFS.items():
            chunks = ingest_pdf(
                client, model, car_model, pdf_filename,
                args.similarity_threshold,
                args.max_chunk_size
            )
            total_chunks += chunks
        
        end_time = time.time()
        total_time = end_time - start_time
        print_summary(total_chunks, total_time, client,
                     args.similarity_threshold,
                     args.max_chunk_size)
        
        print("\nIngestion completed successfully")
        
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        raise
    
    finally:
        client.close()
        print("\nDisconnected from Weaviate")


if __name__ == "__main__":
    main()
