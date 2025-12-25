"""
Data Ingestion Script - Strategy 1: Basic Recursive Chunking

This script ingests car manual PDFs into Weaviate using basic recursive text splitting.
- Chunk size: 400 tokens
- Overlap: 50 tokens
- Method: RecursiveCharacterTextSplitter

Collection: CarManual_BasicRecursive
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
STRATEGY_NAME = "basic_recursive"
COLLECTION_NAME = "CarManual_BasicRecursive"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Paths
PDF_FOLDER = "pdfs"
PDFS = {
    "MG_Astor": "/Users/santhosh/Documents/study_projects/ringcentral_assessment/pdfs/astor_manual.pdf",
    "Tata_Tiago": "/Users/santhosh/Documents/study_projects/ringcentral_assessment/pdfs/tiago_manual.pdf"
}


def connect_to_weaviate():
    """Connect to Weaviate Cloud"""
    print("🔌 Connecting to Weaviate Cloud...")
    
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not weaviate_url or not weaviate_api_key:
        raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in .env file")
    
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )
    
    if client.is_ready():
        print("✅ Connected to Weaviate")
        return client
    else:
        raise ConnectionError("Failed to connect to Weaviate")


def create_collection(client):
    """Create Weaviate collection for this strategy"""
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
    
    # Create new collection
    client.collections.create(
        name=COLLECTION_NAME,
        description="Car manual chunks using basic recursive splitting",
        vectorizer_config=Configure.Vectorizer.none(),  # We provide our own vectors
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
                description="Type of document element"
            )

        ]
    )
    
    print(f"✅ Created collection: {COLLECTION_NAME}")


def load_embedding_model():
    """Load Sentence Transformer model"""
    print(f"\n🤖 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Model loaded")
    return model


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF page by page"""
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        # Only include pages with meaningful text
        if text.strip():
            pages.append({
                "page_number": page_num + 1,
                "text": text.strip()
            })
    
    doc.close()
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into chunks using RecursiveCharacterTextSplitter"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    print(chunks)
    return chunks


def ingest_pdf(client, model, car_model="MG_Astor", pdf_filename="/Users/santhosh/Documents/study_projects/ringcentral_assessment/pdfs/astor_manual.pdf"):
    """Ingest a single PDF into Weaviate"""
    pdf_path = pdf_filename
    
    # Extract text from PDF
    pages = extract_text_from_pdf(pdf_path)
    print(f"   Extracted {len(pages)} pages")
    
    # Get collection
    collection = client.collections.get(COLLECTION_NAME)
    
    # Process each page
    total_chunks = 0
    chunk_index = 0
    
    with collection.batch.dynamic() as batch:
        for page in tqdm(pages, desc=f"   Chunking & embedding"):
            page_text = page["text"]
            page_num = page["page_number"]
            
            # Chunk the page text (renamed function call to be clear)
            text_chunks = chunk_text(page_text)
            
            # Process each chunk
            for chunk in text_chunks:  # Changed from chunk_text to chunk
                # Skip very small chunks
                if len(chunk.strip()) < 50:
                    continue
                
                # Generate embedding
                embedding = model.encode(chunk).tolist()
                
                # Create data object
                data_object = {
                    "text": chunk,  # Changed from chunk_text to chunk
                    "car_model": car_model,
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "source_file": pdf_filename,
                    "chunking_strategy": "basic_recursive",
                    "element_type": "text"
                }
                
                # Add to batch
                batch.add_object(
                    properties=data_object,
                    vector=embedding
                )
                
                chunk_index += 1
                total_chunks += 1
    
    print(f"✅ Inserted {total_chunks} chunks for {car_model}")
    return total_chunks



def print_summary(total_chunks, total_time, client):
    """Print ingestion summary"""
    print("\n" + "="*60)
    print("📊 INGESTION SUMMARY")
    print("="*60)
    print(f"Strategy:          {STRATEGY_NAME}")
    print(f"Collection:        {COLLECTION_NAME}")
    print(f"Chunk Size:        {CHUNK_SIZE} characters")
    print(f"Chunk Overlap:     {CHUNK_OVERLAP} characters")
    print(f"Total Chunks:      {total_chunks}")
    print(f"Time Taken:        {total_time:.2f} seconds")
    print(f"Chunks/Second:     {total_chunks/total_time:.2f}")
    
    # Get collection stats
    collection = client.collections.get(COLLECTION_NAME)
    stats = collection.aggregate.over_all(total_count=True)
    print(f"Verified Count:    {stats.total_count}")
    print("="*60)


def main():
    """Main ingestion pipeline"""
    start_time = time.time()
    
    print("\n" + "="*60)
    print("🚀 STARTING DATA INGESTION - STRATEGY 1")
    print("="*60)
    print(f"Strategy:     Basic Recursive Chunking")
    print(f"Collection:   {COLLECTION_NAME}")
    print(f"Chunk Size:   {CHUNK_SIZE} characters")
    print(f"Overlap:      {CHUNK_OVERLAP} characters")
    print("="*60)
    
    # Connect to Weaviate
    client = connect_to_weaviate()
    
    try:
        # Create collection
        create_collection(client)
        
        # Load embedding model
        model = load_embedding_model()
        
        # Ingest each PDF
        total_chunks = 0
        for car_model, pdf_filename in PDFS.items():
            chunks = ingest_pdf(client, model, car_model, pdf_filename)
            total_chunks += chunks
        
        # Print summary
        end_time = time.time()
        total_time = end_time - start_time
        print_summary(total_chunks, total_time, client)
        
        print("\n✅ Ingestion completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        raise
    
    finally:
        client.close()
        print("\n🔌 Disconnected from Weaviate")


if __name__ == "__main__":
    main()