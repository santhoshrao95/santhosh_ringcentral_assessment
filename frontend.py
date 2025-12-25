"""
Streamlit Frontend for Car Manual RAG System

This frontend provides a user interface for querying car manuals.
Users can select different chunking strategies and get answers with citations.
"""

import os
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Car Manual Assistant",
    page_icon="car",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_backend_health() -> bool:
    """Check if backend is healthy and reachable"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "healthy"
        return False
    except Exception:
        return False


def get_available_strategies() -> Optional[Dict[str, Any]]:
    """Fetch available chunking strategies from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/strategies", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def search_manual(query: str, strategy: str, top_k: int, search_type: str) -> Optional[Dict[str, Any]]:
    """Send search request to backend"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/search",
            json={
                "query": query,
                "strategy": strategy,
                "retrieve_only": False,
                "top_k": top_k,
                "search_type": search_type
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "error",
                "message": f"Backend error: {response.status_code}"
            }
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "message": "Request timed out. Please try again."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error connecting to backend: {str(e)}"
        }


def display_citations(citations):
    """Display citations in a formatted way"""
    st.markdown("### Sources")
    
    for i, citation in enumerate(citations, 1):
        with st.expander(f"Source {i} - Page {citation['page_number']} (Relevance: {citation['relevance_score']:.3f})"):
            st.markdown(f"**File:** {citation['source_file']}")
            st.markdown(f"**Page:** {citation['page_number']}")
            st.markdown("**Excerpt:**")
            st.text(citation['text'])


def display_metadata(metadata):
    """Display metadata in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Query Metadata")
    st.sidebar.markdown(f"**Collection:** {metadata.get('collection', 'N/A')}")
    st.sidebar.markdown(f"**Chunks Retrieved:** {metadata.get('chunks_retrieved', 0)}")
    st.sidebar.markdown(f"**Processing Time:** {metadata.get('processing_time_ms', 0):.2f} ms")

def main():
    # Initialize session state
    if 'selected_strategy' not in st.session_state:
        st.session_state.selected_strategy = None
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5
    if 'search_type' not in st.session_state:
        st.session_state.search_type = "hybrid"
    
    # Title
    st.title("Car Manual Assistant")
    st.markdown("Ask questions about MG Astor or Tata Tiago manuals")
    
    # Check backend health
    if not check_backend_health():
        st.error("Backend is not reachable. Please make sure the backend server is running.")
        st.info(f"Expected backend URL: {BACKEND_URL}")
        st.stop()
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Fetch available strategies
        strategies_data = get_available_strategies()
        
        if strategies_data is None:
            st.error("Failed to fetch strategies from backend")
            st.stop()
        
        strategies = strategies_data.get("strategies", [])
        
        if not strategies:
            st.error("No chunking strategies available")
            st.stop()
        
        # Create strategy options
        strategy_options = {}
        ready_strategies = []
        
        for strategy in strategies:
            strategy_options[strategy["key"]] = strategy["name"]
            if strategy["is_ready"]:
                ready_strategies.append(strategy["key"])
        
        # Strategy selector with session state
        if ready_strategies:
            # Set default value if not set
            if st.session_state.selected_strategy is None or st.session_state.selected_strategy not in ready_strategies:
                st.session_state.selected_strategy = ready_strategies[0]
            
            selected_strategy = st.selectbox(
                "Chunking Strategy",
                options=ready_strategies,
                index=ready_strategies.index(st.session_state.selected_strategy),
                format_func=lambda x: strategy_options[x],
                help="Select the text chunking method to use for retrieval",
                key="strategy_selector"
            )
            
            # Update session state
            st.session_state.selected_strategy = selected_strategy
            
            # Show strategy details
            selected_strategy_info = next(
                (s for s in strategies if s["key"] == selected_strategy),
                None
            )
            
            if selected_strategy_info:
                st.markdown("**Description:**")
                st.markdown(selected_strategy_info["description"])
                st.markdown(f"**Collection:** {selected_strategy_info['collection_name']}")
                st.markdown(f"**Status:** {'Ready' if selected_strategy_info['is_ready'] else 'Not Ready'}")
        else:
            st.error("No strategies are ready. Please run ingestion scripts first.")
            st.stop()
        
        st.markdown("---")
        
        # Top K selector
        top_k_options = [5, 10, 15, 25]
        top_k = st.selectbox(
            "Top K Results",
            options=top_k_options,
            index=top_k_options.index(st.session_state.top_k) if st.session_state.top_k in top_k_options else 0,
            help="Number of document chunks to retrieve",
            key="top_k_selector"
        )
        st.session_state.top_k = top_k
        
        # Search Type selector
        search_type_options = ["semantic", "hybrid"]
        search_type_display = {
            "semantic": "Semantic",
            "hybrid": "Hybrid"
        }
        
        search_type = st.selectbox(
            "Search Type",
            options=search_type_options,
            index=search_type_options.index(st.session_state.search_type) if st.session_state.search_type in search_type_options else 1,
            format_func=lambda x: search_type_display[x],
            help="Semantic: Vector similarity search only\nHybrid: Combines vector and keyword search",
            key="search_type_selector"
        )
        st.session_state.search_type = search_type
        
        st.markdown("---")
        st.markdown("### Available Models")
        st.markdown("- MG Astor")
        st.markdown("- Tata Tiago")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses RAG (Retrieval-Augmented Generation) to answer 
        questions from car manuals.
        
        **Technology:**
        - Vector DB: Weaviate Cloud
        - Embeddings: Sentence Transformers
        - LLM: Groq (Llama 3.1)
        """)
    
    # Main content area
    st.markdown("---")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., How to turn on indicator in MG Astor?",
        help="Ask about MG Astor or Tata Tiago features"
    )
    
    # Search button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Process search
    if search_button:
        if not query.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("Searching manual..."):
                result = search_manual(query, selected_strategy, top_k, search_type)
            
            if result is None:
                st.error("Failed to get response from backend")
            elif result["status"] == "error":
                st.error(f"Error: {result.get('message', 'Unknown error')}")
            elif result["status"] == "not_available":
                st.warning(result["message"])
                st.info("Available models: MG Astor, Tata Tiago")
            elif result["status"] == "no_results":
                st.warning(f"No relevant information found for {result.get('car_model', 'the query')}")
            elif result["status"] == "success":
                # Display answer
                st.markdown("### Answer")
                st.markdown(result["answer"])
                
                # Display citations
                if result.get("citations"):
                    st.markdown("---")
                    display_citations(result["citations"])
                
                # Display metadata in sidebar
                if result.get("metadata"):
                    display_metadata(result["metadata"])
                
                # Show detected car model
                st.info(f"Detected Car Model: {result.get('car_model', 'Unknown').replace('_', ' ')}")
                st.info(f"Strategy Used: {result.get('strategy_used', 'Unknown')}")
    
    # Example queries
    st.markdown("---")
    st.markdown("### Example Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **MG Astor:**
        - How to turn on indicator in MG Astor?
        - What is the fuel tank capacity of MG Astor?
        - How to adjust mirrors in MG Astor?
        - Who is santhosh in mg?
        - How to turn on indicator?
        """)
    
    with col2:
        st.markdown("""
        **Tata Tiago:**
        - Which engine oil to use in Tiago?
        - What is the tire pressure for Tata Tiago?
        - How to change headlight in Tiago?
        - Who is santhosh in tata?
        - How to turn on indicator?
        """)


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    main()

# Run with: streamlit run frontend.py