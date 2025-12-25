#!/bin/bash

# ============================================
# Docker Run Script for Car Manual RAG System
# ============================================
# 
# USAGE:
#   1. Make this script executable:
#      chmod +x run-docker.sh
#
#   2. Ensure you have a .env file with your API keys:
#      cp .env.example .env
#      # Edit .env and add your keys
#
#   3. Run commands:
#      ./run-docker.sh build    - Build the Docker image
#      ./run-docker.sh start    - Start the application
#      ./run-docker.sh stop     - Stop the application
#      ./run-docker.sh logs     - View application logs
#      ./run-docker.sh restart  - Restart the application
#      ./run-docker.sh status   - Check container status
#
# FIRST TIME SETUP:
#   ./run-docker.sh build
#   ./run-docker.sh start
#
# ACCESS APPLICATION:
#   Backend API: http://localhost:8000/docs
#   Frontend UI: http://localhost:8501
# ============================================

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "WARNING: .env file not found. Please create one from .env.example"
    echo "Copy .env.example to .env and add your API keys"
    exit 1
fi

# Function to build Docker image
build() {
    echo "Building Docker image..."
    docker build -t car-manual-rag .
    if [ $? -eq 0 ]; then
        echo "Build complete"
    else
        echo "Build failed"
        exit 1
    fi
}

# Function to start the application container
start() {
    echo "Starting application container..."
    
    # Build image if it doesn't exist
    if ! docker image inspect car-manual-rag >/dev/null 2>&1; then
        echo "Image not found. Building first..."
        build
    fi
    
    # Stop and remove existing container if running
    docker stop rag-app 2>/dev/null || true
    docker rm rag-app 2>/dev/null || true
    
    # Run the container
    docker run -d \
        --name rag-app \
        -p 8000:8000 \
        -p 8501:8501 \
        -e WEAVIATE_URL="${WEAVIATE_URL}" \
        -e WEAVIATE_API_KEY="${WEAVIATE_API_KEY}" \
        -e GROQ_API_KEY="${GROQ_API_KEY}" \
        car-manual-rag
    
    if [ $? -eq 0 ]; then
        echo "Container started successfully"
        echo ""
        echo "Access the application at:"
        echo "  Backend API: http://localhost:8000/docs"
        echo "  Frontend UI: http://localhost:8501"
        echo " Wait for few minutes to warm-up the backend then open the frontend ui using the above link"
        echo ""
        echo "To view logs: ./run-docker.sh logs"
    else
        echo "Failed to start container"
        exit 1
    fi
}

# Function to stop the application container
stop() {
    echo "Stopping application container..."
    docker stop rag-app 2>/dev/null || true
    docker rm rag-app 2>/dev/null || true
    echo "Container stopped and removed"
}

# Function to view container logs
logs() {
    echo "Viewing container logs (Press Ctrl+C to exit)..."
    docker logs -f rag-app
}

# Function to restart the application
restart() {
    echo "Restarting application..."
    stop
    sleep 2
    start
}

# Function to show container status
status() {
    echo "Container status:"
    docker ps -a | grep -E "CONTAINER ID|rag-app"
}

# Main command handler
case "$1" in
    build)
        build
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs
        ;;
    status)
        status
        ;;
    *)
        echo "Car Manual RAG System - Docker Run Script"
        echo ""
        echo "Usage: $0 {build|start|stop|restart|logs|status}"
        echo ""
        echo "Commands:"
        echo "  build   - Build the Docker image"
        echo "  start   - Start the application container"
        echo "  stop    - Stop and remove the container"
        echo "  restart - Restart the application"
        echo "  logs    - View container logs (real-time)"
        echo "  status  - Show container status"
        echo ""
        echo "First time setup:"
        echo "  1. cp .env.example .env"
        echo "  2. Edit .env and add your API keys"
        echo "  3. ./run-docker.sh build"
        echo "  4. ./run-docker.sh start"
        echo ""
        exit 1
        ;;
esac