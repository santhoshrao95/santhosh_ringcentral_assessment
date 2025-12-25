# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for data
RUN mkdir -p /app/data

# Expose ports for backend and frontend
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
uvicorn backend:app --host 0.0.0.0 --port 8000 &\n\
streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run both services
CMD ["/app/start.sh"]