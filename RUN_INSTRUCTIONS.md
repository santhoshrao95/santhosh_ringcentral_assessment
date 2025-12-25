# How to Run the Application

## Prerequisites
- Python 3.10 (for menthods 1 and 2)
- Docker (for methods 3 and 4)
- .env file with required api keys. Check the zip file attachment or google doc link in the email.
- the api keys in the .env file will expire post January 10, 2026.
---

## Method 1: Virtual Environment (Two Terminals)

**Terminal 1 - Backend:**
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
source venv/bin/activate
streamlit run frontend.py
```

**Access:**
- Backend: http://localhost:8000/docs
- Frontend: http://localhost:8501

---

## Method 2: Makefile (Two Terminals)

**Setup (First Time):**
```bash
make setup
```

**Terminal 1 - Backend:**
```bash
make backend
```

**Terminal 2 - Frontend:**
```bash
make frontend
```

**Access:**
- Backend: http://localhost:8000/docs
- Frontend: http://localhost:8501

---

## Method 3: Docker Build and Run

**Setup:**
```bash
chmod +x run_docker.sh
```

**Commands:**
```bash
./run_docker.sh build    # Build image
./run_docker.sh start    # Start application
./run_docker.sh logs     # View logs
./run_docker.sh stop     # Stop application
./run_docker.sh status   # Check status
```

**Access:**
- Backend: http://localhost:8000/docs
- Frontend: http://localhost:8501

---

## Method 4: Docker Pull and Run

**Pull Image:**
```bash
docker pull sandyai2022/car-manual-rag:assessment
```

**Run Container:**
```bash
# Load environment variables from .env file
export $(cat .env | grep -v '^#' | xargs)

docker run -d \
  --name rag-app \
  -p 8000:8000 \
  -p 8501:8501 \
  -e WEAVIATE_URL="${WEAVIATE_URL}" \
  -e WEAVIATE_API_KEY="${WEAVIATE_API_KEY}" \
  -e GROQ_API_KEY="${GROQ_API_KEY}" \
  sandyai2022/car-manual-rag:assessment
```

**Manage Container:**
```bash
docker logs -f rag-app      # View logs
docker stop rag-app         # Stop container
docker rm rag-app           # Remove container
```

**Access:**
- Backend: http://localhost:8000/docs
- Frontend: http://localhost:8501

---

## Notes
- .env file with API keys is included in the package which will expire post Jan 10, 2026
- Backend must be running before frontend for full functionality
- Use Ctrl+C to stop processes in terminal methods