setup:
	python3.10 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

backend:
	. venv/bin/activate && uvicorn backend:app --host 0.0.0.0 --port 8000 --reload

frontend:
	. venv/bin/activate && streamlit run frontend.py
