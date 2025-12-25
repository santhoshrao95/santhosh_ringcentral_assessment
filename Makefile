setup:
	python3.10 -m venv test_venv
	. test_venv/bin/activate && pip install -r requirements.txt

backend:
	. test_venv/bin/activate && uvicorn backend:app --host 0.0.0.0 --port 8000 --reload

frontend:
	. test_venv/bin/activate && streamlit run frontend.py
