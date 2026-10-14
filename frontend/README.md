# Fieldnotes — React interface

This optional React/Vite interface sends feature values to the local FastAPI adapter, which loads the trained scikit-learn pipeline saved by the existing training script.

## Setup

From the repository root:

~~~bash
python -m pip install -r requirements-react-api.txt
~~~

If no trained model exists yet, train one from the repository root:

~~~bash
python src/train.py
~~~

Start the API in terminal 1:

~~~bash
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
~~~

In terminal 2:

~~~bash
cd frontend
npm install
npm run dev
~~~

Open the local URL printed by Vite, normally http://127.0.0.1:5173/.

## Scope

The API uses a real locally saved model when available. It does not fabricate predictions if the model file is missing. This is an educational classification workflow on a small, population-specific dataset, not a validated clinical risk calculator. Model probability is not the same as a calibrated individual clinical risk.
