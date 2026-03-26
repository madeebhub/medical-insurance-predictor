# Medical App

A simple full-stack medical charges prediction app (ML model backend + Next.js frontend).

## Project Structure

- `backend/`: Python API and ML prediction logic
  - `api.py`: Flask/FastAPI endpoints for model predictions
  - `predict.py`: prediction helper
  - `medical_charges_model.joblib`: trained model
  - `data.csv`: sample training data
- `insurance/`: Next.js frontend
  - `app/page.tsx`: main UI
  - `components/`: UI components

## Setup

### Backend

1. `cd backend`
2. Create/activate virtual environment: `python -m venv .venv` and `source .venv/Scripts/activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt` (or install `fastapi`, `uvicorn`, `scikit-learn`, etc.)
4. Run API server (example): `uvicorn api:app --reload --host 0.0.0.0 --port 8000`

### Frontend

1. `cd insurance`
2. `npm install`
3. `npm run dev`
4. Open `http://localhost:3000`

## Usage

1. Start backend
2. Start frontend
3. Enter input values in UI, submit for predicted medical charges

## Git

- Create commit: `git add README.md && git commit -m "Add project README"`
- Push: `git push`.

## Notes

- Adjust backend URL in frontend if needed.
- Add any missing dependency file and instructions in your repo.