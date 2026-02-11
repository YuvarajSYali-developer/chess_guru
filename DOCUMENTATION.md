# Chess Personal Coach — Final Documentation

## Overview
This app analyzes Lichess games or uploaded PGN files to calculate:
- **Stockfish CPL** for each move (ground truth)
- **CNN Difficulty** for each position (model inference)
- **Phase breakdown** (opening/middlegame/endgame)
- **Personalized recommendations**

Outputs:
- Web dashboard with move list + interactive board
- PDF report

## Architecture
- **Frontend**: Static HTML/CSS/JS (`frontend/`)
- **Backend**: FastAPI (`backend/app.py`)
- **Model**: `difficulty_model_final.keras`
- **Engine**: Stockfish `.exe`

## How It Works
1. Fetch games from Lichess (or upload PGN)
2. For each move:
   - Stockfish evaluates best move vs played move → CPL
   - CNN predicts difficulty from board tensor
3. Aggregate by phase and generate insights
4. Return JSON + PDF report

## Run Locally
Install deps:
```
pip install -r backend/requirements.txt
```

Start backend:
```
python -m uvicorn backend.app:app --reload --port 8000
```

Start frontend:
```
python -m http.server 8080 --directory frontend
```

Open: `http://localhost:8080`

## One-Click Start
Windows:
```
run_app.bat
```

macOS/Linux:
```
chmod +x run_app.sh
./run_app.sh
```

## API Endpoints
`POST /api/analyze`
```json
{
  "username": "lichess_user",
  "games": 5,
  "mode": "blitz",
  "max_moves": 400,
  "stockfish_depth": 10
}
```

`POST /api/analyze_pgn`
- Multipart form file upload (`.pgn`)

## Lichess Errors (Privacy)
If Lichess API fails:
1. Open Lichess settings → **Privacy**
2. Enable **game visibility** and **download access**
3. Ensure the username is correct and account is public

## Report Output
Generated PDF path is returned in the API response.

## Model Swap
Replace the model file and update `MODEL_PATH` in `backend/app.py`.

## Deployment Notes
- Backend: FastAPI (Render/Railway/Fly.io free tiers)
- Frontend: static hosting (Netlify/GitHub Pages)
- If deploying separately, set CORS and API base URL in the frontend.
