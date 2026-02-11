from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio
import sys
import os
import time
import shutil
from io import BytesIO, StringIO
from datetime import datetime
from pathlib import Path

import requests
import chess
import chess.pgn
import chess.engine
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from tensorflow import keras


APP_NAME = "Chess Personal Coach API"
APP_VERSION = "0.1.0"

# Ensure subprocess support on Windows (Stockfish engine)
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path(__file__).resolve().parent / "models" / "difficulty_model_final.keras"),
)
DEFAULT_STOCKFISH_PATH = (
    os.getenv("STOCKFISH_PATH")
    or shutil.which("stockfish")
    or str(Path(__file__).resolve().parent / "stockfish" / "stockfish")
)

LICHESS_API = "https://lichess.org/api/games/user"
REPORT_DIR = os.path.join(os.getcwd(), "reports")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def fen_to_tensor(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 14), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        base_channel = PIECE_TO_CHANNEL[piece.piece_type]
        channel = base_channel if piece.color == chess.WHITE else base_channel + 6
        tensor[row, col, channel] = 1.0

    stm_value = 1.0 if board.turn == chess.WHITE else 0.0
    tensor[:, :, 12] = stm_value

    castling_count = sum(
        [
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ]
    )
    tensor[:, :, 13] = float(castling_count)

    if board.turn == chess.BLACK:
        tensor = np.flip(tensor, axis=0)
        tensor[:, :, 0:6], tensor[:, :, 6:12] = (
            tensor[:, :, 6:12].copy(),
            tensor[:, :, 0:6].copy(),
        )
        tensor[:, :, 12] = 1.0

    return tensor


def phase_from_move_number(move_number: int) -> str:
    if move_number <= 15:
        return "Opening"
    if move_number <= 40:
        return "Middlegame"
    return "Endgame"


def load_model() -> keras.Model:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return keras.models.load_model(MODEL_PATH)


def resolve_stockfish_path() -> str:
    if os.path.exists(DEFAULT_STOCKFISH_PATH):
        return DEFAULT_STOCKFISH_PATH
    raise FileNotFoundError(
        "Stockfish engine not found. Set STOCKFISH_PATH or place engine at "
        f"{DEFAULT_STOCKFISH_PATH}"
    )


def parse_pgn_games(pgn_text: str, max_games: int) -> List[chess.pgn.Game]:
    games = []
    pgn_io = StringIO(pgn_text)
    while len(games) < max_games:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        games.append(game)
    return games


def fetch_lichess_pgn(username: str, max_games: int, mode: str) -> str:
    params = {
        "max": max_games,
        "pgnInJson": "false",
        "moves": "true",
    }
    if mode != "standard":
        params["perfType"] = mode
    url = f"{LICHESS_API}/{username}"
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=(
                "Lichess API error. If your games are private, open Lichess "
                "Settings → Privacy and enable game visibility and downloads. "
                "Also ensure the username is correct and the account is public."
            ),
        )
    return resp.text


def evaluate_cpl(
    engine: chess.engine.SimpleEngine, board: chess.Board, move: chess.Move, depth: int
) -> float:
    info_best = engine.analyse(board, chess.engine.Limit(depth=depth))
    best_score = info_best["score"].pov(board.turn).score(mate_score=10000)
    if best_score is None:
        best_score = 0

    board_after = board.copy()
    board_after.push(move)
    info_played = engine.analyse(board_after, chess.engine.Limit(depth=depth))
    played_score = info_played["score"].pov(board.turn).score(mate_score=10000)
    if played_score is None:
        played_score = 0

    return float(max(0, best_score - played_score))


def generate_pdf_report(
    summary: Dict[str, Any], phase_stats: List[Dict[str, Any]], output_path: str
) -> str:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 50, "Chess Personal Coach Report")

    c.setFont("Helvetica", 10)
    c.drawString(40, height - 70, f"Generated: {datetime.now().isoformat(timespec='minutes')}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height - 100, "Summary")
    c.setFont("Helvetica", 10)
    y = height - 120
    for line in summary["summary"].split(". "):
        c.drawString(50, y, line.strip())
        y -= 14

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y - 10, "Phase Breakdown")
    y -= 30
    c.setFont("Helvetica", 10)
    for phase in phase_stats:
        c.drawString(
            50,
            y,
            f"{phase['phase']}: avg CPL {phase['avg_cpl']:.1f}, mistake rate {phase['mistake_rate']:.1%}",
        )
        y -= 14

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y - 10, "Recommendations")
    y -= 30
    c.setFont("Helvetica", 10)
    for rec in summary["recommendations"]:
        c.drawString(50, y, f"- {rec}")
        y -= 14

    c.showPage()
    c.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    return output_path


app = FastAPI(title=APP_NAME, version=APP_VERSION)

# Comma-separated list. Example:
# CORS_ORIGINS=https://your-frontend.onrender.com,http://localhost:8080
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env.strip():
    cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
else:
    cors_origins = ["http://localhost:8080", "http://127.0.0.1:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=40)
    games: int = Field(10, ge=1, le=50)
    include_pgn: bool = False
    mode: str = Field("standard", pattern="^(standard|rapid|blitz|bullet)$")
    max_moves: int = Field(400, ge=50, le=2000)
    stockfish_depth: int = Field(10, ge=8, le=16)
    pgn_text: Optional[str] = None


class PhaseStats(BaseModel):
    phase: str
    accuracy: float
    avg_cpl: float
    mistake_rate: float


class MoveDetail(BaseModel):
    ply: int
    san: str
    uci: str
    fen_before: str
    fen_after: str
    cpl: float
    difficulty: float
    phase: str
    suggestion: str


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    username: str
    games_analyzed: int
    difficulty_score: float
    weakest_phase: str
    summary: str
    phase_breakdown: List[PhaseStats]
    top_patterns: List[str]
    recommendations: List[str]
    moves: List[MoveDetail]
    artifacts: Dict[str, Optional[str]]
    model_path: str
    stockfish_path: str
    report_path: Optional[str]
    report_url: Optional[str]


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "name": APP_NAME,
        "version": APP_VERSION,
        "model_path": MODEL_PATH,
        "stockfish_path": DEFAULT_STOCKFISH_PATH,
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if not req.username and not req.pgn_text:
        raise HTTPException(status_code=400, detail="username or pgn_text required")
    try:
        model = load_model()
        stockfish_path = resolve_stockfish_path()
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        if req.pgn_text:
            pgn_text = req.pgn_text
        else:
            pgn_text = fetch_lichess_pgn(req.username, req.games, req.mode)

        games = parse_pgn_games(pgn_text, req.games)
        if not games:
            raise HTTPException(status_code=400, detail="No games found")

        phase_stats: Dict[str, Dict[str, Any]] = {
            "Opening": {"cpl": [], "mistakes": 0, "moves": 0},
            "Middlegame": {"cpl": [], "mistakes": 0, "moves": 0},
            "Endgame": {"cpl": [], "mistakes": 0, "moves": 0},
        }

        difficulty_scores: List[float] = []
        top_patterns = set()
        move_details: List[MoveDetail] = []

        moves_processed = 0
        for game in games:
            board = game.board()
            for ply, move in enumerate(game.mainline_moves(), start=1):
                if moves_processed >= req.max_moves:
                    break

                fen_before = board.fen()
                cpl = evaluate_cpl(engine, board, move, req.stockfish_depth)
                phase = phase_from_move_number((ply + 1) // 2)

                tensor = fen_to_tensor(board.fen())
                pred = model.predict(tensor[None, ...], verbose=0)[0][0]
                difficulty_scores.append(float(pred))

                phase_stats[phase]["cpl"].append(cpl)
                phase_stats[phase]["moves"] += 1
                if cpl > 50:
                    phase_stats[phase]["mistakes"] += 1

                if pred > 0.7:
                    top_patterns.add("High tactical complexity detected")
                if cpl > 100:
                    top_patterns.add("Blunder under pressure")

                san = board.san(move)
                uci = move.uci()
                board.push(move)
                fen_after = board.fen()

                suggestion = "Solid move."
                if cpl > 100:
                    suggestion = "Blunder: review tactics and threats."
                elif cpl > 50:
                    suggestion = "Inaccuracy: check candidate moves."
                elif pred > 0.7:
                    suggestion = "Complex position: slow down and calculate."

                move_details.append(
                    MoveDetail(
                        ply=ply,
                        san=san,
                        uci=uci,
                        fen_before=fen_before,
                        fen_after=fen_after,
                        cpl=cpl,
                        difficulty=float(pred),
                        phase=phase,
                        suggestion=suggestion,
                    )
                )
                moves_processed += 1
            if moves_processed >= req.max_moves:
                break

        phase_breakdown = []
        weakest_phase = "Opening"
        worst_score = -1.0
        for phase, stats in phase_stats.items():
            if stats["moves"] == 0:
                continue
            avg_cpl = float(np.mean(stats["cpl"])) if stats["cpl"] else 0.0
            mistake_rate = stats["mistakes"] / stats["moves"]
            accuracy = max(0.0, 1.0 - avg_cpl / 200.0)
            phase_breakdown.append(
                PhaseStats(
                    phase=phase,
                    accuracy=accuracy,
                    avg_cpl=avg_cpl,
                    mistake_rate=mistake_rate,
                )
            )
            if mistake_rate > worst_score:
                worst_score = mistake_rate
                weakest_phase = phase

        difficulty_score = float(np.mean(difficulty_scores)) if difficulty_scores else 0.0

        recommendations = [
            f"Focus training on {weakest_phase.lower()} positions.",
            "Review games with CPL spikes above 100.",
            "Drill tactical puzzles around your most frequent motifs.",
        ]

        summary_text = (
            f"Analyzed {moves_processed} moves. Your weakest phase is {weakest_phase}. "
            "CPL spikes align with high difficulty predictions, suggesting tactical "
            "complexity is the main source of errors."
        )

        os.makedirs(REPORT_DIR, exist_ok=True)
        report_name = f"report_{req.username or 'pgn'}_{int(time.time())}.pdf"
        report_path = os.path.join(REPORT_DIR, report_name)
        generate_pdf_report(
            summary={
                "summary": summary_text,
                "recommendations": recommendations,
            },
            phase_stats=[p.model_dump() for p in phase_breakdown],
            output_path=report_path,
        )

        return AnalyzeResponse(
            status="ok",
            username=req.username or "pgn_upload",
            games_analyzed=min(req.games, len(games)),
            difficulty_score=difficulty_score,
            weakest_phase=weakest_phase,
            summary=summary_text,
            phase_breakdown=phase_breakdown,
            top_patterns=sorted(list(top_patterns)) or ["No dominant pattern detected"],
            recommendations=recommendations,
            moves=move_details,
            artifacts={
                "report_pdf": report_path,
                "report_docx": None,
                "charts": None,
            },
            model_path=MODEL_PATH,
            stockfish_path=stockfish_path,
            report_path=report_path,
            report_url=f"/api/report/{report_name}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            engine.quit()
        except Exception:
            pass


@app.post("/api/analyze_pgn", response_model=AnalyzeResponse)
def analyze_pgn(file: UploadFile = File(...)) -> AnalyzeResponse:
    pgn_text = file.file.read().decode("utf-8", errors="ignore")
    req = AnalyzeRequest(
        username="pgn_upload",
        games=10,
        include_pgn=True,
        mode="standard",
        pgn_text=pgn_text,
    )
    return analyze(req)


@app.get("/api/report/{report_name}")
def get_report(report_name: str):
    report_path = os.path.join(REPORT_DIR, report_name)
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path, media_type="application/pdf", filename=report_name)


@app.get("/", include_in_schema=False)
def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Chess Personal Coach API is running."}


if FRONTEND_DIR.exists():
    # Serve frontend assets and index on same host in production (easier deploy/CORS).
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
