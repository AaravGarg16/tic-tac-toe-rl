from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from ai_bridge import agent_move

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

games: Dict[str, Dict] = {}

class NewGameRequest(BaseModel):
    player_mark: str
    difficulty: Optional[str] = "hard"  # "easy" | "hard"

class Move(BaseModel):
    game_id: str
    position: int

LINES = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6),
]

def check_winner(board: List[str]) -> Optional[str]:
    for a,b,c in LINES:
        v = board[a]
        if v and v == board[b] and v == board[c]:
            return v
    if all(v != "" for v in board):
        return "draw"
    return None

@app.post("/new")
def new_game(req: NewGameRequest):
    player = req.player_mark.upper()
    if player not in ("X", "O"):
        raise HTTPException(400, "player_mark must be 'X' or 'O'")

    difficulty = (req.difficulty or "hard").lower()
    if difficulty not in ("easy", "hard"):
        raise HTTPException(400, "difficulty must be 'easy' or 'hard'")

    gid = str(len(games) + 1)
    ai = "O" if player == "X" else "X"
    board = [""] * 9
    games[gid] = {"board": board, "player": player, "ai": ai, "difficulty": difficulty}

    if player == "O":
        try:
            pos = agent_move(board, ai, difficulty)
            board[pos] = ai
        except Exception as e:
            raise HTTPException(500, f"AI error ({difficulty}): {e}")

    return {
        "game_id": gid,
        "board": board,
        "status": check_winner(board) or "in_progress",
        "player": player,
        "ai": ai,
        "difficulty": difficulty,
    }

@app.post("/move")
def player_move(m: Move):
    g = games.get(m.game_id)
    if not g:
        raise HTTPException(404, "game not found")

    board: List[str] = g["board"]
    player: str = g["player"]
    ai: str = g["ai"]
    difficulty: str = g["difficulty"]

    if not (0 <= m.position <= 8):
        raise HTTPException(400, "position must be 0..8")
    if board[m.position] != "":
        raise HTTPException(400, "cell already filled")

    board[m.position] = player
    status = check_winner(board)
    if status:
        return {"board": board, "status": status}

    try:
        ai_pos = agent_move(board, ai, difficulty)
        board[ai_pos] = ai
    except Exception as e:
        raise HTTPException(500, f"AI error ({difficulty}): {e}")

    status = check_winner(board)
    return {"board": board, "status": status or "in_progress"}
