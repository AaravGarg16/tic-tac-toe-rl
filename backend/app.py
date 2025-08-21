from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

games = {}

class Move(BaseModel):
    game_id: str
    position: int

def lines():
    return [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]

def check_winner(board: List[str]) -> str:
    for a, b, c in lines():
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    if "" not in board:
        return "draw"
    return ""

def legal_actions(board: List[str]) -> List[int]:
    actions = []
    for i in range(len(board)):
        if board[i] == "":
            actions.append(i)
    return actions

def agent_move_stub(board: List[str]) -> Optional[int]:
    # code for reinforcement learning algorithim will come here

@app.post("/new")
def new_game():
    gid = str(len(games) + 1)
    games[gid] = [""] * 9
    return {"game_id": gid, "board": games[gid], "status": "in_progress"}

@app.post("/move")
def player_move(m: Move):
    board = games[m.game_id]
    if 0 <= m.position <= 8 and board[m.position] == "":
        board[m.position] = "X"
    status = check_winner(board)
    if status:
        return {"board": board, "status": status}
    ai_pos = agent_move_stub(board)
    if ai_pos is not None:
        board[ai_pos] = "O"
    status = check_winner(board)
    return {"board": board, "status": status or "in_progress"}
