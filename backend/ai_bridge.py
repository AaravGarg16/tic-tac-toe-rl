import ast
import importlib.util
import sys
import types
from typing import List, Optional

_MODS = {}

def _find_spec(module_name: str):
    try:
        return importlib.util.find_spec(module_name)
    except Exception:
        return None

def _sanitize_source(src: str):
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            keep.append(node)
    new_tree = ast.Module(body=keep, type_ignores=[])
    return compile(new_tree, filename="<sanitized>", mode="exec")

def _load_sanitized(module_name: str) -> types.ModuleType:
    if module_name in _MODS:
        return _MODS[module_name]

    spec = _find_spec(module_name)
    if not spec or not spec.origin:
        raise RuntimeError(f"Cannot find {module_name}")

    with open(spec.origin, "r", encoding="utf-8") as f:
        src = f.read()

    code = _sanitize_source(src)
    mod = types.ModuleType(module_name)
    mod.__file__ = spec.origin
    if module_name.endswith(".q_learning"):
        sys.modules["q_learning"] = mod

    if module_name.endswith(".dql") and "q_learning" not in sys.modules:
        _load_sanitized(module_name.rsplit(".", 1)[0] + ".q_learning")

    sys.modules[module_name] = mod
    exec(code, mod.__dict__)
    _MODS[module_name] = mod
    return mod

def _other(mark: str) -> str:
    return "O" if mark.upper() == "X" else "X"

def _to_numeric(board: List[str], ai_mark: str) -> List[int]:
    ai = ai_mark.upper()
    opp = _other(ai)
    out: List[int] = []
    for v in board:
        if v == ai:
            out.append(1)
        elif v == opp:
            out.append(-1)
        else:
            out.append(0)
    return out

def _legal_actions(board: List[str]) -> List[int]:
    return [i for i in range(9) if board[i] == ""]

# ----- Easy (tabular Q-learning) -----
def _easy_move(board_str: List[str], ai_mark: str) -> int:
    ql = _load_sanitized("rl_agents.q_learning")
    Board = getattr(ql, "Board")
    Game = getattr(ql, "TicTacToeGame")

    numeric = _to_numeric(board_str, ai_mark)
    b = Board()
    b.board = numeric[:]
    player_num = 1 if ai_mark.upper() == "X" else -1

    g = Game(b, epsilon=0.0, player=player_num)  # greedy
    before = b.board[:]
    g.make_move()  # applies AI move into b.board

    # detect which cell changed to player's marker
    for i in range(9):
        if before[i] == 0 and b.board[i] == player_num:
            return i
    raise RuntimeError("Q-learning did not change the board")

# ----- Hard -----
def _hard_move(board_str: List[str], ai_mark: str) -> int:
    ql = _load_sanitized("rl_agents.q_learning")
    dql = _load_sanitized("rl_agents.dql")
    Board = getattr(ql, "Board")
    BasicNN = getattr(dql, "BasicNN")
    DQ = getattr(dql, "TicTacToeDQ")

    numeric = _to_numeric(board_str, ai_mark)
    b = Board()
    b.board = numeric[:]
    player_num = 1 if ai_mark.upper() == "X" else -1

    model = BasicNN()
    game = DQ(model, b, player=player_num)
    pos = game.select_move(epsilon=0.0)  # greedy
    if not isinstance(pos, int):
        raise RuntimeError(f"DQL returned non-int: {pos}")
    return pos

# ----- Public entry -----
def agent_move(board: List[str], ai_mark: str, difficulty: str) -> int:
    if len(board) != 9:
        raise RuntimeError("Board must have 9 cells")
    if ai_mark.upper() not in ("X", "O"):
        raise RuntimeError("ai_mark must be 'X' or 'O'")

    difficulty = (difficulty or "hard").lower()
    pos = _easy_move(board, ai_mark) if difficulty == "easy" else _hard_move(board, ai_mark)
    legal = _legal_actions(board)
    if pos not in legal:
        if not legal:
            raise RuntimeError("No legal moves")
        pos = legal[0]
    return pos
