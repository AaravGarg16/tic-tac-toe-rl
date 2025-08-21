import random

class Board:
    def __init__(self):
        self.board = [0 for _ in range(9)]
    
    def clear(self):
        self.board = [0 for _ in range(9)]
    
    def is_full(self) -> bool:
        for i in range(len(self.board)):
            if self.board[i] == 0:
                return False
        return True 
    
    # CHECK FOR PATTERN ------ 
    def values_match(self, idx1, idx2, idx3):
        if self.board[idx1] == self.board[idx2] and self.board[idx2] == self.board[idx3] and self.board[idx1] != 0:
            return True
        return False

    def horizontal_match(self):
        for i in range(0, 7, 3):
            if self.values_match(i, i+1, i+2):
                return self.board[i]
        return None

    def vertical_match(self):
        for i in range(3):
            if self.values_match(i, i+3, i+6):
                return self.board[i]
        return None
    
    def diagonal_match(self):
        # Fixed: ensure we check each diagonal separately
        if self.values_match(0, 4, 8):
            return self.board[4]
        elif self.values_match(2, 4, 6):
            return self.board[4]
        return None
    
    # ----------------------
    def __str__(self):
        # Fixed: convert int to str to avoid TypeError in "".join
        return "".join(str(x) for x in self.board)


# The class is intended to represent 1 game of Tic-Tac-Toe
class TicTacToeGame:
    # all game history will be a class attribute shared between all instances of the class 
    all_game_history = {}
    
    def __init__(self, board: Board, epsilon=0.1, learning_rate=0.15, player=1):
        self.board = board
        self.lr = learning_rate
        self.epsilon = epsilon
        self.player = player # 1 represents first/starting player, -1 represents second player 
        self.history = []

    def valid_moves(self):
        moves = []
        for i in range(len(self.board.board)):  # Fixed: use self.board.board
            if self.board.board[i] == 0:
                moves.append(i)
        return moves 
    
    def is_valid_move(self, idx):
        return self.board.board[idx] == 0  # Simplified

    def store_user_move(self, idx=-1):
        if not self.valid_moves():
            return
        elif idx == -1:
            idx = random.choice(self.valid_moves())
        elif not self.is_valid_move(idx):
            raise Exception("Invalid move allowed - error in front end")
        if (str(self.board), idx) not in self.__class__.all_game_history:
            self.__class__.all_game_history[(str(self.board), idx)] = 0
        self.history.append((str(self.board), idx, -self.player))
        self.board.board[idx] = -self.player 
        # Winner check happens outside

    def make_move(self):
        if not self.valid_moves():
            return
        found = False
        idx = -1

        # Check if current board state exists in Q-table
        for (state, move) in self.__class__.all_game_history:
            if state == str(self.board):
                idx = move
                found = True
                break

        # Epsilon-greedy: explore with probability self.epsilon
        if random.random() < self.epsilon or not found:
            idx = random.choice(self.valid_moves())
            if not found:
                self.__class__.all_game_history[(str(self.board), idx)] = 0

        self.history.append((str(self.board), idx, self.player))
        self.board.board[idx] = self.player  # current player (1 or -1)

    def winner(self) -> int:
        # 1 represents player 1 win, -1 represents player 2 win, 0 = draw
        winner = self.board.horizontal_match() or self.board.vertical_match() or self.board.diagonal_match()
        if winner:
            return winner
        elif self.board.is_full():  # Fixed: call method
            return 0  # represents draw
        else:
            return None  # represents the game is not finished
    
    def update_Q(self, winner):
        if winner == self.player:
            self_reward, other_reward = 1, -1
        elif winner == -self.player:
            self_reward, other_reward = -1, 1
        else:  # marks a draw
            self_reward = other_reward = 0

        for i in range(len(self.history)):  # Fixed: use len(self.history)
            state, move, player = self.history[i]
            reward = self_reward if player == self.player else other_reward
            # Fixed: dictionary indexing uses []
            self.__class__.all_game_history[(state, move)] = \
                self.__class__.all_game_history[(state, move)] + self.lr * \
                (reward - self.__class__.all_game_history[(state, move)])


# -------------------------
# 3. Training loop example
# -------------------------
for i in range(5):
    game = TicTacToeGame(Board(), epsilon=0.1)  # Fixed: provide board and epsilon
    while game.winner() is None:  # Fixed: call winner()
        game.store_user_move()
        game.make_move()
    game.update_Q(game.winner())  # Fixed: call winner() and pass correctly
print(TicTacToeGame.all_game_history)
