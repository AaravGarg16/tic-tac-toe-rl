import random
class Board:
    def __init__(self):
        self.board = [0 for _ in range(9)]
    
    def clear(self):
        self.board = [0 for _ in range(9)]
    
    def is_full(self)->bool:
        for i in range(len(self.board)):
            if self.board[i] == 0:
                return False
        return True 
    
    def values_match(self, idx1, idx2, idx3):
        if self.board[idx1] == self.board[idx2] and self.board[idx2] == self.board[idx3]:
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
        if self.values_match(0, 4, 8) or self.values_match(2, 4, 6):
            return self.board[4]
        return None

    def winner(self) -> int:
        #1 represents player 1 win, -1 represents 
        winner = self.horizontal_match() or self.vertical_match() or self.diagonal_match()
        if winner:
            return winner
        elif self.is_full:
            return 0 #represents draw
        else:
            return None #represents the game is not finished
    
    def __repr__(self):
        return "".join(self.board)

#The class is intended to represent 1 game of Tic-Tac-Toe
class TicTacToeGame:
    #all game history will be a class attribute shared between all instances of the class 
    all_game_history = {}
    def __init__(self, board:Board, epsilon, learning_rate=15, player=1):
        self.board = Board
        self.lr = learning_rate
        self.epsilon = epsilon
        self.player = player 
        self.history = []
    
    def update_Q(cls, self, winner):
        if winner == self.player:
            self_reward, other_reward = 1, -1
        elif winner == -self.player:
            self_reward, other_reward = -1, 1
        else: #marks a draw
            return 
        for i in range(self.history):
            state, move, player = self.history[i]
            reward = self_reward if player == self.player else other_reward
            cls.all_game_history((state, move)) = cls.all_game_history((state, move)) + self.lr * (reward - cls.all_game_history((state, move)))
        
    def valid_moves(self):
        moves = []
        for i in range(len(self.board)):
            if self.board[i] == 0:
                moves.append(i)
        return moves 
    
    def is_valid_move(self, idx):
        if self.board[idx] == 0:
            return True 
        else:
            return False

    def store_user_move(cls, self, idx=-1):
        if idx == -1:
            idx = random.choice(self.valid_moves())
        elif not self.is_valid_move(idx):
            raise Exception("Invalid move allowed - error in front end")
        if (str(self.board), idx) not in cls.all_game_history:
            cls.all_game_history[(str(self.board), idx)] = 0
        self.history.append((str(self.board), idx, -self.player))
        self.board[idx] = -self.player 
        #check if a winner is decided 

    def make_move(cls, self):
        valid_moves = self.valid_moves()
        found = False
        # Check if current board state exists in Q-table, arranged in descending order 
        for (state, move) in cls.all_game_history:
            if state == str(self.board):
                idx = move
                break
        # Epsilon-greedy: explore with probability self.epsilon
        if random.random() < self.epsilon or not found:
            idx = random.choice(valid_moves)
            if not found:
                cls.all_game_history[(str(self.board), idx)] = 0

        self.history.append((str(self.board), idx, self.player))
        self.board[idx] = self.player 



# -------------------------
# 3. Training loop
# -------------------------

# Function: train
# - repeat for many games (e.g. 50,000 times)
# - initialize an empty board
# - keep a history list of (state, move, player) for this game
# - alternate between players making moves until the game ends
# - when game ends, call update_Q to adjust the Q-values


# -------------------------
# 4. Play against human
# -------------------------

# Function: play_human
# - start an empty board
# - loop:
#   - ask human for a move, update board
#   - check if game ended
#   - bot chooses move using Q (epsilon=0 so it always picks best known move)
#   - update board
#   - check if game ended
# - print board after each move


# -------------------------
# 5. Run the program
# -------------------------

# if __name__ == "__main__":
# - call train() first so the bot learns
# - then call play_human() so you can test against it