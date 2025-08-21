
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

#The class is intended to represent 1 game of Tic-Tac-Toe
class TicTacToe:
    def __init__(self, board:Board, learning_rate):
        self.board = Board
        self.lr = learning_rate
        self.history_p1 = {}
        self.history_p2 = {}
    
    def reward_function(self, rewardp1, rewardp2):
        for i in range(self.history_p1):
            self.history_p1[i] = self.history_p1[i] + self.lr * (rewardp1 - self.history_p1[i])
        for i in range(self.history_p2):
            self.history_p2[i] = self.history_p2[i] + self.lr * (rewardp2 - self.history[i])
        
    def valid_moves(self):
        moves = []
        for i in range(len(self.board)):
            if self.board[i] == 0:
                moves.append(i)
        return moves 

            
    def check_winner(self):
        return self.board.
        


# Function: make_move
# - place the current player (1 for X, -1 for O) on the board at the chosen index

# Function: check_winner
# - check all rows, columns, diagonals
# - return 1 if X wins, -1 if O wins
# - return 0 if it's a draw (board full, no winner)
# - return None if game not finished


# -------------------------
# 2. Q-table (the notebook)
# -------------------------

# Dictionary: Q
# - keys: (state_string, move)
# - values: how good that move is (start at 0)

# Function: get_state
# - convert board list into a string (so it can be stored in a dictionary key)

# Function: choose_move
# - epsilon-greedy strategy:
#   - with probability epsilon, choose a random valid move
#   - otherwise, choose the move with the highest Q-value
# - if multiple moves have the same Q-value, pick randomly among them


# -------------------------
# 3. Training loop
# -------------------------

# Function: train
# - repeat for many games (e.g. 50,000 times)
# - initialize an empty board
# - keep a history list of (state, move, player) for this game
# - alternate between players making moves until the game ends
# - when game ends, call update_Q to adjust the Q-values

# Function: update_Q
# - input: history (list of moves made) and the game result
# - for each (state, move, player):
#   - assign reward:
#       +1 if that player won
#       -1 if that player lost
#        0 if draw
#   - update the Q-value using:
#       Q[(state, move)] = old_value + learning_rate * (reward - old_value)


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