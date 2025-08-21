
class Board:
    def __init__(self, player:int=1):
        self.board = [[0 for _ in range(3)] for _ in range(3)]

#The class is intended to represent 1 game of Tic-Tac-Toe
class TicTacToe:
    def __init__(self, board:Board, learning_rate):
        self.board = Board
        self.lr = learning_rate
        self.history = {}
    
    # def reward_function(self, reward):
    #     self.history[(state, move)] = self.history[(state, move)] + self.lr * (reward - self.history[(state, move)] )
        
    def valid_moves(self):
        moves = []
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 0:
                    moves.append((i,j))
        return moves 

            
    # def check_winner(self):



# - return an empty 3x3 board (represented as a list of 9 zeros)

# Function: get_valid_moves
# - return a list of indexes (0-8) that are still empty

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