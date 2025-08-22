import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from q_learning import Board  

# -------------------------------
# Neural Network for Tic-Tac-Toe
# -------------------------------
class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        # input: 9 (board) -> hidden: 12 -> output: 9 (Q-values per move)
        self.fc1 = nn.Linear(9, 12)  
        self.fc2 = nn.Linear(12, 9)  

    # mask illegal moves by setting Q-values to -inf
    def mask(self, tensor, board:Board):
        tensor = tensor.clone()  # avoid modifying original tensor
        for i in range(9):
            if not board.is_valid_move(i):
                tensor[0, i] = -float('inf')
        return tensor

    def forward(self, board:Board):
        state_tensor = torch.tensor(board.board, dtype=torch.float).unsqueeze(0)  # shape [1,9]
        x = torch.relu(self.fc1(state_tensor))
        x = self.fc2(x)
        x = self.mask(x, board)  
        return x  


# -------------------------------
# DQN wrapper / training logic
# -------------------------------
class TicTacToeDQ():
    def __init__(self, model:BasicNN, board:Board, player:int=1):
        self.model = model
        self.board = board
        self.player = player

    # calculate target Q-value with Bellman equation
    def target(self, next_board:Board, gamma:float):
        imm_reward = self.board.reward()
        if next_board.game_over():
            max_next_Q = 0
        else:
            with torch.no_grad():  # don't track gradients for next state
                max_next_Q = self.model(next_board).max().item()
        return imm_reward + gamma * max_next_Q

    def train_step(self, action:int, next_board:Board, gamma:float=0.9):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # current Q-values
        state_tensor = torch.tensor(self.board.board, dtype=torch.float).unsqueeze(0)
        pred_Q = self.model(self.board)
        pred_Q_a = pred_Q[0, action]

        # compute target
        target_Q = torch.tensor(self.target(next_board, gamma), dtype=torch.float)

        # compute loss
        loss = criterion(pred_Q_a, target_Q)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # user move: plays opposite to DQN
    def user_move(self, idx:int=-1):
        if not self.board.valid_moves():  
            return
        elif idx == -1:
            idx = random.choice(self.board.valid_moves())  
        elif not self.board.is_valid_move(idx):  
            raise Exception("Invalid move allowed - error in front end")
        self.board.board[idx] = -self.player 

    # select DQN move with epsilon-greedy policy
    def select_move(self, epsilon:float=0.1):
        if random.random() < epsilon:
            return random.choice(self.board.valid_moves())
        else:
            q_values = self.model(self.board)
            move = torch.argmax(q_values).item()
            return move


# -------------------------------
#  Skeleton left
# -------------------------------
# - Implement training loop:
#     1. Initialize Board and DQN model
#     2. Loop until game_over:
#           a) DQN selects move
#           b) Apply move
#           c) Observe reward and next state
#           d) train_step with (state, action, next_state, reward)
#           e) Opponent/user move
# - Add replay buffer if desired (for batch training)
# - Implement epsilon decay over episodes
# - Logging / printing Q-values and win rates

