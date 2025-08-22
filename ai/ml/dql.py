"""
*Q-value* is the expected total reward you’ll get if you take a particular action 
in a given state and then follow your strategy afterwards.

Input: state of the board [1, 0, 0 ... -1] (9 values, 1 dimensional) 
Output: Q-value per position, unavailable positions masked

Input (9) → Hidden (32, ReLU) → Output (9, raw Q-values)

Relu Activation Function: f(x)=max(0,x) | non-linear activation + vanishing gradient problem {pass gradient if positive --> larger derivatives --> learning does not stop at the earlier layers}
MSE loss = (predicted_Q - target_Q)^2 

**** BELLMAN EQUATION FOR ***
target=immediate reward + (discount factor * best possible future reward) 
target = r + gamma * max(next_Q) 

ε-greedy policy:
probability ε → pick random move (exploration)
probability 1-ε → pick best predicted move (argmax Q(s, :))

"""

import torch
import torch.nn as nn  #weight and bias tensors as a part of the network
import torch.nn.functional as F #activation function
from torch.optim import SGD 
from q_learning import Board

class BasicNN(nn.Module):
    def __init__(self, board:Board):
        super().__init__()
        self.board = board
        # first layer: 2 inputs -> 12 hidden units
        self.fc1 = nn.Linear(9, 12)  
        # second layer: 12 hidden -> 9 output
        self.fc2 = nn.Linear(12, 9)  

    def current_reward(self) -> int:
        winner = self.board.horizontal_match() or self.board.vertical_match() or self.board.diagonal_match()
        if winner:
            return winner
        else:   
            return 0  # draw
    
    def forward(self, x):
        x1 = torch.relu(self.fc1(x))   # apply weights+biases from fc1 with relu activation
        x2 = self.fc2(x)               # apply weights+biases from fc2
        return x2
    
    def target(self, gamma):
        imm_reward = self.game(state)
        reward = self.forward()

    def backwards(self, model, inputs, labels):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

                
