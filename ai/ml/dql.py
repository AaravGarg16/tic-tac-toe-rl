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

class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # first layer: 2 inputs -> 12 hidden units
        self.fc1 = nn.Linear(9, 12)  

        # second layer: 12 hidden -> 1 output
        self.fc2 = nn.Linear(12, 9)  

    
    def forward(self, x):
        x = torch.relu(self.fc1(x))   # apply weights+biases from fc1
        x = self.fc2(x)               # apply weights+biases from fc2
        output = F.relu(x)
        return output
    
    def backwards(self, model, inputs, labels):
        optimizer = SGD(model.parameters(), lr=0.1) 
        for epoch in range(100):
    
            total_loss = 0 
            
            for iteration in range(len(inputs)):
                
                input_i = inputs[iteration]
                label_i = labels[iteration]
                
                output_i = model(input_i)
                
                loss = (output_i - label_i)**2
                
                loss.backward() #calculates the derivative with respect to the parameters we want to optimize
                #accumulates the derivatives each time we go thru the forward loop, for all data points 
                
                total_loss += float(loss)
                
                if (total_loss < 0.0001):
                    print("Num steps: " + str(epoch))
                    break

                #has access to the derivatives from the loss function
                #can step into the direction of decreasing the loss 
                
                optimizer.step()
                optimizer.zero_grad() #set l