
"""
This code has been taken from StatQuest by Josh Starner, in order to understand and build intuition for PyTorch and Tensorflow.
"""

import torch
import torch.nn as nn  #weight and bias tensors as a part of the network
import torch.nn.functional as F #activation function
from torch.optim import SGD 

class BasicNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #weights and biases to the first layer 
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        
        #weights and biases to the second layer 
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
        
        #final bias layer 
        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)
    
    def forward(self, input):
        #the input is scaled by weights and shifted by biases of layer 1
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01

        #the input is scaled by weights and shifted by biases of layer 2
        
        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        #sum of first and second layers 
        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias
        
        output = F.relu(input_to_final_relu)

#-----Visualization
    
input_doses = torch.linspace(start=0, end=1, steps=11) #create 11 tensors with values between 0, 1 

model = BasicNN_train()
output_values = model(input_doses)
sns.set(style='whitegrid')
sns.lineplot(
y=output_values.detach(), #here:detach is used to strip off the gradient 
linewidth=2.5)
plt.ylabel('Effectiveness')
plt.xlabel('Dose')

#-------------------------------------------

#Stochastic Gradient Descent 

#model.params will refer to any param with requires_grad=True, with learning_rate =0.1
optimizer = SGD(model.parameters(), lr=0.1) 

print("Final bias, before optimization: " + str(model.final_bias.data) + "\n")

#epoch -> each time the entire training data is seen by the model

for epoch in range(100):
    
    total_loss = 0 #measure of how well the model fits the data, the loss is larger if the model does not fit the training data 
    
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
    optimizer.zero_grad() #set loss to 0 for the new iteration of the loop

print("Step: " + str(epoch) + " Final Bias: " + str(model.final_bias.data) + "\n")

print("Final bias, after optimization: " + str(model.final_bias.data))