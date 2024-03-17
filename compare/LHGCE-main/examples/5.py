# CREATING PREVIOUS NN FROM SCRATCH USING PYTORCH FOR COMPARISON PURPOSES 

import torch
import torch.nn as nn

# f = w * x this is a linear function (excluding bias values)
# f = 2 * x (2 is the weight)
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) # initialize the weight

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # defining layers
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.001
n_iters = 200

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model.forward(X)
    # loss
    l = loss(Y, y_pred) 
    # gradients
    l.backward() # dl/dw
    # update weights
    optimizer.step()
    # empty the gradients
    optimizer.zero_grad()
    if epoch % 20 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch +1}: w = {w[0][0].item():.3f}, loss= {l:.8f}')
        
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')