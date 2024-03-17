# CREATING PREVIOUS NN FROM SCRATCH USING PYTORCH FOR COMPARISON PURPOSES

import torch

# f = w * x this is a linear function (excluding bias values)
# f = 2 * x (2 is the weight)
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) # initialize the weight

# model prediction
def forward(x):
    return x * w

# loss = MSE
def loss(y, y_predicted):
    return ((y - y_predicted)**2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 150

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred) 
    
    # gradients
    l.backward() # dl/dw
    
    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
        
    # empty the gradients
    w.grad.zero_()
    
    if epoch % 15 == 0:
        print(f'epoch {epoch +1}: w = {w:.3f}, loss= {l:.8f}')
        
print(f'Prediction after training: f(5) = {forward(5):.3f}')