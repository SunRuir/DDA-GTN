import torch

N_SAMPLES = 100
N_FEATURES = 50
N_HIDDEN = 60
N_OUTPUT = 10
N_EPOCHS = 100
LEARNING_RATE = 0.001


x = torch.rand(N_SAMPLES, N_FEATURES, dtype=torch.float)
y = torch.rand(N_SAMPLES, N_OUTPUT, dtype=torch.float)

w1 = torch.rand(N_FEATURES, N_HIDDEN, requires_grad=True, dtype=torch.float)
w2 = torch.rand(N_HIDDEN, N_OUTPUT, requires_grad=True, dtype=torch.float)

for epoch in range(100):
    y_predicted = torch.mm(x, w1).clamp(min=0).mm(w2)
    loss = (y_predicted - y).pow(2).sum()
    loss.backward()

    if epoch % 10 == 0:
        print(f'loss: {loss:5f}')
    with torch.no_grad():   
        w1 -= w1.grad * LEARNING_RATE
        w2 -= w2.grad * LEARNING_RATE
        w1.grad.zero_()
        w2.grad.zero_()
