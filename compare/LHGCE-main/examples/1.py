import torch
x = torch.randn(3, requires_grad=True)
print('x: ', x)
y = x + 2
print('y: ', y)
y = y * 2
print('y: ', y)
y = x.mean()
print('y: ', y)

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

y_hat = w * x
print('y_hat: ', y_hat)
loss = (y_hat - y)**2
print('lose: ', loss)
print('loss.grad: ', loss.grad)
loss.backward()
print('loss.grad: ', loss.grad)