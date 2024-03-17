# Does number of neurons should be the same as the features of the samples? lets try
import torch 
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, features, output):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return nn.ReLU(self.linear(x))

model = MyModel(3, 1)

# X = torch.tensor([1, 2, 3])
# Y = torch.tensor([2, 4, 6])
# epochs = 50
# for epoch in range(epochs):
#     predictions = model(X)

lFun = nn.Linear(2, 3, bias=False)
X = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
print('lFun weights: ', lFun.weight)
print('X: ', X)
out = lFun(X)
print('out: ', out)