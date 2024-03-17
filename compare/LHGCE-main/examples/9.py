# FEED FORWARD NET

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyper parameters
learning_rate = 0.001
input_shape = 784
hidden1 = 128
hidden2 = 64
output_shape = 10
batch_size = 100
num_epochs = 5

train_dataset = torchvision.datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST('./mnist', train=False, download=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

examples = iter(train_loader)
samples, labels = examples.next()
print('samples.shape: ', samples.shape, 'labels.shape: ', labels.shape)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
    
plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_shape, hidden1, hidden2, output_shape):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_shape, hidden1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden1, hidden2)
        self.l3 = nn.Linear(hidden2, output_shape)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # don't need to add softmax as the last activation function since
        # we're going to use cross entropy loss function and that will use
        # softmax internally
        return out
        
model = NeuralNet(input_shape, hidden1, hidden2, output_shape)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
print ('n_total_steps: ', n_total_steps)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28) # what does -1 mean here? 

        predictions = model(images)

        loss = criterion(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            print(f'batch = {i + 1} / {n_total_steps}, loss = {loss:.3f}')
    print (f'--------------epoch {epoch + 1}---------------')

# testing the model
# with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.reshape(-1, 28 * 28)
        predictions = model(images)
        predictionsIndex = torch.argmax(input=predictions, dim=1)
        n_correct += torch.sum((predictionsIndex == labels))
        n_samples += labels.shape[0]
        
    acc = 100.0 * n_correct / n_samples
    print(f'accuray = {acc}')
    