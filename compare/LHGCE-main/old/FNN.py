import torch
from torch import nn, optim
import numpy as np
from torch.nn.modules.container import Sequential
class FCNN(nn.Module):
    def __init__(self, inputShape, dropout):
        super(FCNN, self).__init__()
        self.DNN = nn.Sequential(
            nn.Linear(inputShape, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 4),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(4, 1),
            nn.LeakyReLU(),
        )
        # self.encoder = nn.Sequential(
        #     nn.Linear(inputShape, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(128, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(32, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(64, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(128, inputShape),
        #     nn.ReLU(),
        # )
        
    def forward(self, x):
        return self.DNN(x)

def trainFNN(dataDic, nEpochs, nBatchsize, dropout, lr):
    labels = torch.from_numpy(dataDic['y'])
    X = dataDic['X']
    model = FCNN(X.shape[1], dropout)
    # should add weighted loss
    BCELoss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.01)
    for epoch in range(nEpochs):
        indices = np.random.permutation(labels.shape[0])
        for boundary in range(0, len(indices), nBatchsize):
            batchIndex = indices[boundary: boundary + nBatchsize]
            batchLoss = 0
            XTrain = X[batchIndex].float()
            y_pred = model(XTrain)
            y = labels[batchIndex].float()
            dnnLoss = BCELoss(y_pred, y)
            batchLoss += dnnLoss
            batchLoss.backward()
            optimizer.step()
        # scheduler.step()
        # print('last LR: ', scheduler.get_last_lr())

        if epoch % 10 == 0:
            print('-> epoch: ', epoch, 'DNN batchLoss: ', batchLoss.item())
            # for name, param in model.named_parameters():
                # print('-> model parameters: ', name, '=', param)

    return model

def testFNN(model, dataDic):
    # DNN Input
    model.eval()
    X = dataDic['X'].float()
    y_pred = model(X)
    # BCEWithLogits append a sigmoid
    sig = nn.Sigmoid()
    return sig(y_pred)
