# CNN using Pytorch by my own using scifi data
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def dataInNumpy(dataPath):
    data = unpickle(dataPath)
    raw_images = data[b'data']
    labels = np.array(data[b'labels'])
    images_list = []
    for i in range(raw_images.shape[0]):
        images_list.append(raw_images[i].reshape(32, 32, 3, order='F'))
    images = np.array(images_list)
    return images, labels

images, labels = dataInNumpy('./cifar-10-python/cifar-10-batches-py/data_batch_2')
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(images[i])

def CNN(input_shape, hidden, output_shape):
    class CNN(nn.Module):
        def __init__(self, input_shape, hidden, output_shape):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d()
    