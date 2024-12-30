# import statements
import cv2 as cv
import que1
import sys
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



#learning_rate = 0.01
#momentum = 0.5

continued_network = que1.MyNetwork()
continued_optimizer = optim.SGD(que1.network.parameters(), lr=que1.learning_rate,
                                momentum=que1.momentum)

network_state_dict_model = torch.load('C:/Users/srika/PycharmProjects/pythonProject/network_state_dict_model.pt')
continued_network.load_state_dict(network_state_dict_model)

optimizer_state_dict_model = torch.load('C:/Users/srika/PycharmProjects/pythonProject/optimizer_state_dict_model.pt')
continued_optimizer.load_state_dict(optimizer_state_dict_model)


for i in range(4,7):  #running few more ephocs
  que1.test_counter.append(i*len(que1.train_loader.dataset))
  que1.train(i)
  que1.test()


#test on our sample examples

with torch.no_grad():
    output = continued_network(que1.example_data)

fig = plt.figure()
for i in range(9):
    plt.subplot(3, 4, i + 1)
    plt.tight_layout()
    tmp = 'C:/Users/srika/Downloads/raoo' + str(i) + '.jpg'
    img = cv.imread(tmp, 0)
    plt.imshow(img, cmap='gray', interpolation='none')
    #tmp = 'C:/Users/srika/Downloads/raoo' + str(i) + '.jpg'
    plt.imsave(tmp, img, cmap="Greys")
    plt.title("Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
fig


fig = plt.figure()
plt.plot(que1.train_counter, que1.train_losses, color='blue')
plt.scatter(que1.test_counter, que1.test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig