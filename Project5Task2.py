# import statements
from torchvision.datasets import mnist

import que1
import sys
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, device
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


print(que1.network.conv1.weight[1,0])
print(que1.network.conv1.weight)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


model_weights = []
conv_layers = []
model = que1.MyNetwork()
#model = Submodel()
model_children = list(model.children())


model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion= nn.CrossEntropyLoss()

# counter to keep count of the conv layers
counter = 0
# append all the conv layers and their respective weights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolutional layers: {counter}")


for weight, conv in zip(model_weights, conv_layers):
    # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")


# visualize the first conv layer filters
def visualize():
    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(4, 4, i+1) # we have 5x5 filters and total of 16 (see printed shapes)
        plt.imshow(filter[0, :, :].detach().cpu().numpy(), cmap='viridis')
        plt.axis('off')
        plt.savefig('filter1.png')
        plt.title("filter " + str(i))
        plt.xticks([])
        plt.yticks([])
    plt.show()





activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook



def effect_of_conv1_filter():
    model.conv1.register_forward_hook(get_activation('conv1'))
    data = que1.example_data[5][0]
    #sub = Submodel()
    #data = sub.forward(data)
    data = data.to(device)
    data.unsqueeze_(0)
    output = model(data)

    k = 0
    act = activation['conv1'].squeeze()
    # fig,ax = plt.subplots(4,4)
    plt.figure(figsize=(20, 17))
    # print('--------------------------------------', act.size(0)//1)

    for j in range(act.size(0)//1):
       #ax[i,j].imshow(act[k].detach().cpu().numpy())
       #k+=1
       #plt.savefig('fm1.png')
        plt.subplot(4, 4, j+1) # we have 5x5 filters and total of 16 (see printed shapes)
        plt.imshow(act[k].detach().cpu().numpy(), cmap='viridis')
        plt.axis('off')
        plt.savefig('fm1.png')
        plt.title("filter " + str(j))
        plt.xticks([])
        plt.yticks([])
        k+=1





def effect_of_conv2_filter():
    model.conv2.register_forward_hook(get_activation('conv2'))
    data = que1.example_data[5][0]
    # sub = Submodel()
    # data = sub.forward(data)
    data=data.to(device)
    data.unsqueeze_(0)
    output = model(data)
    act = activation['conv2'].squeeze()

    plt.figure(figsize=(20, 17))
    k=0

    #print('--------------------------------------', act.size(0)//2)
    for j in range(act.size(0)//2):  #for conv2 we had 20 filters but we are printing only for first 10
      plt.subplot(4, 4, j + 1)  # we have 5x5 filters and total of 16 (see printed shapes)
      plt.imshow(act[k].detach().cpu().numpy(), cmap='viridis')
      plt.axis('off')
      plt.savefig('fm2.png')
      plt.title("filter " + str(j))
      plt.xticks([])
      plt.yticks([])
      k += 1







class Submodel(que1.MyNetwork):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward( self, x ):
        x = F.relu( F.max_pool2d( self.conv1(x), 2 ) ) # relu on max pooled results of conv1
        x = F.relu( F.max_pool2d( self.conv2_drop( self.conv2(x)), 2 ) ) # relu on max pooled results of dropout of conv2
        return x


# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv

    visualize()
    effect_of_conv1_filter()
    effect_of_conv2_filter()

    #now we had to show the effects of conv1 and conv2 after applying override by using Submodel instance
    #and change the code inside in affect 1 and affect 2 such that we are using submodule filters on data
    effect_of_conv1_filter()
    effect_of_conv2_filter()

    # main function code
    return

if __name__ == "__main__":
    main(sys.argv)