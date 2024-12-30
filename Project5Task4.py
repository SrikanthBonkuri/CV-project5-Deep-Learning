# import statements
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


#class definitions
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5,5), stride=(1,1))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5,5), stride=(1,1))
        self.conv2_drop = nn.Dropout2d(0.5)   #change dropout and analyze the performance
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        pass

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)





n_epochs = 3               #change epochs and analyze the performance
batch_size_train = 6000    #change batch size and analyze the performance
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 1

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

# we will test the network by experimenting on Changing Batch size, Change epochs, Change Drop rate.

network = MyNetwork()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]



def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
          train_losses.append(loss.item())
          train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
          torch.save(network.state_dict(), 'network_state_dict_model1.pt')
          torch.save(optimizer.state_dict(), 'optimizer_state_dict_model1.pt')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))



# useful functions with a comment for each function
def train_network():
    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()
    return

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig
    #plot graph based on parameters and success ratio



# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv

    torch.manual_seed(52)
    torch.backends.cudnn.enabled = False

    network = que1.MyNetwork()
    print(network)

    train_network()
    #we will test the network by experimenting on Changing Batch size, Change epochs, Change Drop rate.
    #here i changed batch size, change epochs and change drop rate and ran train network to test and anayze the performance
    #and pointed out a graph for the performance.

    return



if __name__ == "__main__":
    main(sys.argv)