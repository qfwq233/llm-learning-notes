import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# number of times the entire training dataset is passed through the model
n_epochs = 4
# number of training samples per batch (used for gradient updates)
batch_size_train = 64
# number of test/validation samples per batch (forward pass only, no backprop)
batch_size_test = 1000
# step size for updating model parameters
#learning_rate = 0.05
learning_rate = 0.001
# momentum factor to accelerate SGD and reduce oscillation
momentum = 0.5
# how many batches to wait before logging training status
log_interval = 10
# fixed seed for reproducibility (ensures same random numbers each run)
random_seed = 1
# set random seed for PyTorch (CPU)
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# download train dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,),(0.3081,)
                                               )
                                           ])),
    batch_size=batch_size_train, shuffle=True)
# download test dataset
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,),(0.3081,)
                                   )
                               ])),
batch_size=batch_size_test, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5, padding=2)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(270, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 270)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

network = Net().to(device)

optimizer = optim.AdamW(
    network.parameters(),
    lr=learning_rate,
)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(1, n_epochs + 1)]

def train(epoch):
    network.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # clear grad
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_loader.dataset),
                100. * idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (idx*64) + ((epoch - 1) * len(train_loader.dataset))
            )
            torch.save(network.state_dict(), '../model_pth')
            torch.save(optimizer.state_dict(), '../optimizer_pth')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

for epoch in range(1, n_epochs+1):
    train(epoch)
    test()

plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Training loss', 'Test loss'],  loc='upper right')
plt.xlabel('number of training examples')
plt.ylabel('loss')
plt.show()