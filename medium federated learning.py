import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy  # PySyft import
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

# Initialize the hook
hook = sy.TorchHook(torch)

# Defining virtual clients (workers)
client0 = sy.VirtualWorker(hook, id="client0")
client1 = sy.VirtualWorker(hook, id="client1")
client2 = sy.VirtualWorker(hook, id="client2")
client3 = sy.VirtualWorker(hook, id="client3")

# Configuration arguments
args = edict({
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 0.01
})

# Federated DataLoader (training dataset)
federated_train_loader = sy.FederatedDataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])).federate((client0, client1, client2, client3)),
    batch_size=args.batch_size, shuffle=True)

# Standard DataLoader (testing dataset)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.batch_size, shuffle=True)


# Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Training function
def train(args, model, federated_train_loader, optimizer, epoch, train_loss_batch, train_loss_epoch, id0, id1, id2, id3):
    model.train()
    t_loss = 0
    total = 0
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        # Get the client location (worker)
        client = data.location.id
        print('Processed at :', client)
        
        # Send the model to the correct worker
        model.send(data.location)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = F.nll_loss(output, target)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Get the model back to the local machine
        model.get()
        
        # Log and update loss
        if batch_idx % 30 == 0:
            loss = loss.get()  # Get loss back from remote worker
            t_loss += loss.item()
            total += 1
            train_loss_batch.append(loss.item())
            
            # Track client losses
            if client == 'client0':
                id0.append(loss.item())
            elif client == 'client1':
                id1.append(loss.item())
            elif client == 'client2':
                id2.append(loss.item())
            elif client == 'client3':
                id3.append(loss.item())

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
                100. * batch_idx / len(federated_train_loader), loss.item()))
    
    t_loss /= total
    train_loss_epoch.append(t_loss)


# Testing function
def test(args, model, test_loader, testing_loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum loss over batch
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    testing_loss.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Instantiate the model and optimizer
model = Net()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

# Containers for tracking losses
train_loss_batch = []
train_loss_epoch = []
testing_loss = []
id0, id1, id2, id3 = [], [], [], []

# Training loop
for epoch in range(1, args.epochs + 1):
    train(args, model, federated_train_loader, optimizer, epoch, train_loss_batch, train_loss_epoch, id0, id1, id2, id3)
    test(args, model, test_loader, testing_loss)

# Plotting the losses
plt.plot(train_loss_batch)
plt.title("Training Loss in batches")
plt.show()

plt.plot(train_loss_epoch)
plt.title("Training Loss per epoch")
plt.show()

# Client-specific loss plotting
f = plt.figure()
f.set_figwidth(10)
f.set_figheight(8)
plt.plot(id0, 'r', label='client 0')
plt.plot(id1, 'g', label='client 1')
plt.plot(id2, 'b', label='client 2')
plt.plot(id3, 'y', label='client 3')
plt.legend()
plt.title("Client Losses during training wrt batches")
plt.show()

# Save the model
torch.save(model.state_dict(), "model.pt")