import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pylab as plt

import multiprocessing as mp

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection  import train_test_split
from sklearn import preprocessing

import os
if not os.path.isdir('figures'):
    os.makedirs('figures')


lr = 0.001

no_features = 8
no_hidden = 5
no_labels = 1
no_epochs = 200
batch_size = 32

SEED = 10
torch.manual_seed(SEED)
np.random.seed(SEED)

X, y = fetch_california_housing(return_X_y=True)

# Split the data into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


class MyDataset(Dataset):
  def __init__(self, X, y):
    self.X =torch.tensor(X, dtype=torch.float)
    self.y =torch.tensor(y, dtype=torch.float).unsqueeze(1)
    
  def __len__(self):
    return len(self.y)

  def __getitem__(self,idx):
    return self.X[idx], self.y[idx]

train_data = MyDataset(x_train, y_train)
test_data = MyDataset(x_test, y_test)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

class Net(nn.Module):

    def __init__(self, no_features, no_labels, no_hidden, no_layers):
        super(Net, self).__init__()
        current_dim = no_features
        self.layers = nn.ModuleList()
        for l in range(no_layers):
            self.layers.append(nn.Linear(current_dim, no_hidden))
            nn.ReLU()
            current_dim = no_hidden
        self.layers.append(nn.Linear(current_dim, no_labels))
 

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        out = self.layers[-1](x)
        return out    

def train_loop(train_dataloader, model, loss_fn, optimizer):
    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()

    train_loss /= size
    return train_loss
    

def test_loop(test_dataloader, model, loss_fn):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= size
 
    return test_loss
    
    
    
def my_train(hidden_layers):
    
    model = Net(no_features, no_hidden, no_labels, hidden_layers)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    test_loss_ = []

    for epoch in range(no_epochs):
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loss = test_loop(test_dataloader, model, loss_fn)
    
        test_loss_.append(test_loss)

    return test_loss_



def main():

  hidden_layers = [1, 2, 3, 4, 5, 6]

  no_threads = mp.cpu_count()
  p = mp.Pool(processes = no_threads)
  cost = p.map(my_train, hidden_layers)

  # plot learning curves
  plt.figure(1)

  min_cost = []
  for l in range(len(hidden_layers)):
    plt.plot(range(no_epochs), cost[l], label = 'hidden layers = {}'.format(hidden_layers[l]))
    min_cost.append(min(cost[l]))

  plt.xlabel('epochs')
  plt.ylabel('mean square error')
  plt.title('GD learning')
  plt.legend()
  plt.savefig('figures/4.4b_1.png')

  
  plt.figure(2)
  plt.plot(hidden_layers, min_cost)
  plt.xlabel('no of hidden layers')
  plt.ylabel('test error')
  plt.title('test error vs.the depth of DNN')
  plt.savefig('figures/4.4b_2.png')


if __name__ == '__main__':
  main()






