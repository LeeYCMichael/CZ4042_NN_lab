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

class FFN(nn.Module):
    def __init__(self, no_features, no_hidden, no_labels):
        super().__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(no_features, no_hidden),
            nn.ReLU(),
            nn.Linear(no_hidden, no_labels),
        )

    def forward(self, x):
        logits = self.relu_stack(x)
        return logits

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
    
    
    
def my_train(hidden_neurons):
    
    model = FFN(no_features, hidden_neurons, no_labels)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    test_loss_ = []

    for epoch in range(no_epochs):
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loss = test_loop(test_dataloader, model, loss_fn)
    
        test_loss_.append(test_loss)

    return test_loss_



def main():

  hidden_neurons = [4, 8, 16, 32, 64, 128]

  no_threads = mp.cpu_count()
  p = mp.Pool(processes = no_threads)
  cost = p.map(my_train, hidden_neurons)
    

  # plot learning curves
  plt.figure(1)

  min_cost = []
  for l in range(len(hidden_neurons)):
    plt.plot(range(no_epochs), cost[l], label = 'hidden_neurons = {}'.format(hidden_neurons[l]))
    print(l)
    min_cost.append(min(cost[l]))


  plt.xlabel('epochs')
  plt.ylabel('mean square error')
  plt.title('GD learning')
  plt.legend()
  plt.savefig('figures/4.3b_1.png')

  
  plt.figure(2)
  plt.plot(hidden_neurons, min_cost)
  plt.xlabel('number of hidden neurons')
  plt.ylabel('test error')
  plt.xticks([4, 8, 16, 32, 64, 128])
  plt.title('test error vs. the number of hidden neurons')
  plt.savefig('figures/4.3b_2.png')

#  plt.show()


if __name__ == '__main__':
  main()






