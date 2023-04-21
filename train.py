import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.optim as optim
import matplotlib.pyplot as plt
from model.MLP import mlp5
from model.linear import linear
import numpy as np

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath("../"))
sys.path.append("./model")
from read_tabular import read_tabular

torch.set_default_tensor_type(torch.DoubleTensor)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
params = {
    'lr': 1e-3,          # lr: 1e-4
    'weight_decay': 0,
    'epochs': 100,
    'k_dim': 4,  # matrix W: k * d
    'alpha': 1000,
} 

def plot_curve(train_acc_values, train_loss_values):
    # plot train/test accuracy and loss
    # accuracy
    path = f"./output/AccAndLoss/alpha_{params['alpha']}-k_{params['k_dim']}-epoch_{params['epochs']}-lr_{params['lr']}"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.plot(train_acc_values)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    plt.savefig(f'{path}/accuracy.jpg')
    plt.close()

    # loss
    plt.plot(train_loss_values)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    plt.savefig(f'{path}/loss.jpg')
    plt.close()

def plot_point_line(X_train, y_label, model):
    # points and line
    plt.figure(figsize=(8, 4))
    x_point = X_train.detach().cpu().numpy()
    y_label = y_label.reshape(-1).detach().cpu().numpy()
    colors = ['red', 'green']  
    labels = ['Zero', 'One']  
    # draw point
    for i in range(x_point.shape[1]):  
        plt.scatter(x_point[y_label == i, 0], 
                    x_point[y_label == i, 1],  
                    c=colors[i],  
                    label=labels[i]) 
    plt.legend() 
    # draw line
    w1 = model.linear1.weight.detach().cpu().numpy()[0][0]
    w2 = model.linear1.weight.detach().cpu().numpy()[0][1]
    b = model.linear1.bias.detach().cpu().numpy()[0]
    x = np.linspace(0, 1, 50)
    y = -w1/w2 * x - b/w2
    plt.plot(x, y)
    plt.savefig(f"./output/scatter-alpha_{params['alpha']}-k_{params['k_dim']}-epoch_{params['epochs']}-lr_{params['lr']}.jpg")

def get_pair(x_tr,y_tr):
    sample_num = x_tr.shape[0]
    if (sample_num % 2 == 1):  # ensure the number of samples is even
        x_tr = x_tr[:-2, :]
        y_tr = y_tr[:-2]
    # shuffle the dataset
    index_shuffle = torch.randperm(sample_num)

    x_tr = x_tr[index_shuffle,:]
    y_tr = y_tr[index_shuffle]

    # pairing
    x_pair_1 = x_tr[:sample_num // 2,:]
    x_pair_2 = x_tr[sample_num // 2:,:]
    delta = (y_tr[:sample_num // 2] == y_tr[sample_num // 2:]) + 0
    delta = delta.double()
    # Transform
    x_pair_1_trans = torch.zeros([sample_num // 2,params['k_dim']]).to(device)
    x_pair_2_trans = torch.zeros([sample_num // 2,params['k_dim']]).to(device)

    for index in range(sample_num // 2):  # better not to use for loop
        W_matrix = torch.normal(mean=0, std=1, size=(params['k_dim'], x_tr.shape[1])).to(device)
        x_pair_1_trans[index] = torch.mv(W_matrix, x_pair_1[index])  # matrix * vector
        x_pair_2_trans[index] = torch.mv(W_matrix, x_pair_2[index])  # matrix * vector

    return x_pair_1_trans, x_pair_2_trans, delta

def get_trans(x_tr,y_tr):
    sample_num = x_tr.shape[0]
    # shuffle the dataset
    index_shuffle = torch.randperm(sample_num)

    x_tr = x_tr[index_shuffle,:].to(device)
    y_tr = y_tr[index_shuffle]

    # Transform
    x_tr_trans = torch.zeros([sample_num,params['k_dim']]).to(device)

    for index in range(sample_num):  # better not to use for loop
        W_matrix = torch.abs(torch.normal(mean=0, std=params['std'], size=(params['k_dim'], x_tr.shape[1]))).to(device)
        x_tr_trans[index] = torch.mv(W_matrix, x_tr[index])  # matrix * vector  
    return x_tr_trans,y_tr

# Function to save the model 

def pair_model(model,x_pair_1,x_pair_2):
    y_pred_1 = model(x_pair_1)
    y_pred_2 = model(x_pair_2)
    delta_pred = 2 - 2 * F.sigmoid(params['alpha'] * (y_pred_1 - y_pred_2) * (y_pred_1 - y_pred_2))
    return delta_pred

def saveModel(): 
    path = './model/model_pths/NetModel.pth'
    torch.save(model.state_dict(), path)

def train(model,
          optimizer,
          device,
          X_train,
          y_label,
          epochs=params['epochs']):

    #! loss calculation is defined here
    criterion = nn.BCELoss()
    train_loss_values = []
    train_acc_values = []

    # train
    for epoch in range(epochs):
        train_loss, train_correct = 0, 0
        train_loss = 0
        total_num = 0

        model.train()

        x_tr = X_train.to(device)
        y_tr = y_label.to(device)

        optimizer.zero_grad()
        x_tr.requires_grad_(True)
        y_pred = model(x_tr)
        
        # loss = torch.sum(torch.square(y_pred - y_tr))
        loss = criterion(y_pred, y_tr)
        train_loss += loss.item()
        print(y_pred.shape)
        train_correct += (torch.abs(y_pred - y_tr) < 0.5).sum().item()


        total_num += x_tr.size(0)
        loss.backward()
        optimizer.step()

        avg_tr_loss = train_loss / total_num
        avg_tr_acc = train_correct / total_num
        train_loss_values.append(avg_tr_loss)
        train_acc_values.append(avg_tr_acc)
        print(f"epoch: {epoch} train_loss: {avg_tr_loss:.4f}  train_acc: {avg_tr_acc:.4f}")
        saveModel()

    plot_curve(train_acc_values, train_loss_values)
    plot_point_line(X_train, y_label, model)

def train_pair(model,
          optimizer,
          device,
          X_train,
          y_label,
          epochs=params['epochs']):

    #! loss calculation is defined here
    criterion = nn.BCELoss()
    train_loss_values = []
    train_acc_values = []

    # train
    for epoch in range(epochs):
        train_loss, train_correct = 0, 0
        train_loss = 0
        total_num = 0

        model.train()

        x_tr = X_train.to(device)
        y_tr = y_label.to(device)

        x_pair_1, x_pair_2, delta = get_pair(x_tr, y_tr)


        optimizer.zero_grad()
        x_pair_1.requires_grad_(True)
        x_pair_2.requires_grad_(True)

        delta_pred = pair_model(model,x_pair_1,x_pair_2)
        loss = criterion(delta_pred, delta)
        train_loss += loss.item()
        train_correct += (torch.abs(delta - delta_pred) < 0.5).sum().item()

        total_num += x_pair_1.size(0)
        loss.backward()
        optimizer.step()

        avg_tr_loss = train_loss / total_num
        avg_tr_acc = train_correct / total_num
        train_loss_values.append(avg_tr_loss)
        train_acc_values.append(avg_tr_acc)
        print(f"epoch: {epoch} train_loss: {avg_tr_loss:.4f}  train_acc: {avg_tr_acc:.4f}")
        saveModel()

    plot_curve(train_acc_values, train_loss_values)
    plot_point_line(X_train, y_label, model)


def test(model,X_train,y_label,X_test,y_test_label):
    x_tr_trans,y_tr = get_trans(X_train,y_label)
    x_tr_trans = x_tr_trans.to(device)
    X_test = X_test.to(device)
    y_tr = y_tr.long().to(device)
    y_test_label = y_test_label.to(device)

    test_sample_num = X_test.shape[0]
    y_pred_list = []
    for index in range(test_sample_num):
        delta = pair_model(model,x_tr_trans,X_test[index].repeat(x_tr_trans.shape[0],1))
        delta_discrete = (delta > 0.5) + 0
        y_pred = delta_discrete^y_tr
        if torch.sum(y_pred) > x_tr_trans.shape[0]//2:
            y_pred_list.append(1)
        else:
            y_pred_list.append(0)
        
    y_pred_list = torch.tensor(y_pred_list).to(device)
    acc = torch.sum(y_pred_list == y_test_label)/test_sample_num
    print(f"acc:{acc.cpu().numpy()}")
    return acc.cpu().numpy()

if __name__ == "__main__": 
    # X_train, y_train, X_test, y_test, train_loader, test_loader = read_tabular()
    X = Variable(torch.tensor(np.load("../data/x_point.npy")))
    y = Variable(torch.tensor(np.load("../data/y_label.npy").astype(float)))


    X_train = X[0:400, :]
    y_label = y[0:400].reshape(-1,1)

    X_test = X[400:, :]
    y_test_label = y[400:].reshape(-1,1)

    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = linear(in_dim=params['k_dim'], out_dim=1).to(device)

    # optimizer: Adam
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    print("The model will be running\n") 
    train_pair(model=model,
          optimizer=optimizer,
          device=device,
          X_train = X_train,
          y_label = y_label)
    print('Finished Training\n') 
    test(model,X_train,y_label,X_test,y_test_label)