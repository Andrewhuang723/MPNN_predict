import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import dgl

device = torch.device("cuda")

def split(X, Y, rate):
    X_train = X[int(len(X)*rate):]
    y_train = Y[int(len(Y)*rate):]
    X_val = X[:int(len(X)*rate)]
    y_val = Y[:int(len(Y)*rate)]
    return X_train, y_train, X_val, y_val

y_train = torch.load("./data/train_regression.pkl")
y_test = torch.load("./data/test_regression.pkl")
y_train = y_train[:, 5:6]
y_test = y_test[:, 5:6]

print(y_train.shape)

train_graphs, _ = dgl.data.load_graphs("./data/train_graphs_explicit_H.bin")
test_graphs, _ = dgl.data.load_graphs("./data/test_graphs_explicit_H.bin")

train_graphs, y_train, val_graphs, y_val = split(train_graphs, y_train, 0.1)
print("train: ", len(train_graphs), "\nval: ", len(val_graphs),
      "\ny_train: ", y_train.shape, "\ny_val: ", y_val.shape)
print(train_graphs[0])

from torch import optim
from Regressor_model import GCN_graph_to_num_regressor, Mol2NumNet_regressor, MolDataset, collate
from torch.optim.lr_scheduler import MultiStepLR

train_loader = DataLoader(dataset=MolDataset(train_graphs, y_train),
                            batch_size=64, shuffle=False, collate_fn=collate)
val_loader = DataLoader(dataset=MolDataset(val_graphs, y_val),
                        batch_size=64, shuffle=False, collate_fn=collate)

#net = GCN_graph_to_num_regressor(74, 100, y_train.shape[-1])
#net = net.cuda()
#optimizer = optim.Adam(net.parameters(), lr=0.01)

net_1 = Mol2NumNet_regressor(64, 128, y_train.shape[-1])
net_1 = net_1.cuda()
optimizer = optim.Adam(net_1.parameters(), lr=1e-3, weight_decay=0.01)
schedular_lr = MultiStepLR(optimizer=optimizer, milestones=[30, 80], gamma=0.1)
criterion = torch.nn.MSELoss().cuda()

import time
from pytorch_tools import EarlyStopping

iteration_num = 1000
epoch_losses = []
val_epoch_losses = []
dur = []
early_stopping = EarlyStopping(patience=20)

for epoch in range(iteration_num):
    epoch_loss = 0
    val_epoch_loss = 0
    i = 0
    if epoch >= 1:
        t0 = time.time()
    for bg, properties in train_loader:
        properties = torch.stack(properties).to(device, dtype=torch.float)
        bg = bg.to(device)
        hidden, prediction = net_1(bg, bg.ndata["h"], bg.edata["h"])
        #hidden, prediction = net(bg)
        loss = 0
        for i in range(len(properties)):
            loss += criterion(prediction[i,:], properties[i,:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += (loss/len(properties)).detach().item()
    epoch_loss /= len(train_loader)

    for bg, properties in val_loader:
        properties = torch.stack(properties).to(device, dtype=torch.float)
        bg = bg.to(device)
        val_hidden, val_prediction = net_1(bg, bg.ndata["h"], bg.edata["h"])
        #val_hidden, val_prediction = net(bg)
        val_loss = 0
        for i in range(len(properties)):
            val_loss += criterion(val_prediction[i,:], properties[i,:])
        val_epoch_loss += (val_loss/len(properties)).detach().item()
    val_epoch_loss /= len(val_loader)
    if epoch >= 1:
        dur.append(time.time() - t0)
    print('Epoch {} | loss {:.4f} | Time(s) {:.4f} | val loss {:.4f}'.format(epoch, epoch_loss, np.mean(dur), val_epoch_loss))
    epoch_losses.append(epoch_loss)
    val_epoch_losses.append(val_epoch_loss)
    early_stopping(val_loss=val_epoch_loss, model=net_1)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# dict = net.state_dict()
# dict["loss"] = epoch_losses
# dict["val_loss"] = val_epoch_losses
# torch.save(dict, "GCN_regression_param.pkl")

dic_1 = net_1.state_dict()
dic_1["loss"] = epoch_losses
dic_1["val_loss"] = val_epoch_losses
torch.save(dic_1, "./Model/MPNN_regression_dipole_explicit_H.pkl")