import matplotlib.pyplot as plt
import torch
import os
from Regressor_model import GCN_graph_to_num_regressor, Mol2NumNet_regressor
import dgl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda")



dict = torch.load("./Model/MPNN_regression_dipole_explicit_H.pkl")
loss = dict["loss"]
val_loss = dict["val_loss"]
print(loss[-1], val_loss[-1])
plt.figure()
plt.plot(range(len(loss)), loss, label="Training")
plt.plot(range(len(val_loss)), val_loss, label="Validation")
plt.legend(loc="upper right")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("MPNN_regress: lr=0.0001")

plt.show()

y_test = torch.load("./data/test_regression.pkl")
y_test = y_test[:, 3:4]
test_graphs, test_features = dgl.data.load_graphs("./data/test_graphs_explicit_H.bin")

print("test: ", len(test_graphs), "y_test: ", y_test.shape)
print(test_graphs[0])
class MolDataset(Dataset):
    def __init__(self, graphs, properties):
        self.graphs = graphs
        self.properties = properties
        print('Dataset includes {:d} graphs'.format(len(graphs)))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item], self.properties[item]

def collate(samples):
    graphs, property= map(list, zip(*samples))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return bg, property

test_loader = DataLoader(dataset=MolDataset(test_graphs, y_test), batch_size=64, shuffle=False, collate_fn=collate)

model = Mol2NumNet_regressor(64, 128, y_test.shape[-1]).cuda()
model.eval()
dict = torch.load("./Model/MPNN_regression_dipole_explicit_H.pkl")
del dict["loss"]
del dict["val_loss"]
model.load_state_dict(dict)

criterion = torch.nn.MSELoss().cuda()
test_loss = 0
for j, (bg, properties) in zip(range(len(test_loader)), test_loader):
    properties = torch.stack(properties).to(device, dtype=torch.float)
    bg = bg.to(device)
    hidden, prediction = model(bg, bg.ndata["h"], bg.edata["h"])

    if j == 0:
        y_pred = prediction.detach().cpu().numpy()
    else:
        y_pred = np.concatenate([y_pred, prediction.detach().cpu().numpy()], axis=0)
    loss = 0
    for i in range(len(properties)):
        loss += criterion(prediction[i, :], properties[i, :])
    test_loss += (loss/len(properties)).detach().item()
test_loss /= len(test_loader)
print(test_loss)

stand = np.load("stat.npz")
mean = stand["mean"][:, 3:4]
std = stand["std"][:, 3:4]

# Ha --> eV
#mean_re = [mean[:,i] * 27.21138602 for i in range(mean.shape[1])]
#std_re = [std[:,i] * 27.21138602 for i in range(mean.shape[1])]
#print(mean_re, std_re)
'''
# Ha --> eV
ha = [1, 2, 3, 5, 6, 7, 8, 9]
if mean.shape[-1] == 11:
    for i in range(mean.shape[1]):
        if i in ha:
            mean[:,i] = mean[:,i] * 27.21138602
    for i in range(mean.shape[1]):
        if i in ha:
            std[:,i] = std[:,i] * 27.21138602
'''
reconstruct_y_test = y_test.numpy() * std + mean
reconstruct_y_pred = y_pred * std + mean

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def metrics(y_pred, y_test):
    R_square = []
    mae = []
    for i in range(y_test.shape[1]):
        mae.append(mean_absolute_error(y_test[:,i], y_pred[:,i]))
        R_square.append(r2_score(y_test[:,i], y_pred[:,i]))
    return np.array(R_square), np.array(mae)

real_R_square, real_MAE = metrics(reconstruct_y_pred, reconstruct_y_test)



def plot(y_pred, y_test, R_square, MAE, data_columns):
    for i in range(y_test.shape[1]):
        z = np.polyfit(y_test[:,i], y_pred[:,i], 1)
        plt.subplots(figsize=(9, 7))
        plt.scatter(y_pred[:,i], y_test[:,i], c='g', s=10)
        plt.plot(z[1] + z[0] * y_test[:,i], y_test[:,i], c='blue', ls="--" ,linewidth=1, label="Prediction")
        plt.plot(y_test[:,i], y_test[:,i], color='k', ls="--", linewidth=1, label="Testing")
        plt.legend(loc="upper right")
        plt.title(data_columns[i] + "\nR_square: {:.4f} \nmae: {:.4f}".format(R_square[i], MAE[i]))
        plt.xlabel('Predictedion')
        plt.ylabel('Testing sets')
        plt.show()
plot(reconstruct_y_pred, reconstruct_y_test, real_R_square, real_MAE, [
    "mu"])

