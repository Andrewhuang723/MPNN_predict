import torch
import dgl
from dgl import DGLGraph
import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import networkx as nx
from torch import optim
from Regressor_model import GCN_graph_to_num_regressor, Mol2NumNet_regressor, MolDataset, collate
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_tools import EarlyStopping
from Regressor_model import train_model, predict_property

device = torch.device("cuda")

gdb_dir = './data/GDB9Quantum.csv'
prop_name = 'HOMO (Ha)'

if not os.path.exists(prop_name):
    os.makedirs(prop_name)
    os.makedirs(os.path.join(prop_name, 'Model'))

Propers = pd.read_csv(gdb_dir, index_col=None)[prop_name]
smiles = pd.read_csv(gdb_dir, index_col=None)['SMILES']

print(Propers.shape) #(138855, 15)

def start_plot(figsize=(10, 8), style = 'whitegrid'):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1,1)
    plt.tight_layout()
    with sns.axes_style(style):
        ax = fig.add_subplot(gs[0,0])
    return ax

### Standardization
ymax = np.mean(Propers) + 3 * np.std(Propers)
ymin = np.mean(Propers) - 3 * np.std(Propers)
outliers_above = Propers.loc[Propers > ymax]
outliers_below = Propers.loc[Propers < ymin]
outliers = pd.concat([outliers_above, outliers_below], axis=0)
new_data = Propers.drop(index=outliers.index, axis=0)
new_smiles = smiles.drop(index=outliers.index, axis=0)
value = [ymax, ymin, np.mean(Propers)]

### Plot Outliers
color = ['red', 'red', 'darkorange']
text = ['$\mu$ + 3 $\sigma$', '$\mu$ - 3 $\sigma$', '$\mu$ = %.4f' % np.mean(Propers)]

ax = start_plot(style='darkgrid')
ax.scatter(range(len(outliers)), outliers.values, c='navy', label=r'$\mathbf{Outliers: %s}$' % len(outliers), s=5)
ax.scatter(range(len(new_data.values)), new_data, c='teal', label=r'$\mathbf{Pass: %s}$' % len(new_data), s=5)

for i, k in zip(value, color):
    ax.axhline(i, ls='--', c=k)

trans = transforms.blended_transform_factory(
    ax.get_yticklabels()[0].get_transform(), ax.transData)

for i, j, k in zip(value, text, color):
    ax.text(0, i, j, c=k, va='center', ha='right', transform=trans)

ax.legend(loc='upper right', frameon=True, shadow=True)
plt.xlabel('datasize')
plt.ylabel(prop_name)
plt.title('%s\n  $\sigma = %.4f $'%(prop_name, np.std(Propers)))
plt.savefig(os.path.join(prop_name, 'scatterplot.png'))
print('\n----------------------------------------------')
print('Outliers scatterplot saved')
print('----------------------------------------------')

# data scaling ~ N(0,1)
scaler = StandardScaler(copy=True)
scaler.fit(new_data.values.reshape(-1, 1))
rescale_std = new_data.std()
rescale_mean = new_data.mean()
Propers1 = scaler.transform((new_data.values.reshape(-1, 1)))
Propers1 = torch.tensor(Propers1)

print('data after removing outliers: ', Propers1.shape)

X_train, X_test = train_test_split(new_smiles.values, test_size=0.2, random_state=0)
y_train, y_test = train_test_split(Propers1, test_size=0.2, random_state=0)

# torch.save(y_train, "./%s/sd_train_regression.pkl" % prop_name)
# torch.save(y_test, "./%s/sd_test_regression.pkl" % prop_name)

print('\n------------------SPLITTING DATA---------------------')
print("X_train: ", X_train.shape, "  y_train: ", y_train.shape)
print("\nX_test: ", X_test.shape, "  y_test: ", y_test.shape)
print('-----------------------------------------------------')

### An example for molecular graphs
# mol0 = get_mol(smiles[130000], explicit_H=True)
# SAMPLE_MOL = mol2graph(smiles[130000], explicit_H=True)
#
# print(SAMPLE_MOL, "\n node features: ", SAMPLE_MOL.ndata["h"],
#       "\n edge features: ", SAMPLE_MOL.edata["h"])
#
# nx.draw(SAMPLE_MOL.to_networkx(), with_labels=True)
# plt.show()

### Load the graphs and splitted into training set & testing set
All_graphs = dgl.data.load_graphs('./data/graphs_explicit_H.bin')[0]

### All graphs should cancel out the outliers
outliers_idx = list(outliers.index)
Gs = [i for j, i in enumerate(All_graphs) if j not in outliers_idx]
train_graphs, test_graphs = train_test_split(Gs, test_size=0.2, random_state=0)
print('\n-------------------LOADING GRAPHS-----------------------------------')
print('train graphs: ', len(train_graphs), '\ntest_graphs: ', len(test_graphs))
print('--------------------------------------------------------------------')

def split(X, Y, rate):
    X_train = X[int(len(X)*rate):]
    y_train = Y[int(len(Y)*rate):]
    X_val = X[:int(len(X)*rate)]
    y_val = Y[:int(len(Y)*rate)]
    return X_train, y_train, X_val, y_val

train_graphs, y_train, val_graphs, y_val = split(train_graphs, y_train, 0.1)
print('\n------------------------SPLITTING GRAPHS--------------------------')
print("train_graphs: ", len(train_graphs), "  val_graphs: ", len(val_graphs),
      "\ny_train: ", len(y_train), "  y_val: ", len(y_val))
print('------------------------------------------------------------------')

train_loader = DataLoader(dataset=MolDataset(train_graphs, y_train),
                            batch_size=64, shuffle=False, collate_fn=collate)
val_loader = DataLoader(dataset=MolDataset(val_graphs, y_val),
                        batch_size=64, shuffle=False, collate_fn=collate)

net_1 = Mol2NumNet_regressor(64, 128, y_train.shape[-1])
net_1 = net_1.cuda()
optimizer = optim.Adam(net_1.parameters(), lr=1e-3, weight_decay=0.01)
schedular_lr = MultiStepLR(optimizer=optimizer, milestones=[10, 30], gamma=0.1)
criterion = torch.nn.MSELoss().cuda()

early_stopping = EarlyStopping(patience=20)
iteration_num = 1000
model, model_dict = train_model(net_1, train_loader, val_loader, epochs=iteration_num, optimizer=optimizer,
                                loss_function=criterion, early_stopping=early_stopping, schedule_lr=schedular_lr)

torch.save(model_dict, "./%s/Model/sched_sd_MPNN_regression_explicit_H.pkl" % prop_name)

### Plot loss iteration curve
model_dict = torch.load("./%s/Model/sched_sd_MPNN_regression_explicit_H.pkl" % prop_name)
epoch_losses = model_dict['loss']
val_epoch_losses = model_dict['val_loss']
print('\nTRAINING RESULTS')
print('=================================================')
print('TRAINING LOSS: ', epoch_losses[-1], '\nVALIDATION LOSS: ', val_epoch_losses[-1])
print('=================================================')

plt.figure(figsize=(7, 7))
plt.plot(range(len(epoch_losses)), epoch_losses, label="Training")
plt.plot(range(len(val_epoch_losses)), val_epoch_losses, label="Validation")
plt.legend(loc="upper right")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("MPNN_regress_explicit_H")
plt.savefig(os.path.join(prop_name, 'sched_sd_Train&val_curve.png'))

### Evaluation

#### Loading model
model_dict = torch.load("./%s/Model/sched_sd_MPNN_regression_explicit_H.pkl" % prop_name)
model = Mol2NumNet_regressor(64, 128, y_test.shape[-1]).cuda()
del model_dict["loss"]
del model_dict["val_loss"]
model.load_state_dict(model_dict)

#### Testing
test_loader = DataLoader(dataset=MolDataset(test_graphs, y_test), batch_size=64, shuffle=False, collate_fn=collate)
criterion = torch.nn.MSELoss().cuda()

y_pred, test_loss = predict_property(model, test_loader, loss_function=criterion)
print('\n=======================================')
print('TEST LOSS: ', test_loss)
print('=======================================')

#### Rescale y and y_hat
reconstruct_y_test = y_test.numpy() * rescale_std + rescale_mean
reconstruct_y_pred = y_pred * rescale_std + rescale_mean

if prop_name == 'HOMO (Ha)' or 'LUMO (Ha)':
    reconstruct_y_test *= 27.21138602
    reconstruct_y_pred *= 27.21138602
    prop_name_T = prop_name[:5] + '(eV)'
else:
    prop_name_T = prop_name

#### OVERALL R2 plot
ax = start_plot(style='darkgrid')
sns.regplot(reconstruct_y_test.reshape(-1), reconstruct_y_pred.reshape(-1))
ax.scatter(reconstruct_y_test.reshape(-1), reconstruct_y_pred.reshape(-1), color='darkorange', edgecolor='navy',
           label=r'$R^2:\quad %.4f$' % r2_score(reconstruct_y_test, reconstruct_y_pred) + '\n' +
                 r'$MAE: \quad %.4f$' % mean_absolute_error(reconstruct_y_test, reconstruct_y_pred))
ymin = min(np.min(reconstruct_y_test), np.min(reconstruct_y_pred)) - 0.1
ymax = max(np.max(reconstruct_y_test), np.max(reconstruct_y_pred)) + 0.1
lim = [ymin, ymax]
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.plot(lim, lim, c='brown', ls='--', label=r'$y=\hat y, identity$')
ax.legend(loc='best', frameon=True, shadow=True)
plt.xlabel('TRUE %s' % prop_name_T)
plt.ylabel('PREDICTED %s' % prop_name_T)
plt.title('%s Testing: %d' % (prop_name_T, len(test_graphs)))
plt.savefig('./%s/bag_sched_sd_explicit_H_R2.png' % prop_name)