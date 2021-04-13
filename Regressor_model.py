import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import dgl
from dgl.nn.pytorch import GraphConv, NNConv
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda")

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

class GCN_graph_to_num_regressor(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN_graph_to_num_regressor, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size, allow_zero_in_degree=True).cuda()
        self.conv2 = GraphConv(hidden_size, hidden_size, allow_zero_in_degree=True).cuda()
        self.linear = nn.Linear(hidden_size, hidden_size).cuda()
        self.gru = nn.GRU(input_size = 10, hidden_size = hidden_size, num_layers = 1, batch_first = True).cuda()
        self.predict = nn.Linear(hidden_size, out_feats).cuda()
        self.hidden_size = hidden_size
    def forward(self, g):
        g = g.to(device)
        inputs = g.ndata['h']
        h = self.conv1(g, inputs)
        h = self.conv2(g, h)
        g.ndata['predict'] = h
        hg = dgl.mean_nodes(g, 'predict')
        h = F.leaky_relu(self.linear(hg), 0.25)
        h = h.unsqueeze(0)
        inputs = torch.zeros(h.shape[1], 50, 10).to(device)
        output, hn = self.gru(inputs, h)
        output = output.reshape(g.batch_size, -1)
        output = nn.Linear(output.shape[1], self.hidden_size).cuda()(output)
        output = self.linear(output)
        predict = F.leaky_relu(self.predict(output), 0.5)
        return h, predict

class Set2Set(nn.Module):

    def __init__(self, input_dim, n_iters, n_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = torch.nn.LSTM(input_size=self.output_dim, hidden_size=self.input_dim, num_layers=n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, graph, feat):
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                 feat.new_zeros((self.n_layers, batch_size, self.input_dim))) #(6, 32, 100)

            q_star = feat.new_zeros(batch_size, self.output_dim) #(32, 200)
            #print(q_star.shape)
            for i in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * dgl.broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)

                graph.ndata['e'] = e
                alpha = dgl.softmax_nodes(graph, 'e')
                graph.ndata['r'] = feat * alpha
                readout = dgl.sum_nodes(graph, 'r')
                q_star = torch.cat([q, readout], dim=-1)

            return q_star

class Mol2NumNet_regressor(nn.Module):
    def __init__(self, node_hidden_dim, edge_hidden_dim, n_classes, node_input_dim=23, edge_input_dim=19,
                 num_step_message_passing=6, num_step_set2set=6, num_layer_set2set=3):
        super(Mol2NumNet_regressor, self).__init__()
        self.n_classes = n_classes
        self.num_step_message_passing = num_step_message_passing
        self.lin_0 = nn.Linear(node_input_dim, node_hidden_dim)
        edge_net = nn.Sequential(nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
                                 nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv_2 = NNConv(in_feats=node_hidden_dim, out_feats=node_hidden_dim, edge_func=edge_net, aggregator_type="sum")
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim, num_layers=1, )
        nn.GRU()
        self.set2set = Set2Set(input_dim=node_hidden_dim, n_iters=num_step_set2set, n_layers=num_layer_set2set)
        self.lin2 = nn.Linear(node_hidden_dim, node_hidden_dim)
        self.lin3 = nn.Linear(2 * node_hidden_dim, 2 * node_hidden_dim)
        self.predict = nn.Linear(2 * node_hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, g, n_feat, e_feat):
        g = g.to(device)
        out = self.lin_0(n_feat)
        h = out.unsqueeze(0)
        for i in range(self.num_step_message_passing):
            m = self.conv_2(g, out, e_feat)
            m = self.dropout(m)
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        out = self.set2set(g, out)
        out = F.relu(self.lin3(out))
        predict = self.predict(out)
        return h, predict
