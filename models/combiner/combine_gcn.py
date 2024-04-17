"""
Contains the trainable sub-network of an ensemble classifier.
Handles calling the training and evaluation.
"""

from torch import FloatTensor
from torch.nn.parameter import Parameter

from scipy.spatial.distance import pdist, squareform
import torch.optim as optim
import numpy as np
from matplotlib.pyplot import show

from utility.fusion_functions import (train_nn_combiner_model,
                                      test_nn_combiner)
from definitions import (MODELS_TRAIN_OUTPUTS_FILE, MODELS_VAL_OUTPUTS_FILE,
                         MODELS_TEST_OUTPUTS_FILE,
                         BEST_COMBINER_MODEL)
from utility.utilities import FusionData, Features
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
# from layers import GraphConvolution, SimilarityAdj, DistanceAdj
from torch.nn.modules.module import Module
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, residual=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        if not residual:
            self.residual = lambda x: 0
        elif (in_features == out_features):
            self.residual = lambda x: x
        else:
            # self.residual = linear(in_features, out_features)
            self.residual = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=5, padding=2)
    def reset_parameters(self):
        # stdv = 1. / sqrt(self.weight.size(1))
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.1)

    def forward(self, input, adj):
        # To support batch operations
        support = input.matmul(self.weight)
        output = adj.matmul(support)

        if self.bias is not None:
            output = output + self.bias
        if self.in_features != self.out_features and self.residual:
            input = input.permute(0,2,1)
            res = self.residual(input)
            res = res.permute(0,2,1)
            output = output + res
        else:
            output = output + self.residual(input)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SimilarityAdj(Module):

    def __init__(self, in_features, out_features):
        super(SimilarityAdj, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight0 = Parameter(FloatTensor(in_features, out_features))
        self.weight1 = Parameter(FloatTensor(in_features, out_features))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / sqrt(self.weight0.size(1))
        nn.init.xavier_uniform_(self.weight0)
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, input, seq_len):
        # To support batch operations
        soft = nn.Softmax(1)
        theta = torch.matmul(input, self.weight0)
        phi = torch.matmul(input, self.weight0)
        phi2 = phi.permute(0, 2, 1)
        sim_graph = torch.matmul(theta, phi2)

        theta_norm = torch.norm(theta, p=2, dim=2, keepdim=True)  # B*T*1
        phi_norm = torch.norm(phi, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = theta_norm.matmul(phi_norm.permute(0, 2, 1))
        sim_graph = sim_graph / (x_norm_x + 1e-20)

        output = torch.zeros_like(sim_graph)
        if seq_len is None:
            for i in range(sim_graph.shape[0]):
                tmp = sim_graph[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = sim_graph[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class DistanceAdj(Module):

    def __init__(self):
        super(DistanceAdj, self).__init__()
        self.sigma = Parameter(FloatTensor(1))
        self.sigma.data.fill_(0.1)

    def forward(self, batch_size, max_seqlen):
        # To support batch operations
        self.arith = np.arange(max_seqlen).reshape(-1, 1)
        dist = pdist(self.arith, metric='cityblock').astype(np.float32)
        self.dist = torch.from_numpy(squareform(dist)).to('cuda')
        self.dist = torch.exp(-self.dist / torch.exp(torch.tensor(1.)))
        self.dist = torch.unsqueeze(self.dist, 0).repeat(batch_size, 1, 1).to('cuda')
        return self.dist

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.1)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # n_features = args.feature_size
        n_class = 10

        # self.conv1d1 = nn.Conv1d(in_channels=25, out_channels=3, kernel_size=1, padding=0)
        # self.conv1d2 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, padding=0)
        # self.conv1d3 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=5, padding=2)
        # self.conv1d4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        # Graph Convolution
        self.gc1 = GraphConvolution(10, 10, residual=True)  # nn.Linear(128, 32)
        self.gc2 = GraphConvolution(10, 10, residual=True)
        self.gc3 = GraphConvolution(10, 10, residual=True)  # nn.Linear(128, 32)
        self.gc4 = GraphConvolution(10, 10, residual=True)
        self.gc5 = GraphConvolution(30, 32, residual=True)  # nn.Linear(128, 32)
        self.gc6 = GraphConvolution(32, 32, residual=True)
        self.simAdj = SimilarityAdj(30, 32)
        self.disAdj = DistanceAdj()

        self.classifier = nn.Linear(20, n_class)
        # self.approximator = nn.Sequential(nn.Conv1d(128, 64, 1, padding=0), nn.ReLU(),
        #                                   nn.Conv1d(64, 32, 1, padding=0), nn.ReLU())
        # self.conv1d_approximator = nn.Conv1d(32, 1, 5, padding=0)
        self.dropout = nn.Dropout(0.1)
        # self.dropout = nn.Dropout(0.6)  # Dropout 这个参数一般在0.1-0.3之间
        self.relu = nn.ReLU()
        self.SiLu = nn.SiLU()
        self.leakyReLu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.MaxPool1d(2)
        self.apply(weight_init)  # 遍历整个模型

        # self.transformer = torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6,
        #                                         num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
        #                                         activation=self.relu, custom_encoder=None, custom_decoder=None,
        #                                         layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)



    def forward(self, inputs, seq_len):
        adj = self.adj(inputs, seq_len)
        disadj = self.disAdj(inputs.shape[0], inputs.shape[1])
        x1_gc1 = self.gc1(inputs, adj)
        x1_h = self.relu(x1_gc1)
        x1_h = self.dropout(x1_h)
        x2_h = self.relu(self.gc3(inputs, disadj))
        x2_h = self.dropout(x2_h)
        x1 = self.relu(self.gc2(x1_h, adj))
        x1 = self.dropout(x1)
        # x1 = x1.permute(0, 2, 1)
        # x1 = self.avgpool(x1)
        # x1 = x1.permute(0, 2, 1)
        # x1 = torch.squeeze(x1, 1)
        x2 = self.relu(self.gc4(x2_h, disadj))
        x2 = self.dropout(x2)

        # 维度匹配
        # x2 = x2.permute(0, 2, 1)
        # x2 = self.avgpool(x2)
        # x2 = x2.permute(0, 2, 1)
        # x2 = torch.squeeze(x2, 1)
        x = torch.cat((x1, x2), 2)
        # x = self.transformer()
        x = x.permute(0, 2, 1)
        x = self.avgpool(x)
        x = x.permute(0, 2, 1)
        x = torch.squeeze(x, 1)
        x = self.classifier(x)
        return x

    def sadj(self, logits, seq_len):
        lens = logits.shape[1]
        soft = nn.Softmax(1)
        logits2 = self.sigmoid(logits).repeat(1, 1, lens)
        tmp = logits2.permute(0, 2, 1)
        adj = 1. - torch.abs(logits2 - tmp)
        self.sig = lambda x:1/(1+torch.exp(-((x-0.5))/0.1))
        adj = self.sig(adj)
        output = torch.zeros_like(adj)
        if seq_len is None:
            for i in range(logits.shape[0]):
                tmp = adj[i]
                adj2 = soft(tmp)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = adj[i, :seq_len[i], :seq_len[i]]
                adj2 = soft(tmp)
                output[i, :seq_len[i], :seq_len[i]] = adj2
        return output


    def adj(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0,2,1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0,2,1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output



if __name__ == '__main__':
    run_train = True
    run_test = True

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    model_ensemble = Model()

    if run_train:
        optimizer = optim.Adam(model_ensemble.parameters(), lr=0.000020)
        epochs = 100
        batch_size = 25
        feature_train = Features(feature_file=MODELS_TRAIN_OUTPUTS_FILE,
                                 arg_names=['X', 'y'])
        data_train = FusionData(features=[feature_train], do_reshape=False)
        feature_val = Features(feature_file=MODELS_VAL_OUTPUTS_FILE,
                               arg_names=['X', 'y'])
        data_val = FusionData(features=[feature_val], do_reshape=False)
        train_nn_combiner_model(model=model_ensemble,
                                optimizer=optimizer,
                                train_data=data_train.get_data(),
                                val_data=data_val.get_data(),
                                best_model=BEST_COMBINER_MODEL,
                                device=device,
                                epochs=epochs,
                                batch_size=batch_size)

    if run_test:
        feature_test = Features(feature_file=MODELS_TEST_OUTPUTS_FILE,
                                arg_names=['X', 'y'])
        data_test = FusionData(features=[feature_test], do_reshape=False)
        model_ensemble.load_state_dict(torch.load(BEST_COMBINER_MODEL))
        model_ensemble.eval()
        test_nn_combiner(model=model_ensemble,
                         test_data=data_test.get_data(),
                         device=device,
                         verbose=True)

    show()

"""
with logits, lr=0.000015, test: 0.872 (epoch 87) batchnormed, 0.871 without
with probabilities, lr=0.000025, test: 0.872 (epoch 46) batchnormed, 0.849 without
with softmax of probabilities, lr=0.000025 test: 0.872 (epoch 46) batchnormed, 0.814 without higher lr
with softmax of logits, lr=0.000020, 0.865, (epoch 46) batchnormed,
with logits:
test: 0.872     val: 0.892     val_loss: 0.2937   avg: 86.61%	   test_loss: 0.3840
optimizer = optim.Adam(model_ensemble.parameters(), lr=0.000015)
epochs = 92 # (87), 0.871 without batchnorm
batch_size = 32
self.model = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.Dropout(0.5),

            nn.Linear(32, 10)
        )
"""
