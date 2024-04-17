from torch.nn import Parameter
from utils import *
import torch
import torch.nn as nn

import torch.optim as optim
from matplotlib.pyplot import show
from utility.fusion_functions import (train_nn_combiner_model,
                                      test_nn_combiner)
from definitions import (MODELS_TRAIN_OUTPUTS_FILE, MODELS_VAL_OUTPUTS_FILE,
                         MODELS_TEST_OUTPUTS_FILE,
                         BEST_COMBINER_MODEL)
from utility.utilities import FusionData, Features

def get_a(A):
    d = torch.Tensor([
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0.5]
    ])
    a = torch.Tensor([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    a = a.to(device)
    a = a + A
    d = d.to(device)
    a1 = torch.matmul(d, a)
    a2 = torch.matmul(a1, d)
    return a2

def get_A():
    d = torch.Tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    return d

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        adj = adj.to(device)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output



    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        # batch_size, head, length, d_tensor = k.size()
        length, d_tensor = k.size()
        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(0, 1)  # transpose
        # score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        score = (q @ k_t)
        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class GCNResnet(nn.Module):
    def __init__(self):
        super(GCNResnet, self).__init__()
        self.gc = GraphConvolution(10, 10)
        # self.gc2 = GraphConvolution(1024, 2048)
        # self.relu = nn.LeakyReLU(0.2)

        # _adj = gen_A(num_classes, t, adj_file)  # 这个t怎么设置
        # self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        # adj = gen_adj(self.A).detach()
        # self.linear1 = nn.Linear(30, 3)
        adj = get_A()
        self.A = Parameter(adj)
        self.linear1 = nn.Linear(30, 32)
        self.BarchNorm1d = nn.BatchNorm1d(32)
        # self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 10)
        self.avgpool = nn.AvgPool1d(2)
        #self.relu = nn.Softplus()
        self.relu = nn.ELU()
        self.transformer = ScaleDotProductAttention()

    def make_adj(self, x, a):
        adj = torch.empty((len(x), len(a[0]), len(a[1])), device=device)
        for i in range(len(x)):
            for j in range(len(a)):
                for k in range(len(a)):
                    adj[i][j][k] = a[j][k]

        return adj

    def forward(self, x):

        adj = get_a(self.A).detach()
        adj = self.make_adj(x, adj)
        x = self.gc(x, adj)
        # x = self.relu(x)
        # x = self.transformer(x, x, x)
        x = x.permute(0, 2, 1)
        x = self.avgpool(x)
        x = torch.squeeze(x, 2)
        v, score = self.transformer(x, x, x)
        # mini_length = len(x)
        # for i in range(mini_length):
        #     x1 = self.gc(x[i], adj)
        #     x[i] = x1
        #     print(i)




        # x = self.linear1(x)
        # x = self.BarchNorm1d(x)
        # x = self.relu(x)
        # x = self.linear2(x)

        # x = self.gc(x, adj)  # 3 * 10
        # x = x.flatten(start_dim=1)
        # # x = self.gc(x, adj)  # 出来的结果是3*10
        # x = self.relu(x)
        # x1 = x[0][0]*0.3 + x[0][1]*0.3 + x[0][2]*0.3
        return v



if __name__ == '__main__':
    run_train = True
    run_test = True

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    model_ensemble = GCNResnet()




    if run_train:
        optimizer = optim.Adam(model_ensemble.parameters(), lr=0.0001)
        #optimizer = optim.SGD(model_ensemble.parameters(), lr=0.00000005,momentum=0.8)
        epochs = 2000
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