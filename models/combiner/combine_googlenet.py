"""
Contains the trainable sub-network of an ensemble classifier.
Handles calling the training and evaluation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from matplotlib.pyplot import show

from utility.fusion_functions import (train_nn_combiner_model,
                                      test_nn_combiner)
from definitions import (MODELS_TRAIN_OUTPUTS_FILE, MODELS_VAL_OUTPUTS_FILE,
                         MODELS_TEST_OUTPUTS_FILE,
                         BEST_COMBINER_MODEL)
from utility.utilities import FusionData, Features


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits


        self.aux1 = InceptionAux(512, num_classes)
        self.aux2 = InceptionAux(528, num_classes)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        aux1 = self.aux1(x)
        aux2 = self.aux2(x)

        x = torch.flatten(x, 1)
        # N x 1024

        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux2, aux1
        # return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# class Inception(nn.Module):
#     def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
#         super(Inception, self).__init__()
#
#         self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
#
#         self.branch2 = nn.Sequential(
#             BasicConv2d(in_channels, ch3x3red, kernel_size=1),
#             BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
#         )
#
#         self.branch3 = nn.Sequential(
#             BasicConv2d(in_channels, ch5x5red, kernel_size=1),
#             # 在官方的实现中，其实是3x3的kernel并不是5x5，这里我也懒得改了，具体可以参考下面的issue
#             # Please see https://github.com/pytorch/vision/issues/906 for details.
#             BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
#         )
#
#         self.branch4 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#             BasicConv2d(in_channels, pool_proj, kernel_size=1)
#         )
#
#     def forward(self, x):
#         branch1 = self.branch1(x)
#         branch2 = self.branch2(x)
#         branch3 = self.branch3(x)
#         branch4 = self.branch4(x)
#
#         outputs = [branch1, branch2, branch3, branch4]
#         return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        # x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    run_train = True
    run_test = True

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    model_ensemble = GoogLeNet()

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
