#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 01:09:07 2020

@author: krishna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


"""


import torch.nn as nn
import argparse
import numpy as np
from torch import optim
from model1.audio_sf_network import Model
from model1.model_feature import Model_Feature
from DataGenerator import SpeechDataGenerator, collate_fn
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import classification_report
# from definitions import (TEST_CSV, TRAIN_CSV, AUDIO_SELFATTENTION_TRAIN_FEATURES_FILE,
#                          VAL_CSV, AUDIO_VGGISH_VAL_FEATURES_FILE,
#                          AUDIO_VGGISH_TEST_FEATURES_FILE,
#                          TAU_AUDIO_DIR, BEST_SELFATTENTION_MODEL)
torch.multiprocessing.set_sharing_strategy('file_system')



def arg_parser():
    ########## Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('-training_filepath',type=str,default='meta/train.txt')
    parser.add_argument('-valing_filepath', type=str, default='meta/val.txt')
    parser.add_argument('-testing_filepath', type=str, default='meta/test.txt')
    parser.add_argument('-label_filepath', type=str, default='meta/train_label.txt')
    parser.add_argument('-val_label_filepath', type=str, default='meta/val_label.txt')
    parser.add_argument('-test_label_filepath',type=str, default='meta/test_label.txt')
    parser.add_argument('-num_classes', action="store_true", default=10)
    parser.add_argument('-batch_size', action="store_true", default=1)
    parser.add_argument('-use_gpu', action="store_true", default=True)
    parser.add_argument('-num_epochs', action="store_true", default=100)
    parser.add_argument('-temp', action="store_true", default=0.2)
    
    return parser.parse_args()


def train(model, model_feature, data_loader, device, optimizer, criterion, epoch):
    model.train()
    total_loss =[]
    gt_labels = []
    pred_labels =[]
    X = np.empty((86120, 32), dtype=np.float32)
    j = 0
    counter = 0
    lowest_loss = 100
    wangic = 0
    for i_batch, sample_batched in enumerate(data_loader):
        wangic = wangic + 1
        print(wangic)
        features = torch.stack(sample_batched[0])
        # features.reshape(10*80000)
        embedding_batch = model_feature(features)
        embedding_batch = embedding_batch.detach().numpy()
        dim = embedding_batch.shape[0]
        X[counter:counter + dim] = embedding_batch
        counter += dim
        labels = torch.stack(sample_batched[1])
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(features)  # 这里是batchsize*num_class
        #现在y就是gt_label 然后把这个pred变成10*128 跟上个一样存到x中就结束了
        #既然是batchsize*num_class 那么这个pred就是预测的结果 那就不是特征了 那就要从上面的feature入手
        #上面的思路不对 只要这个保存的文件是由X和y组成的二维的npy文件就行了 然后主要是适配outputsaving里面的内容
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
        gt_labels = gt_labels + list(labels.detach().cpu().numpy())
        pred_labels = pred_labels + list(np.argmax(pred.detach().cpu().numpy(),axis=1))
        j += 1
        # print(j)


        if loss < lowest_loss:
            lowest_loss = loss
            torch.save(model.state_dict(), 'best_audio_selfattention_model.pkt')
    print("1-2")
    y = gt_labels
    np.savez_compressed('audio_selfattention_train_data.npz', X=X, y=y)
    mean_loss = np.mean(np.asarray(total_loss))
    print(f'Training loss {mean_loss} after {epoch} epochs') 
    
    target_names = ['airport', 'bus', 'metro', 'metro_station', 'park',
                    'public_square', 'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']
    print(classification_report(gt_labels, pred_labels, target_names=target_names, digits=4))
        
    
def evaluation(model, model_feature, data_loader, device, optimizer, criterion, epoch, num):
    model.eval()
    total_loss =[]
    gt_labels = []
    pred_labels =[]
    Z = np.empty((num, 32), dtype=np.float32)
    j = 0
    counter = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(data_loader):
            features = torch.stack(sample_batched[0])
            embedding_batch = model_feature(features)
            embedding_batch = embedding_batch.detach().numpy()
            dim = embedding_batch.shape[0]
            Z[counter:counter + dim] = embedding_batch
            counter += dim
            labels = torch.stack(sample_batched[1])
            features, labels = features.to(device), labels.to(device)
            pred = model(features)
            loss = criterion(pred, labels)
            total_loss.append(loss.item())
            gt_labels = gt_labels + list(labels.detach().cpu().numpy())
            pred_labels = pred_labels +list(np.argmax(pred.detach().cpu().numpy(),axis=1))
            # print('labels:', labels)
            # print('preds:', torch.argmax(outputs, dim=1))
            j += 1
            # print(j)

    z = gt_labels
    if num == 12470:
        np.savez_compressed('audio_selfattention_val_data.npz', X=Z, y=z)
    else:
        np.savez_compressed('audio_selfattention_test_data.npz', X=Z, y=z)

            
    mean_loss = np.mean(np.asarray(total_loss))
    print(f'Testing loss {mean_loss} after {epoch} epochs') 
    
    target_names = ['airport', 'bus', 'metro', 'metro_station', 'park',
                    'public_square', 'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']
    
    print(classification_report(gt_labels, pred_labels, target_names=target_names, digits=4))



        
    
    


def main():
    args = arg_parser()
    ### Data loaders
    dataset_train = SpeechDataGenerator(manifest=args.training_filepath, labellist=args.label_filepath)
    # dataset_train = SpeechDataGenerator(manifest=args.training_filepath)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    dataset_eval = SpeechDataGenerator(manifest=args.valing_filepath, labellist=args.val_label_filepath)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, collate_fn=collate_fn)

    dataset_test = SpeechDataGenerator(manifest=args.testing_filepath, labellist=args.test_label_filepath)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, collate_fn=collate_fn)
    ## Model related
    test_num = 24320
    val_num = 12470

    
    if args.use_gpu:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    else:
        device='cpu'
    model = Model(num_classes=args.num_classes)
    model_feature = Model_Feature(num_classes=args.num_classes)
    model = model.to(device)
    # torch.save(model.state_dict(), 'best_audio_selfattention_model.pkt')
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss()
    i = 0
    for epoch in range(1, args.num_epochs+1):
        print("1")
        train(model, model_feature, dataloader_train, device, optimizer, criterion, epoch)
        print("2")
        evaluation(model, model_feature, dataloader_eval, device, optimizer, criterion, epoch, val_num)
        print("3")
        evaluation(model, model_feature,  dataloader_test, device, optimizer, criterion, epoch, test_num)
        print("4")
        i = i+1
        print(i)

    
if __name__=='__main__':
    main()
