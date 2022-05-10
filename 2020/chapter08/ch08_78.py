#-*- coding: utf-8 -*-
from os import path
import argparse
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from sklearn.metrics import accuracy_score
import time



class Model(nn.Module):
    def __init__(self, input_dim, num_class):
        super(Model, self).__init__()
        self.l1 = nn.Linear(input_dim, num_class, bias=False)

    def forward(self, x):
        logit = self.l1(x)
        return logit

def calc_loss_and_accuracy(model, loss_func, dataset):
    softmax = nn.Softmax(dim=-1)
    gold_labels = []
    pred_labels = []
    total_loss = 0

    for x, y in dataset:
        logit = model.forward(x)
        loss = loss_func(logit, y)
        total_loss += loss.item()
        pred_label = torch.argmax(softmax(logit), dim=-1)
        pred_labels.append(pred_label)
        gold_labels.append(y)
    ave_loss = total_loss / len(dataset)
    accuracy = accuracy_score(gold_labels, pred_labels)

    return ave_loss, accuracy

def train(batch_size, num_epoch, learning_rate, train_x_file, train_y_file, valid_x_file, valid_y_file, output_file, log_file):
    train_x = np.load(train_x_file)
    train_y = np.load(train_y_file)
    valid_x = np.load(valid_x_file)
    valid_y = np.load(valid_y_file)
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).long()
    valid_x = torch.from_numpy(valid_x).float()
    valid_y = torch.from_numpy(valid_y).long()

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    valid_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_iter = torch.utils.data.DataLoader(valid_dataset)
    model = Model(input_dim=300, num_class=4)
    loss_func = nn.CrossEntropyLoss(reduction='mean')


    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    with open(log_file, "w") as f_log:
        f_log.write("{}\n".format("batch_size"))
        #訓練
        for epoch in range(1, num_epoch+1):

            #時間を記録
            start = time.time()
            for x, y in train_iter:
                optimizer.zero_grad()
                logit = model.forward(x)
                loss = loss_func(logit, y)
                loss.backward()
                optimizer.step()

            #終了時刻
            end = time.time()
            total_time = end - start
            print("Batch_size:{}\t{}".format(batch_size, total_time))
            f_log.write("{}\t{}\n".format(batch_size, total_time))

        #model.state_dictメソッドでモデルのパラメータを保存できる
        #optimizer.state_dict()→内部状態を保存するメソッド
        torch.save({'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict()} , output_file)



parser = argparse.ArgumentParser(description='Process some integers.')

#def main():

if __name__ == "__main__":
    args = parser.parse_args()
    num_epoch = 1
    learning_rate = 0.01
    train_x_file = "work/train_x.npy"
    train_y_file = "work/train_y.npy"
    valid_x_file = "work/valid_x.npy"
    valid_y_file = "work/valid_y.npy"
    utput_file = "work/model78.pt"
    log_file = "work/log78.txt"

    #バッチサイズを指定
    for batch_size in [ 2 ** i for i in range(10)]:
        train(batch_size,num_epoch, learning_rate, train_x_file, train_y_file, valid_x_file, valid_y_file, output_file, log_file)
