#-*- coding: utf-8 -*-
from os import path
import argparse
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from sklearn.metrics import accuracy_score
import time

#多層化したモデルの定義(隠れ層を追加)
class Multipled_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Multipled_model, self).__init__()
        #Linearは全結合層を示す
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    #forward pass
    def forward(self, x):
        x = F.relu(self.fc1(x))
        logit = self.fc2(x)
        return logit

from sklearn.metrics import accuracy_score

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
    train_iter = torch.utils.data.DataLoader(train_dataset,shuffle = True)
    valid_iter = torch.utils.data.DataLoader(valid_dataset)
    model = Multipled_model(input_dim=300, hidden_dim=100,output_dim=4)
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with open(log_file, "w") as f_log:
        f_log.write("{}\t{}\t{}\t{}\n".format("train_loss", "train_acc", "valid_loss", "valid_acc"))
        # 訓練
        print("{}\t{}\t{}\t{}\t{}".format("Epoch", "Train_loss", "Train_acc", "Valid_loss", "Valid_acc"))
        for epoch in range(1, num_epoch+1):

            for x, y in train_iter:
                optimizer.zero_grad()
                logit = model(x)
                loss = loss_func(logit, y)
                loss.backward()
                optimizer.step()

            #評価
            train_loss, train_acc = calc_loss_and_accuracy(model, loss_func, train_iter)
            valid_loss, valid_acc = calc_loss_and_accuracy(model, loss_func, valid_iter)
            if epoch % 10 ==0:
                print("Epoch:{}\t{}\t{}\t{}\t{}".format(epoch, train_loss, train_acc, valid_loss, valid_acc))
            f_log.write("{}\t{}\t{}\t{}\n".format(train_loss, train_acc, valid_loss, valid_acc))

        #model.state_dictメソッドでモデルのパラメータを保存できる
        #optimizer.state_dict()→内部状態を保存するメソッド
        torch.save({'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict()} , output_file)

train(256, 100, 0.01, "./work/train_x.npy", "./work/train_y.npy", "./work/valid_x.npy", "./work/valid_y.npy", "./work/model79.pt", "./work/log79.txt")

#parser = argparse.ArgumentParser(description='Process some integers.')

def main():
    #args = parser.parse_args()

    batch_size = 256
    num_epoch = 100
    learning_rate = 0.01
    train_x_file = "./work/train_x.npy"
    train_y_file = "./work/train_y.npy"
    valid_x_file = "./work/valid_x.npy"
    valid_y_file = "./work/valid_y.npy"
    utput_file = "./work/model79.pt"
    log_file = "./work/log79.txt"

    train(batch_size, num_epoch, learning_rate, train_x_file, train_y_file, valid_x_file, valid_y_file, output_file, log_file)

if __name__ == '__main__':
    main()