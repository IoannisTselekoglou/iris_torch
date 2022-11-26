import torch
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

class data_loader:

    # tensor[0:,1] = 4 to accses and sets: specific value 
    def accuracy(model,test_loader, batch_size: int):
        sum_acc= 0
        def transform_label(label_data):
            data = []
            for i in label_data:
                if i == "Iris-setosa":
                    data.append(torch.tensor([0]))
                if i == "Iris-versicolor":
                    data.append(torch.tensor([1]))
                if i == "Iris-virginica":
                    data.append(torch.tensor([2]))
            return torch.stack(data)

#        def final_x(x_hat):
#            list_tensors = []
#            for i in x_hat:
#                final_tensor = torch.zeros(1,3)
#                final_tensor[0:,i] = 1
#                list_tensors.append(final_tensor)
#            return torch.cat(list_tensors)
#
        for i,(X_test, test_labels) in enumerate(test_loader):
            test_labels = transform_label(test_labels)
            x_label_pre = model(X_test)
            _, x_label_pre_hat = torch.max(x_label_pre, 1)
            idx = 0
            number_pred = 0
            counter = 0
            #print(model_prediction, test_labels)
            while idx < len(X_test):
                if x_label_pre_hat[idx].item() == test_labels[idx].item():
                    number_pred += 1
                idx +=1
            accuracy_per_epoch = (number_pred/len(X_test))*100
            print(f"accuracy batch {i}:\n{accuracy_per_epoch}%")
            sum_acc += accuracy_per_epoch
        return print(f"\ntotal accuracy of model {(sum_acc/len(test_loader)):.2f}%")

