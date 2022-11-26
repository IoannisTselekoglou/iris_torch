import torch
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

class data_loader:

    # tensor[0:,1] = 4 to accses specific value 
    def accuracy(model, test_dataset, batch_size: int):
        summe = 0
        def transform_label(label_data):
            data = []
            for i in label_data:
                if i == "Iris-setosa":
                    data.append(torch.tensor([1,0,0], dtype=torch.float32))
                elif i == "Iris-versicolor": 
                    data.append(torch.tensor([0,1,0], dtype=torch.float32))
                else:
                    data.append(torch.tensor([0,0,1], dtype=torch.float32))
            return torch.stack(data)

        def final_x(x_hat):
            list_tensors = []
            for i in x_hat:
                final_tensor = torch.zeros(1,3)
                final_tensor[0:,i] = 1
                list_tensors.append(final_tensor)
            return torch.cat(list_tensors)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        for i,(X_test, test_labels) in enumerate(test_loader):
            test_labels = transform_label(test_labels)
            x_label_pre = model(X_test)
            _, x_label_pre_hat = torch.max(x_label_pre, 1)
            model_prediction = final_x(x_label_pre_hat)
            idx = 0
            number_pred = 0
            counter = 0
            while idx < 10:
                print(f" \n Tensor[{idx+1}]")
                for i in range(3):
                    print(f"model: {model_prediction[idx][i].item()}, label: {test_labels[idx][i].item()}")
                    if model_prediction[idx][i].item() == test_labels[idx][i].item():
                        counter += 1
                        if counter == 3:
                            number_pred += 1
                    counter = 0
                idx +=1
            accuracy_per_epoch = number_pred/10 *100
        summe += accuracy_per_epoch
        return print(f"\naccuracy of model {summe/iters}%")

