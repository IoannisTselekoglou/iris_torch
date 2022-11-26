import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math



dataloader = DataLoader(dataset="./data_set.csv",batch_size=4,shuffle=True)


dataiter = iter(dataloader)


print(next(dataiter),next(dataiter),next(dataiter),next(dataiter))
