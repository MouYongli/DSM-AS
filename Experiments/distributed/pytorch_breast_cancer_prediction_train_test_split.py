import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configurations
SEED = 7
X_FEATURES = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
Y_FEATURE = 'diagnosis'

def randsplit(n,a,b,s):
    hit = False
    while not hit:
        total, count = 0,0
        nums = []
        while total < s and count < n:
            r = np.random.randint(a,b)
            total += r
            count += 1
            nums.append(r)
        if total == s and count == n: hit = True
    nums = [num / s for num in nums]
    return nums

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    def forward(self, x):
        out = self.linear(x)
        return out

# hyperparameters
num_classes = 2
num_epochs = 10000
learning_rate = 0.01

# Set up randomness seed for reproduction
np.random.seed(SEED)
# Input csv df
data = pd.read_csv('../input/data.csv')
# Data cleaning
list = ['Unnamed: 32', 'id']
data = data.drop(list, axis=1)
# Reassign the target: M=1, B=0
data.diagnosis.replace(to_replace=dict(M=1, B=0), inplace=True)
# Data train and test spliting
X_train, X_test, y_train, y_test = train_test_split(data[X_FEATURES], data[Y_FEATURE], test_size=0.2, random_state=5)
print("Num_X_train", X_train.shape[0], "Num_X_test", X_test.shape[0])

