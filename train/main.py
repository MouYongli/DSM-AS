import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Configurations
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

station_name = "breugel"
mode = "train"

seed = 7
num_epochs = 10000
input_dim = 30
hidden_dim = 64
num_classes = 2
learning_rate = 0.01
station_name = station_name.lower()
mode = mode.lower()



class LogisticRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.output_layer = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out = self.output_layer(self.sigmoid(self.hidden_layer(x)))
        return out

def train(X_train, y_train, model, criterion, optimizer):
    inputs = torch.from_numpy(X_train).float()
    targets = torch.from_numpy(y_train).long()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def valid(X_test, y_test, model, criterion):
    inputs = torch.from_numpy(X_test).float()
    targets = torch.from_numpy(y_test).long()
    outputs = model(inputs)
    val_loss = criterion(outputs, targets)
    _, predicted = torch.max(outputs, 1)
    cm = confusion_matrix(targets.numpy(), predicted.numpy())
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return val_loss.item(), (tp+tn)/(tp+fp+fn+tn), tp/(tp+fp), tp/(tp+fn), 2*tp/(2*tp+fn+fp)

np.random.seed(seed)
torch.manual_seed(seed)
# Input csv df
data = pd.read_csv('../input/data.csv')
data = data.drop(['Unnamed: 32', 'id'], axis=1)
data.diagnosis.replace(to_replace=dict(M=1, B=0), inplace=True)



X_train = data[X_FEATURES]
y_train = data[Y_FEATURE]
scaler = StandardScaler()
print("Mean: ", np.mean(X_train, axis=0).to_numpy())
print("Std: ", np.std(X_train, axis=0).to_numpy())
B, M = y_train.value_counts()
print('Number of Benign: ', B)
print('Number of Malignant : ', M)
X_train = scaler.fit_transform(X_train)
y_train = y_train.to_numpy()

model = LogisticRegression(input_dim, hidden_dim, num_classes)
if osp.exists("./model.pth.tar"):
    checkpoint = torch.load("./model.pth.tar")
    model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# loss_list = []
# val_loss_list = []
# val_acc_list = []
# val_precision_list = []
# val_recall_list = []
# val_f1_score_list = []
for epoch in range(num_epochs):
    perm = np.arange(X_train.shape[0])
    np.random.shuffle(perm)
    X_train = X_train[perm]
    y_train = y_train[perm]
    loss = train(X_train, y_train, model, criterion, optimizer)
    val_loss, accuracy, precesion, recall, f1_score = valid(X_train, y_train, model, criterion)
    if epoch % 1000 == 0:
        print('==================================================================')
        print('Epoch', epoch)
        # print('Train loss: ', loss)
        # print('Val loss: ', val_loss)
        print('Val acc : ', accuracy)
        # print('Val precesion : ', precesion)
        # print('Val recall : ', recall)
        # print('Val f1_score : ', f1_score)


