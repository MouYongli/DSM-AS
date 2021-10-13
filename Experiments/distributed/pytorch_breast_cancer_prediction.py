import os
import os.path as osp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
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


def randsplit(n, a, b, s):
    hit = False
    while not hit:
        total, count = 0, 0
        nums = []
        while total < s and count < n:
            r = np.random.randint(a, b)
            total += r
            count += 1
            nums.append(r)
        if total == s and count == n: hit = True
    return nums


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


# hyperparameters
num_clients = 6
num_classes = 2
learning_rate = 0.01
weight_decay = 0.0005
min_rate = 0.2
max_rate = 1.8
dim_features = 30
num_epochs = 100
num_cycles = 100
num_hops = num_cycles * num_clients

station_name_list = ["ukleipzig", "ukaachen", "ukgoettingen", "imse", "rwth", "ukcologne"]

## Set up randomness seed for reproduction

np.random.seed(SEED)
cuda = torch.cuda.is_available()
torch.manual_seed(SEED)
if cuda:
    torch.cuda.manual_seed(SEED)
## Input csv df
data = pd.read_csv('../input/data.csv')
## Data cleaning
list = ['Unnamed: 32', 'id']
data = data.drop(list, axis=1)
## Reassign the target: M=1, B=0
## Data distribution
data = shuffle(data)
train_sample_num = randsplit(num_clients, np.floor(min_rate * data.shape[0] / num_clients),
                             np.floor(max_rate * data.shape[0] / num_clients), data.shape[0])
train_sample_num[-1] = data.shape[0] - np.sum(train_sample_num[:-1])
train_start_idx = [int(sum(train_sample_num[:i - num_clients])) for i in range(num_clients)]
train_end_idx = [int(sum(train_sample_num[:i])) for i in range(1, num_clients + 1)]
train_end_idx[-1] = data.shape[0]
print("Data quantity distribution: ", train_sample_num)
X_trains = [data[X_FEATURES].iloc[train_start_idx[client_idx]:train_end_idx[client_idx]] for client_idx in
            range(num_clients)]
y_trains = [data[Y_FEATURE].iloc[train_start_idx[client_idx]:train_end_idx[client_idx]] for client_idx in
            range(num_clients)]

## Normalization of the df
df_mean_X = []
df_std_X = []
df_y_counts = []
for client_idx in range(num_clients):
    scaler = StandardScaler()
    print("======================================================================")
    print("Client {}".format(client_idx))
    df_mean_X.append(np.mean(X_trains[client_idx][X_FEATURES], axis=0).to_numpy())
    df_std_X.append(np.std(X_trains[client_idx][X_FEATURES], axis=0).to_numpy())
    B, M = y_trains[client_idx].value_counts()
    df_y_counts.append(np.array([B, M]))
    print('Number of Benign: ', B)
    print('Number of Malignant : ', M)
    X_trains[client_idx] = scaler.fit_transform(X_trains[client_idx])
    y_trains[client_idx].replace(to_replace=dict(M=1, B=0), inplace=True)
    y_trains[client_idx] = y_trains[client_idx].to_numpy()

df_mean_X = pd.DataFrame(df_mean_X, columns=X_FEATURES)
df_std_X = pd.DataFrame(df_std_X, columns=X_FEATURES)
df_y_counts = pd.DataFrame(df_y_counts, columns=["B", "M"])

# df_y_counts.plot.bar(rot=0)
# plt.show()

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
    return val_loss.item(), (tp + tn) / (tp + fp + fn + tn), tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fn + fp)

#################################################################
## Single site training
#################################################################
single_site_training_dir = './results/single_site_training'
if not os.path.exists(single_site_training_dir):
    os.makedirs(single_site_training_dir)


criterion = nn.CrossEntropyLoss()
models = [LogisticRegression() for _ in range(num_clients)]
optims = [torch.optim.SGD(models[i].parameters(), lr=learning_rate, weight_decay=weight_decay) for i in range(num_clients)]
for client_idx in range(num_clients):
    if not osp.exists(osp.join(single_site_training_dir, 'log_{}.csv'.format(client_idx))):
        with open(osp.join(single_site_training_dir, 'log_{}.csv'.format(client_idx)), 'w') as f:
            f.write(','.join(["epoch", "train_loss", "client_id", "val_acc", "val_prec", "val_rec", "val_f1"]) + '\n')
    for epoch in range(num_epochs):
        perm = np.arange(X_trains[client_idx].shape[0])
        np.random.shuffle(perm)
        X_train = X_trains[client_idx][perm]
        y_train = X_trains[client_idx][perm]
        loss = train(X_train, y_train, models[client_idx], criterion, optims[client_idx])
        with open(osp.join(single_site_training_dir, 'log_{}.csv'.format(client_idx)), 'a') as f:
            log = map(str, [epoch, loss]+[''] * 5)
            f.write(','.join(log) + '\n')
        for val_client_id in range(num_clients):

            val_loss, val_acc, val_precesion, val_recall, val_f1_score = valid(X_train, y_train, models[client_idx], criterion)



# loss_list = []
# val_loss_list = []
# val_acc_list = []
# for epoch in range(num_epochs):
#     perm = np.arange(X_train.shape[0])
#     np.random.shuffle(perm)
#     X_train = X_train[perm]
#     y_train = y_train[perm]
#     loss = train(X_train, y_train)
#     val_loss, val_acc = valid(X_test, y_test)
#     if epoch % 1000 == 0:
#         print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f'
#               % (epoch, loss, val_loss, val_acc))
#     # logging
#     loss_list.append(loss)
#     val_loss_list.append(val_loss)
#     val_acc_list.append(val_acc)
# # plot learning curve
# plt.figure()
# plt.plot(range(num_epochs), loss_list, 'r-', label='train_loss')
# plt.plot(range(num_epochs), val_loss_list, 'b-', label='val_loss')
# plt.legend()
# plt.figure()
# plt.plot(range(num_epochs), val_acc_list, 'g-', label='val_acc')
# plt.legend()
# plt.show()