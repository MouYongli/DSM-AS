import numpy as np
import pandas as pd
from sklearn.utils import shuffle

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
    return nums

seed = 7
client_names = ["UKA", "UKG", "UKK", "UKL", "IMISE", "Mittweida"]
num_clients = 6
assert len(client_names) == num_clients
min_rate = 0.2
max_rate = 1.8
# Set up randomness seed for reproduction
np.random.seed(seed)
# Input csv df
df = pd.read_csv('./results/data.csv')
df = shuffle(df)

train_sample_num = randsplit(num_clients, np.floor(min_rate * df.shape[0] / num_clients), np.floor(max_rate * df.shape[0] / num_clients), df.shape[0])
train_sample_num[-1] = df.shape[0] - np.sum(train_sample_num[:-1])
train_start_idx = [int(sum(train_sample_num[:i - num_clients])) for i in range(num_clients)]
train_end_idx = [int(sum(train_sample_num[:i]))for i in range(1, num_clients + 1)]
train_end_idx[-1] = df.shape[0]

train_dfs = [df[["patient_id"]].iloc[train_start_idx[client_idx]:train_end_idx[client_idx]] for client_idx in
            range(num_clients)]

for idx in range(num_clients):
    train_dfs[idx].to_csv("./results/{}.csv".format(client_names[idx].lower()), index=False)

