import os
import os.path as osp
import numpy as np
import pandas as pd
import shutil
import datetime
import pytz
import torch
import torch.nn as nn
from fhirpy import SyncFHIRClient

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

## Get the env vars from station software
fhir_server = str(os.environ['FHIR_SERVER'])
fhir_port = str(os.environ['FHIR_PORT'])
station_name = str(os.environ['STATION_NAME'])
# TODO: comment out when deploy
# fhir_server, fhir_port = "137.226.232.119", "8080"
# station_name = "UKA"

assert station_name in ["UKA", "UKG", "UKK", "UKL", "IMISE", "Mittweida"]
# Configurations
seed = 7
num_epochs = 10000
input_dim = 30
hidden_dim = 64
num_classes = 2
learning_rate = 0.01
weight_decay = 0.0005
station_name = station_name.lower()

# Define directory of output
here = osp.dirname(osp.abspath(__file__))
out_dir = osp.join(here, 'output')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

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
    with np.errstate(divide='ignore', invalid='ignore'):
        val_acc = (tp+tn)/(tp+fp+fn+tn)
        val_precesion = tp/(tp+fp)
        val_recall = tp/(tp+fn)
        val_f1_score = 2*tp/(2*tp+fn+fp)
    return val_loss.item(), val_acc, val_precesion, val_recall, val_f1_score

np.random.seed(seed)
cuda = torch.cuda.is_available()
torch.manual_seed(1337)
if cuda:
    torch.cuda.manual_seed(1337)

"""
@Laurenz, please replace here with your code for data extraction (FHIR -> DataFrame)
"""
###############################################################################################
# data = pd.read_csv('../input/data.csv')
# data = data.drop(['Unnamed: 32', 'id'], axis=1)
###############################################################################################
X_FEATURES = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave.points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave.points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave.points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
Y_FEATURE = 'label'

client = SyncFHIRClient('http://{}:{}/fhir'.format(fhir_server, fhir_port))
patients = client.resources('Patient')  # Return lazy search set
patients_data = []
for patient in patients:
    patient_birthDate = None
    try:
        patient_birthDate = patient.birthDate
    except:
        pass
    patients_data.append([patient.id, patient.gender, patient_birthDate])
patients_df = pd.DataFrame(patients_data, columns=["patient_id", "gender", "birthDate"])
patients_observation = {}
observations = client.resources("Observation").include("Patient", "subject")
for observation in observations:
    try:
        feature = observation["category"][0]["coding"][0]["code"]
        if feature in X_FEATURES:
            value = observation["valueQuantity"]["value"]
            patient_id_str = observation["subject"]["reference"]
            if patient_id_str[:7] == "Patient":
                patient_id = patient_id_str[8:]
                if patient_id not in patients_observation:
                    patients_observation[patient_id] = {}
                patients_observation[patient_id][feature] = float(value)
    except KeyError:
        print("Key error encountered, skipping Observation...")
for k in patients_observation.keys():
    patients_observation[k].update(patient_id=k)
observation_df = pd.DataFrame.from_dict(patients_observation.values())
observation_df.set_index(["patient_id"])
patients_condition = []
conditions = client.resources("Condition")
for condition in conditions:
    try:
        label = condition["code"]["coding"][0]["code"]
        patient_id_str = condition["subject"]["reference"]
        if patient_id_str[:7] == "Patient":
            patient_id = patient_id_str[8:]
            patients_condition.append([patient_id, label])
    except KeyError:
        print("Key error encountered, skipping Condition...")
condition_df = pd.DataFrame(patients_condition, columns=["patient_id", "label"])
data = pd.merge(pd.read_csv("./input/{}.csv".format(station_name)), pd.merge(pd.merge(patients_df, observation_df, on="patient_id", how="outer"), condition_df, on="patient_id", how="outer"), on="patient_id", how="inner")
data['label'] = data['label'].map(lambda x: "B" if x == "non-cancer" else "M")
print(data.head(5))
"""
@Toralf, please add the code for enlarging the dataset size.
"""
###############################################################################################

###############################################################################################
## Exploratory data analysis (EDA) and data normalization
X_train = data[X_FEATURES]
y_train = data[Y_FEATURE]
scaler = StandardScaler()
X_mean = np.mean(X_train, axis=0).to_numpy()
X_std = np.std(X_train, axis=0).to_numpy()
print("Mean: ", X_mean)
print("Std: ", X_std)
B, M = y_train.value_counts()
print('Number of Benign: ', B)
print('Number of Malignant : ', M)
X_train = scaler.fit_transform(X_train)
y_train.replace(to_replace=dict(M=1, B=0), inplace=True)
y_train = y_train.to_numpy()
## Saving statistical data for visualization
stat_dir = os.path.join(out_dir, 'stat', station_name)
if not os.path.exists(stat_dir):
    os.makedirs(stat_dir)
df_X_mean = pd.DataFrame([X_mean], columns=X_FEATURES)
df_X_std = pd.DataFrame([X_std], columns=X_FEATURES)
df_y_dist = pd.DataFrame([[B, M]], columns=['B','M'])
df_X_mean.to_csv(osp.join(stat_dir, 'X_mean.csv'))
df_X_std.to_csv(osp.join(stat_dir, 'X_std.csv'))
df_y_dist.to_csv(osp.join(stat_dir, 'X_std.csv'))

model = LogisticRegression(input_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
best_acc = 0.0

model_dir = os.path.join(out_dir, 'models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    torch.save({
        'epoch': -1,
        'optim_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'best_acc': 0.0,
    }, osp.join(model_dir, 'dnn.pth.tar'))
else:
    checkpoint = torch.load(osp.join(model_dir, "dnn.pth.tar"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    best_acc = checkpoint['best_acc']

training_dir = osp.join(out_dir, 'training')
if not os.path.exists(training_dir):
    os.makedirs(training_dir)
if not osp.exists(osp.join(training_dir, 'log.csv')):
    with open(osp.join(training_dir, 'log.csv'), 'w') as f:
        f.write(','.join(["epoch", "station_name", "train_loss", "val_loss", "val_acc", "val_prec", "val_rec", "val_f1", "time"]) + '\n')

for epoch in range(num_epochs):
    perm = np.arange(X_train.shape[0])
    np.random.shuffle(perm)
    X_train = X_train[perm]
    y_train = y_train[perm]
    loss = train(X_train, y_train, model, criterion, optimizer)
    val_loss, val_acc, val_precesion, val_recall, val_f1_score = valid(X_train, y_train, model, criterion)
    with open(osp.join(training_dir, 'log.csv'), 'a') as f:
        log = map(str, [epoch, station_name, loss, val_loss, val_acc, val_precesion, val_recall, val_f1_score, str(datetime.datetime.now(pytz.timezone('Europe/Berlin')))])
        f.write(','.join(log) + '\n')
    torch.save({
        'epoch': epoch,
        'optim_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'best_acc': val_acc,
    }, osp.join(model_dir, 'checkpoint.pth.tar'))
    if val_acc > best_acc:
        shutil.copy(osp.join(model_dir, 'checkpoint.pth.tar'),
                    osp.join(model_dir, 'dnn.pth.tar'))


# loss_list = []
# val_loss_list = []
# val_acc_list = []
# val_precision_list = []
# val_recall_list = []
# val_f1_score_list = []
# for epoch in range(num_epochs):
#     perm = np.arange(X_train.shape[0])
#     np.random.shuffle(perm)
#     X_train = X_train[perm]
#     y_train = y_train[perm]
#     loss = train(X_train, y_train, model, criterion, optimizer)
#     val_loss, val_acc, val_precesion, val_recall, val_f1_score = valid(X_train, y_train, model, criterion)
#     loss_list.append(loss)
#     val_loss_list.append(val_loss)
#     val_acc_list.append(val_acc)
#     val_precision_list.append(val_precesion)
#     val_recall_list.append(val_recall)
#     val_f1_score_list.append(val_f1_score)
#     if epoch % 1000 == 0:
#         print('==================================================================')
#         print('Epoch', epoch)
#         print('Train loss: ', loss)
#         print('Val loss: ', val_loss)
#         print('Val acc : ', val_acc)
#         print('Val precesion : ', val_precesion)
#         print('Val recall : ', val_recall)
#         print('Val f1_score : ', val_f1_score)
# plot learning curve
# plt.figure()
# plt.plot(range(num_epochs), loss_list, 'r-', label='train_loss')
# plt.plot(range(num_epochs), val_loss_list, 'b-', label='val_loss')
# plt.plot(range(num_epochs), val_acc_list, 'g-', label='val_acc')
# plt.plot(range(num_epochs), val_recall_list, 'r+', label='val_recall')
# plt.plot(range(num_epochs), val_f1_score_list, 'g+', label='val_f1_score')
# plt.legend()
# plt.show()


