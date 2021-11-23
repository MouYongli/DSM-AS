import os
import os.path as osp
import sys
from random import randrange
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
from keras.layers import Dense, Flatten, Embedding, Multiply, Concatenate, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import adam_v2
import csv
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# for the data generation part there are two variable
# 1. is for if data generation method should be used or not
# 2. is for to handle outliners like minus values which could be appear
# the generated data can hold negative value, this value are biological not explainable, but from a informatics point of
# view they are reasonable
# so the station owner can decide if the methode should remove the "false" value

# Get the env vars from station software
# fhir_server = str(os.environ['FHIR_SERVER'])
# fhir_port = str(os.environ['FHIR_PORT'])
# station_name = str(os.environ['STATION_NAME'])
# generation_synthetic_data = os.environ['GENERATION_SYNTHETIC_DATA']
# autoremove = os.environ['AUTOREMOVE']

# TODO: check if the enviroment variables can hold the item for synthetic-data generation and autoremove

# TODO: comment out when deploy
fhir_server, fhir_port = "10.50.9.90", "80"
station_name = "UKL"
generation_synthetic_data = False
# autoremove = False

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

# values for the build of the Discriminator and Generator of the cGAN
# if we need to change them, we could adjust them here and not in every definition
img_rows = 1
img_cols = 31

img_shape = (img_rows, img_cols)

zdim = 100
num_classes = 2


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
        val_acc = (tp + tn) / (tp + fp + fn + tn)
        val_ppv = tp / (tp + fp)
        val_precesion = tp / (tp + fp)
        val_recall = tp / (tp + fn)
        val_f1_score = 2 * tp / (2 * tp + fn + fp)
    return val_loss.item(), val_acc, val_precesion, val_recall, val_f1_score, val_ppv


def shuffle_list(lst):
    lst2 = lst.copy()
    random.shuffle(lst2)
    return lst2


def train_synthetic(data, iterations, batch_size, interval, number_of_row):
    Xtrainnew = data
    mydata = Xtrainnew.values.tolist()
    ytrain = []
    for j in mydata:
        ytrain.append(j[0])
    Xtrainnew = pd.DataFrame(data=mydata)
    Ytrainnew = np.array(ytrain)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(Xtrainnew)
    Xtrain = scaled
    ytrain = Ytrainnew
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        ids = np.random.randint(0, Xtrain.shape[0], batch_size)
        imgs = Xtrain[ids]
        labels = ytrain[ids]

        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = gen_v.predict([z, labels])

        dloss_real = dis_v.train_on_batch([imgs, labels], real)
        dloss_fake = dis_v.train_on_batch([gen_imgs, labels], fake)

        dloss, accuracy = 0.5 * np.add(dloss_real, dloss_fake)

        z = np.random.normal(0, 1, (batch_size, 100))
        labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)
        gloss = gan_v.train_on_batch([z, labels], real)

        if (iteration + 1) % interval == 0:
            losses.append((dloss, gloss))
            accuracies.append(100.0 * accuracy)
            iteration_checks.append(iteration + 1)

            print("%d [D loss: %f , acc: %.2f] [G loss: %f]" %
                  (iteration + 1, dloss, 100.0 * accuracy, gloss))
    return show_data(gen_v, scaler, number_of_row)


def savelist2csv(mynamefile, mylist):
    with open('./' + mynamefile, 'w') as myfile:
        wr = csv.writer(myfile, delimiter='\n', quoting=csv.QUOTE_MINIMAL)
        wr.writerow(mylist)


def show_data(gen, scaler, number_of_rows):
    z = np.random.normal(0, 1, (number_of_rows, 100))
    labels = np.random.randint(2, size=number_of_rows)
    gen_imgs = gen.predict([z, labels])
    gen_imgs = scaler.inverse_transform(gen_imgs)
    for index in range(0, number_of_rows):
        gen_imgs[index] = np.around(gen_imgs[index], 4)
        gen_imgs[index][0] = np.around(gen_imgs[index][0], 0)
    return gen_imgs


def build_gen(zdim):
    model = Sequential()
    model.add(Dense(31, input_dim=zdim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1 * 31, activation='tanh'))
    return model


def build_cgen(zdim):
    z = Input(shape=(zdim,))
    lable = Input(shape=(1,), dtype='int32')
    lable_emb = Embedding(num_classes, zdim, input_length=1)(lable)
    lable_emb = Flatten()(lable_emb)
    joined_rep = Multiply()([z, lable_emb])
    gen_v = build_gen(zdim)
    c_img = gen_v(joined_rep)
    return Model([z, lable], c_img)


def build_dis(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(31))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_cdis(img_shape):
    img = Input(shape=(img_cols,))
    lable = Input(shape=(1,), dtype='int32')
    lable_emb = Embedding(num_classes, np.prod((31)), input_length=1)(lable)
    lable_emb = Flatten()(lable_emb)
    # lable_emb=Reshape(img_shape)(lable_emb)
    concate_img = Concatenate(axis=-1)([img, lable_emb])
    dis_v = build_dis((img_rows, img_cols * 2))
    classification = dis_v(concate_img)
    return Model([img, lable], classification)


def build_cgan(genrator, discriminator):
    z = Input(shape=(zdim,))
    lable = Input(shape=(1,), dtype='int32')
    f_img = genrator([z, lable])
    classification = discriminator([f_img, lable])
    model = Model([z, lable], classification)
    return model


dis_v = build_cdis(img_shape)
dis_v.compile(loss='binary_crossentropy',
              optimizer=adam_v2.Adam(),
              metrics=['accuracy'])

gen_v = build_cgen(zdim)
dis_v.trainable = False
gan_v = build_cgan(gen_v, dis_v)
gan_v.compile(loss='binary_crossentropy',
              optimizer=adam_v2.Adam()
              )

losses = []
accuracies = []
iteration_checks = []

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
data = pd.merge(pd.read_csv("./input/{}.csv".format(station_name)),
                pd.merge(pd.merge(patients_df, observation_df, on="patient_id", how="outer"), condition_df,
                         on="patient_id", how="outer"), on="patient_id", how="inner")
data['label'] = data['label'].map(lambda x: "B" if x == "non-cancer" else "M")
print(data.head(5))

###############################################################################################

## Synthetic data generation

# for the Generation we use a cGAN-Algorithm with an intuitive structure
# in the cGAN structure there is a Discriminator and a Generator
# the Generator produce synthetic data based on the real world data
# the Discriminator try to separate the real world data from the synthetic data
# the output of both get the input of the other module

# for this case we use a static value to break the cycle
# the number of iteration, batch_size, interval and number of rows can be set
# iteration is the number of cycles
# batch size is the size of the data which go from Discriminator and Generator
# intervall is show of cycle data
# number of generated row can be set

# formating the training data of the cGAN
# label need to be 0 and 1 for better transistion

if generation_synthetic_data:
    labels_training = data[Y_FEATURE]
    labels_training = labels_training.map(dict(M=1, B=0))
    features_training = data[X_FEATURES]

    # merge the data to one dataframe again for the training of cGAN
    train_data_synthetic = pd.concat([labels_training, features_training], axis=1, join='inner')

    # main method of the training of the cGAN as explained above
    synthetic_data = train_synthetic(data=train_data_synthetic, iterations=5000, batch_size=128, interval=1000,
                                     number_of_row=data.shape[1])

    # the random generation for the patient-id, gender and birthday
    # label needs to be transformed to the original values
    label_synthetic = []
    patient_id_synthetic = []
    patient_gender_synthetic = []
    patient_birthday_synthetic = []

    # very simple approach for the missing data generation, note: can be improved
    for row in range(0, len(synthetic_data)):
        synthetic_data_row = synthetic_data[row,]
        if (synthetic_data_row[0] == 1):
            label_synthetic.append("M")
        else:
            label_synthetic.append("B")
        patient_id_synthetic.append(("bbmri" + str(row)))
        p_g = "female"
        p_b = "01.01.2000"
        patient_birthday_synthetic.append(p_b)
        patient_gender_synthetic.append(p_g)

    # use the X_FEATURES of the synthetic data
    synthetic_data = synthetic_data[:, 1:31]

    # write everything in a dataframe for representation
    synthetic_df = pd.DataFrame(np.c_[
                                    patient_id_synthetic, patient_gender_synthetic, patient_birthday_synthetic, synthetic_data, label_synthetic],
                                columns=["patient_id", "gender", "birthDate", "radius_mean", "texture_mean",
                                         "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
                                         "concavity_mean", "concave.points_mean", "symmetry_mean",
                                         "fractal_dimension_mean",
                                         "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
                                         "compactness_se", "concavity_se", "concave.points_se", "symmetry_se",
                                         "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
                                         "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                                         "concave.points_worst", "symmetry_worst", "fractal_dimension_worst", "label"])

"""
@Toralf, please add the code for enlarging the dataset size.
"""

##############################################################################################

## splitting the station-data into train and test

# splitting will be performed as 60% training data and 40% test data

# create a data frame dictionary to store your data frames
DataFrameDict = {elem: pd.DataFrame for elem in ['M', 'B']}

for key in DataFrameDict.keys():
    DataFrameDict[key] = data[:][data.label == key]

data_label_B = DataFrameDict['B']
data_label_M = DataFrameDict['M']

data_train_B, data_test_B = train_test_split(data_label_B, test_size=0.4)
data_train_M, data_test_M = train_test_split(data_label_M, test_size=0.4)

data_train = data_train_B.append(data_train_M)
data_test = data_test_B.append(data_test_M)

###############################################################################################
## Exploratory data analysis (EDA) and data normalization
## training data will be in this example the synthetic data generated by the GAN-Algorithm
## test data will be the real data

if generation_synthetic_data:
    X_train = data_train[X_FEATURES].append(synthetic_df[X_FEATURES].astype(str).astype(float))
    y_train = data_train[Y_FEATURE].append(synthetic_df[Y_FEATURE])
else:
    X_train = data_train[X_FEATURES]
    y_train = data_train[Y_FEATURE]
X_test = data_test[X_FEATURES]
y_test = data_test[Y_FEATURE]

## Saving statistical data for visualization of train data
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
print("Mean: ", X_train_mean)
print("Std: ", X_train_std)
B = (y_train == 'B').sum()
M = (y_train == 'M').sum()
print('Number of Benign: ', B)
print('Number of Malignant : ', M)
stat_dir = os.path.join(out_dir, 'stat', station_name)
if not os.path.exists(stat_dir):
    os.makedirs(stat_dir)
df_X_train_mean = pd.DataFrame([X_train_mean], columns=X_FEATURES)
df_X_train_std = pd.DataFrame([X_train_std])
df_y_train_dist = pd.DataFrame([[B, M]], columns=['B', 'M'])
df_X_train_mean.to_csv(osp.join(stat_dir, 'X_train_mean.csv'))
df_X_train_std.to_csv(osp.join(stat_dir, 'X_train_std.csv'))
df_y_train_dist.to_csv(osp.join(stat_dir, 'Y_train_Dist.csv'))  # changed the name of the file

X_test_mean = X_test.mean(axis=0)
X_test_std = X_test.std(axis=0)
print("Mean: ", X_test_mean)
print("Std: ", X_test_std)
B = (y_test == 'B').sum()
M = (y_test == 'M').sum()
print('Number of Benign: ', B)
print('Number of Malignant : ', M)
stat_dir = os.path.join(out_dir, 'stat', station_name)
if not os.path.exists(stat_dir):
    os.makedirs(stat_dir)
df_X_test_mean = pd.DataFrame([X_test_mean], columns=X_FEATURES)
df_X_test_std = pd.DataFrame([X_test_std])
df_y_test_dist = pd.DataFrame([[B, M]], columns=['B', 'M'])
df_X_test_mean.to_csv(osp.join(stat_dir, 'X_test_mean.csv'))
df_X_test_std.to_csv(osp.join(stat_dir, 'X_test_std.csv'))
df_y_test_dist.to_csv(osp.join(stat_dir, 'Y_test_Dist.csv'))  # changed the name of the file
# normalization of the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
y_train.replace(to_replace=dict(M=1, B=0), inplace=True)
y_train = y_train.to_numpy()

X_test = scaler.fit_transform(X_test)
y_test.replace(to_replace=dict(M=1, B=0), inplace=True)
y_test = y_test.to_numpy()

## logistic regression model

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
        f.write(','.join(["epoch", "station_name", "train_loss", "val_loss", "val_acc", "val_prec", "val_rec", "val_f1",
                          "time"]) + '\n')

if generation_synthetic_data:
    filename_output_log = station_name + "synthetic_logs.csv"
else:
    filename_output_log = station_name + "logs.csv"

for epoch in range(num_epochs):
    perm = np.arange(X_train.shape[0])
    np.random.shuffle(perm)
    X_train = X_train[perm]
    y_train = y_train[perm]
    loss = train(X_train, y_train, model, criterion, optimizer)
    val_loss, val_acc, val_precesion, val_recall, val_f1_score, val_ppv = valid(X_test, y_test, model, criterion)  # changed
    with open(osp.join(training_dir, filename_output_log), 'a') as f:
        log = map(str, [epoch, station_name, loss, val_loss, val_acc, val_precesion, val_recall, val_f1_score,
                        str(datetime.datetime.now(pytz.timezone('Europe/Berlin')))])
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
