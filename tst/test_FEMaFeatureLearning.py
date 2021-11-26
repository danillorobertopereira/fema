import sys
import os
from typing import Tuple

sys.path.append('C:\\Users\\coton\\Desktop\\github\\fema\\src\\')


import fema_classifier
import fema_feature_learning
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\coton\\Desktop\\github\\fema\\data\\IrisDataset.csv')

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
target = ['Species']

le = preprocessing.LabelEncoder()
le.fit(df['Species'].values)
df['Species'] = le.transform(df['Species'].values)


train_x, test_x, train_y, test_y = train_test_split(df[features].values, df[target].values, test_size=0.4)
eval_x, test_x, eval_y, test_y = train_test_split(test_x, test_y, test_size=0.5)

scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
eval_x = scaler.transform(test_x)



model_fl = fema_feature_learning.FEMaFeatureLearning(k=2,basis=fema_classifier.Basis.radialBasis)

model_fl.fit(train_x, train_y, eval_x, eval_y)