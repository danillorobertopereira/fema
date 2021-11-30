import sys
import os
from typing import Tuple

sys.path.append('C:\\Users\\coton\\Desktop\\github\\fema\\src\\')


import fema_semi
import fema_classifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\coton\\Desktop\\github\\fema\\data\\fetal_health.csv').reset_index()

features = [
            'baseline value', 'accelerations', 'fetal_movement',
            'uterine_contractions', 'light_decelerations', 'severe_decelerations',
            'prolongued_decelerations', 'abnormal_short_term_variability',
            'mean_value_of_short_term_variability',
            'percentage_of_time_with_abnormal_long_term_variability',
            'mean_value_of_long_term_variability', 'histogram_width',
            'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
            'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
            'histogram_median', 'histogram_variance', 'histogram_tendency',
            ]
target = ['fetal_health']

df[target] = df[target].astype(int)
#With FEMa the class label need start from 0
df[target] = df[target] - 1 


train_x, test_x, train_y, test_y = train_test_split(df[features].values, df[target].values, test_size=0.9)
uknw_x, test_x, uknw_y, test_y = train_test_split(test_x, test_y, test_size=0.5)

scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
uknw_x = scaler.transform(uknw_x)



model_fema_original = fema_classifier.FEMaClassifier(k=3,basis=fema_classifier.Basis.radialBasis)
model_fema_original.fit(train_x,train_y)

pred, confidence_level = model_fema_original.predict(test_x,3)

cm_fema_original = confusion_matrix(test_y,pred)
acc_original = balanced_accuracy_score(test_y, pred)




model_semi = fema_semi.FEMaSemiSupervisedClassifier(k=2,basis=fema_semi.Basis.radialBasis)
model_semi.fit(train_x,train_y,uknw_x)

pred, confidence_level = model_semi.predict(test_x,3)

cm_fema_semi = confusion_matrix(test_y,pred)
acc_semi = balanced_accuracy_score(test_y, pred)

print(cm_fema_original, acc_original)
print(cm_fema_semi, acc_semi)

