import sys
import os
from typing import Tuple

sys.path.append('C:\\Users\\coton\\Desktop\\github\\fema\\src\\')


import fema_classifier
import fema_feature_learning
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
eval_x, test_x, eval_y, test_y = train_test_split(test_x, test_y, test_size=0.5)

scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
eval_x = scaler.transform(eval_x)



model_fl = fema_feature_learning.FEMaFeatureLearning(k=3,basis=fema_classifier.Basis.radialBasis)
features_weigths = model_fl.fit(train_x, train_y, eval_x, eval_y)


model_fema_original = fema_classifier.FEMaClassifier(k=3,basis=fema_classifier.Basis.radialBasis)
model_fema_original.fit(train_x,train_y)

pred, confidence_level = model_fema_original.predict(test_x,3)

cm_fema_original = confusion_matrix(test_y,pred)
acc_original = balanced_accuracy_score(test_y, pred)




cm_fema_adjusted = []
acc_adjsted = []

for c in range(len(set(train_y[:,0]))):
    train_x_cp = train_x.copy()
    test_x_cp  = test_x.copy()
    

    for f in range(len(features)):
        mask_inter = train_y[:,0] != c
        mask_intra = train_y == c
        
        #train_x_cp[:,f] =   ( (train_x_cp[:,f]  - train_x_cp[mask_intra[:,0],f].mean())/train_x_cp[mask_intra[:,0],f].std() ) * (features_weigths[model_fl.INTER, c, f]/features_weigths[model_fl.INTRA, c, f])
        #test_x_cp[:,f] =   ( (test_x_cp[:,f]  - train_x_cp[mask_intra[:,0],f].mean())/train_x_cp[mask_intra[:,0],f].std() ) * (features_weigths[model_fl.INTER, c, f]/features_weigths[model_fl.INTRA, c, f])
        train_x_cp[:,f] =   (train_x_cp[:,f]) * 2**(features_weigths[model_fl.INTER, c, f] -features_weigths[model_fl.INTRA, c, f])
        test_x_cp[:,f] =    (test_x_cp[:,f]) * 2**(features_weigths[model_fl.INTER, c, f] - features_weigths[model_fl.INTRA, c, f])
        

    model_fema_adjusted = fema_classifier.FEMaClassifier(k=3,basis=fema_classifier.Basis.radialBasis)
    model_fema_adjusted.fit(train_x_cp,train_y)

    pred, confidence_level = model_fema_adjusted.predict(test_x_cp,3)

    cm_fema_adjusted.append(confusion_matrix(test_y,pred))

    acc_adjsted.append(balanced_accuracy_score(test_y, pred))

print(cm_fema_adjusted[0], acc_adjsted[0])
print(cm_fema_adjusted[1], acc_adjsted[0])
print(cm_fema_adjusted[2], acc_adjsted[0])
print(cm_fema_original, acc_adjsted[0])

