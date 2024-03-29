import sys
import os
from typing import Tuple

sys.path.append('/home/danillorp/Área de Trabalho/github/fema/src/')


import fema_classifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


df = pd.read_csv('/home/danillorp/Área de Trabalho/github/fema/data/classificationData.csv',sep=';')

features = ['A', 'B', 'C']	
target = ['class']

df = df[features+target].dropna()

train_x, test_x, train_y, test_y = train_test_split(df[features].values, df[target].values, test_size=0.5)

scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)


model = fema_classifier.FEMaClassifier(k=2,basis=fema_classifier.Basis.radialBasis)
model.fit(train_x,train_y)

pred, confidence_level = model.predict(test_x,10)

print(confusion_matrix(test_y,pred))



#train_x,_  = model_fema_original.FEMaRelax(test_x,3,1)

#scaler = StandardScaler()

#train_x = scaler.fit_transform(train_x)
#test_x = scaler.transform(test_x)
#model = fema_classifier.FEMaClassifier(k=3,basis=fema_classifier.Basis.radialBasis)
#model.fit(train_x,train_y)

#pred, confidence_level = model.predict(test_x,3)

#cm_fema_relax = confusion_matrix(test_y,pred)

#print(cm_fema_original)
#print(cm_fema_relax)


