import sys
import os

sys.path.append('C:\\Users\\coton\\Desktop\\github\\fema\\src\\')


import fema_regression
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt


df = pd.read_csv('C:\\Users\\coton\\Desktop\\github\\fema\\data\\regressionData.csv')

features = ['MSSubClass', 'LotFrontage', 'LotArea', 'PoolArea', 'MoSold', 'YrSold']	
target = ['SalePrice']

df = df[features+target].dropna()

train_x, test_x, train_y, test_y = train_test_split(df[features].values, df[target].values, test_size=0.1)

scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)


model = fema_regression.FEMaRegressor(train_x,train_y,3,basis=fema_regression.Basis.radialBasis)

pred = model.predict(test_x,2)

plt.plot(pred,c='r')
plt.plot(test_y,c='b')
plt.show()

print(mean_absolute_percentage_error(test_y,pred))

