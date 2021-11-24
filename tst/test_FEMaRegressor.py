import sys

sys.path.insert(1,'../src/fema')

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

df = pd.read_csv('../data/regressionData.csv')

features = ['MSSubClass', 'LotFrontage', 'LotArea', 'PoolArea', 'MoSold', 'YrSold']	
target = ['SalePrice']

df = df[features+target].dropna()

train_x, test_x, train_y, test_y = train_test_split(df[features].values, df[target].values, test_size=0.3)

scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)


model = FEMaRegressor(train_x,train_y,1,basis=Basis.radialBasis)

pred = model.predict(test_x,0.01)

plt.plot(pred,c='r')
plt.plot(test_y,c='b')
plt.show()

print(mean_absolute_percentage_error(test_y,pred))

