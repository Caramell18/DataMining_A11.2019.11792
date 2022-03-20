import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""import numpy matplot dan pandas"""

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print ('==============Read=================')
print (x)
print (y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

"""untuk mendapatkan nilai nan kita dapat menggunakan simpleimputer dengan mean,modus / median"""
print ('================SimpleImputer===============')
print (x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
coltrans = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
x = np.array(coltrans.fit_transform(x))
print ('===============Sklearn================')
print (x)

from sklearn.preprocessing import LabelEncoder
laben = LabelEncoder()
y = laben.fit_transform(y)
print (y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
print ('=================Split==============')
print (x_train)

print (x_test)

print (y_train)

print (y_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print ('===============Scaler================')
print (x_train)

print (x_test)