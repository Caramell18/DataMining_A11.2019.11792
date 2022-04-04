from pyexpat.errors import XML_ERROR_TAG_MISMATCH
from tracemalloc import start
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.multioutput import ClassifierChain

dataset = pd.read_csv("Tugas 5/Social Network Ads.csv")
x = dataset.iloc [:, [2,3]].values
y = dataset.iloc [:, -1 ].values
# print ("========= x =======")
# print (x)
# print ("========= y =======")
# print (y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# print ("=========x train===========")
# print (x_train)
# print ("=========x test=============")
# print (x_test)
# print ("=========y train===========")
# print (y_train)
# print ("=========y test=============")
# print (y_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# print ("=========x train===========")
# print (x_train)
# print ("=========x test=============")
# print (x_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_predic = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predic)
print (cm)

# ===========COLOR MAP DATA TRAINING==============
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start = x_set[:, 1].min()-1, stop = x_set[:, 0].max() + 1, step=0.01))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set==j,1],
    c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Klasifikasi Data Dengan Naive Bayes (Data Trining)')
plt.xlabel('Umur')
plt.ylabel('Estimasi Gaji')
plt.legend()
plt.show()

# ===========COLOR MAP DATA TESTING==============
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start = x_set[:, 1].min()-1, stop = x_set[:, 0].max() + 1, step=0.01))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set==j,1],
    c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Klasifikasi Data Dengan Naive Bayes (Data Testing)')
plt.xlabel('Umur')
plt.ylabel('Estimasi Gaji')
plt.legend()
plt.show()
