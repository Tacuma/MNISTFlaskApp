import numpy as np
import pickle
import os
import re
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Fetches MNIST
mnist = fetch_mldata('MNIST original')

#Creates the Data and labels
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

#Splits the test and train sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#Shuffles the rows and puts them in test and train sets
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#Creates and fits the data in a Random Forest Model
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)
some_data = X[36000]
result = forest_clf.predict([some_data])
y_pred_ranf = forest_clf.predict(X_test)
acc_for = accuracy_score(y_test, y_pred_ranf)
print("random forest accuracy: ", acc_for)

#Pickles the trained model
#dest = os.path.join('MNISTapp', 'model')
#if not os.path.exists(dest):
#    os.makedirs(dest)
pickle.dump(forest_clf,
           open(os.path.join('rforest_clf.pkl'), 'wb'),
           protocol=4)



