import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("train.csv")
dataset.head()
df = pd.DataFrame(data = dataset)
df = df.drop('Name', axis = 1)
df = df.drop('Ticket', axis = 1)
df = df.drop('Cabin', axis = 1)
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
categorical_features = df.select_dtypes(exclude = [np.number])
numerical_features = df.select_dtypes(include = [np.number])
categorical_features = categorical_features.values
numerical_features = numerical_features.values
dummies1 = pd.get_dummies(categorical_features[:, 0])
dummies1 = dummies1.values


dummies2 = pd.get_dummies(categorical_features[:, 1])
dummies2 = dummies2.values
dummies2 = dummies2[:, 1:3]
X = np.append(numerical_features, dummies2, axis = 1)
X = np.append(X, dummies1, axis = 1)
X = X[:, :-1]

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
X[:, 3:4] = imputer.fit_transform(X[:, 3:4])
X = X[:, 2:]

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X, y)

df1 = pd.read_csv("test.csv")
df1 = df1.drop('Name', axis = 1)
df1 = df1.drop('Ticket', axis = 1)
df1 = df1.drop('Cabin', axis = 1)

categorical_features1 = df1.select_dtypes(exclude = [np.number])
numerical_features1 = df1.select_dtypes(include = [np.number])
categorical_features1 = categorical_features1.values
numerical_features1 = numerical_features1.values
dummies11 = pd.get_dummies(categorical_features1[:, 0])
dummies11 = dummies11.values


dummies21 = pd.get_dummies(categorical_features1[:, 1])
dummies21 = dummies21.values
dummies21 = dummies21[:, 1:3]
X_test = np.append(numerical_features1, dummies21, axis = 1)
X_test = np.append(X_test, dummies11, axis = 1)
X_test = X_test[:, :-1]
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
X_test[:, 2:3] = imputer.fit_transform(X_test[:, 2:3])
X_test = X_test[:, 1:]
df2 = pd.DataFrame(data = X_test)
nulls = pd.DataFrame(df2.isnull().sum().sort_values())
X_test[:, 4:5] = imputer.fit_transform(X_test[:, 4:5])
nulls1 = pd.DataFrame(df2.isnull().sum().sort_values())
y_pred = regressor.predict(X_test)
y_true = pd.read_csv("gender_submission.csv").values
y_true = y_true[:,1]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)

submission = pd.DataFrame()
submission['PassengerId'] = df1.PassengerId
submission['Survived'] = y_pred
submission.to_csv('submission.csv', index = False)

# knn classifier
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1 = sc.fit_transform(X)
X_test1 = sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X1,y)
ypred1 = classifier.predict(X_test1)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_true, ypred1)

#kernel svm
from sklearn.svm import SVC
classifier2 = SVC(kernel = 'rbf', random_state = 0)
classifier2.fit(X, y)
ypred2 = classifier2.predict(X_test)
classifier22 = SVC(kernel = 'rbf', random_state = 0)
classifier22.fit(X1,y)
ypred22 = classifier22.predict(X_test1)
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_true, ypred2)
cm22 = confusion_matrix(y_true, ypred22)

# naive bayes
from sklearn.naive_bayes import GaussianNB
classifier3 = GaussianNB()
classifier3.fit(X, y)
ypred3 = classifier3.predict(X_test)
classifier33 = GaussianNB()
classifier33.fit(X1, y)
ypred33 = classifier33.predict(X_test1)
cm3 = confusion_matrix(y_true, ypred3)
cm33 = confusion_matrix(y_true, ypred33)
# decission tree classifier
from sklearn.tree import DecisionTreeClassifier
classifier4 = DecisionTreeClassifier()
classifier4.fit(X,y)
ypred4 = classifier4.predict(X_test)
cm4 = confusion_matrix(y_true, ypred4)
# random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier5 = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
classifier5.fit(X,y)
ypred5 = classifier5.predict(X_test)
cm5 = confusion_matrix(y_true, ypred5)
