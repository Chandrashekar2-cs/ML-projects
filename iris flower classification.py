import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#load dataset
data = load_iris()
X = data.data
y = data.target
#train_test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#train model
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train,y_train)
#evalute
pred = model.predict(X_test)
print(accuracy_score(y_test,pred))