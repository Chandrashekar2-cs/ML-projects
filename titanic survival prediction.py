
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.impute import SimpleImputer
#LOAD DATA 
df = pd.read_csv("titanic.csv")
print(df.head())
#EXPLORE DATA
print(df.info())
print(df.describe())
print(df.isnull().sum())
#CLEAN DATA
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop("Cabin",axis=1,inplace=True)
#ENCODE CATEGORICAL
df["Sex"]=df["Sex"].map({"male":0,"female":1})
df["Embarked"]=df["Embarked"].map({"S":0,"C":1,"Q":2})
#SELECT FEATURES AND TARGET
X = df[["Pclass","Age","Fare","Sex","Embarked"]]
y = df["Survived"]
# Impute missing values (extra safety)
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)
#TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#TRAIN MODEL
model = LogisticRegression(max_iter=500)
model.fit(X_train,y_train)
#evaluate
pred =model.predict(X_test)
print(accuracy_score(y_test,pred))
 



