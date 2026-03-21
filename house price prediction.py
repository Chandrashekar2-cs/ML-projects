import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
data = {
    "Area": [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000],
    "bedrooms":[1,2,3,4,5,6,7,8,9,10,11],
    "Price": [10,15,20,25,15,35,40,45,50,55,60]
}
df = pd.DataFrame(data)

#CLEAN DATA

df["Area"].fillna(df["Area"].mean())
df["bedrooms"].fillna(df["bedrooms"].mean())

X = df[["Area","bedrooms"]]
y = df["Price"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

pred= model.predict(X_test_scaled)
mse = mean_squared_error(y_test, pred)
print("Mean Squared Error:", mse)
print("Predictions:", pred)
import matplotlib.pyplot as plt

plt.scatter(y_test, pred, color="blue", label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.show()