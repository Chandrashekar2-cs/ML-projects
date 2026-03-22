import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = {
    "message": [
        "Win money now!!!",
        "Hello how are you",
        "Free entry in contest",
        "Let's meet tomorrow",
        "Congratulations you won prize",
        "Call me now"
    ],
    "label": [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
df["message"] = df["message"].str.lower()
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


