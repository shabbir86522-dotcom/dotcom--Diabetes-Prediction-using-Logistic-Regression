import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "Glucose": [80,85,90,95,100,110,120,130,140,150],
    "BMI": [20,21,22,23,24,26,28,30,32,35],
    "Diabetes": [0,0,0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Glucose", "BMI"]]
y = df["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

while True:
    glucose = float(input("Enter Glucose Level: "))
    bmi = float(input("Enter BMI: "))

    patient = pd.DataFrame({
        "Glucose": [glucose],
        "BMI": [bmi]
    })

    prediction = model.predict(patient)

    if prediction[0] == 1:
        print("Diabetes Detected")
    else:
        print("No Diabetes")

    again = input("Check another patient? (yes/no): ").lower()

    if again != "yes":
        print("Thank You!")
        break