import pandas as pd

data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(data)

print(data.head())
print(data.tail())
print("Rows & columns",data.shape)
print(data.columns)

print(data.info())

print(data.isnull().sum())
print("Duplicate rows:",data.duplicated().sum())

print(data.columns.tolist())
print(data.drop(["EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"],axis=1))


print(data["Attrition"].value_counts())

print(data.head())
print(data.columns)

print(data.describe())
print(data["Attrition"].value_counts())

import matplotlib.pyplot as plt

data["Attrition"].value_counts().plot(kind="bar")

plt.title("Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.show()


import seaborn as sns
#Overtime and attrition
sns.countplot(data=data, x="OverTime", hue="Attrition")

plt.title("OverTime vs Attrition")
plt.show()
#department and attrition
sns.countplot(data=data, x="Department", hue="Attrition")

plt.title("Department vs Attrition")
plt.xticks(rotation=45)
plt.show()

#Job role and Attrition
plt.figure(figsize=(12,5))

sns.countplot(data=data, x="JobRole", hue="Attrition")

plt.title("Job Role vs Attrition")
plt.xticks(rotation=45)
plt.show()

#Monthly income and attrition
sns.boxplot(data=data,x="Attrition",y="MonthlyIncome")
plt.title("Monthly Income vs  Attrition")

plt.show()
data["Attrition"] = data["Attrition"].map({
    "Yes": 1,
    "No": 0
})

print(data["Attrition"].head())

data["OverTime"] = data["OverTime"].map({
    "Yes": 1,
    "No": 0
})
print(data.select_dtypes(include="object").columns)
data=(pd.get_dummies(data, drop_first=True))
print(data.head())
print(data.shape)
df = data.astype(int)


X = data.drop("Attrition", axis=1)
y = data["Attrition"]

print("X shape:", X.shape)
print("y shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_pred)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy %:", accuracy * 100)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

print(cm)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Employee Attrition Confusion Matrix")
plt.show()

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print(rf_pred)
from sklearn.metrics import accuracy_score

rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)
print("Accuracy %:", rf_accuracy * 100)
from sklearn.metrics import classification_report

print(classification_report(y_test, rf_pred))
from sklearn.metrics import confusion_matrix

rf_cm = confusion_matrix(y_test, rf_pred)

print(rf_cm)
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()
importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(10))
prediction = rf_model.predict(X_test.iloc[[0]])

print("Prediction:", prediction)