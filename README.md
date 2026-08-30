# 📊 Employee Attrition Prediction using Machine Learning

## 📌 Project Overview

Employee Attrition Prediction is a Machine Learning project that analyzes employee data and predicts whether an employee is likely to leave the company.

The project uses **Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn** for data analysis, visualization, preprocessing, model building, and evaluation.

Two Machine Learning algorithms are used:

* Logistic Regression
* Random Forest Classifier

---

## 🎯 Project Objective

The main objectives of this project are:

* Analyze employee information
* Understand the factors related to employee attrition
* Perform Exploratory Data Analysis (EDA)
* Clean and preprocess the dataset
* Build Machine Learning classification models
* Compare Logistic Regression and Random Forest
* Evaluate model performance
* Identify important features affecting attrition
* Predict whether an employee is likely to leave

---

## 📂 Dataset

The project uses the **IBM HR Employee Attrition dataset**.

The dataset contains employee information such as:

* Age
* Department
* Job Role
* Monthly Income
* OverTime
* Job Satisfaction
* Years at Company
* Job Level
* Work-Life Balance
* Education
* Marital Status
* Attrition

### Target Variable

`Attrition`

```text
Yes → 1 → Employee leaves the company
No  → 0 → Employee stays in the company
```

---

## 🛠️ Technologies Used

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| Python           | Programming               |
| Pandas           | Data manipulation         |
| NumPy            | Numerical operations      |
| Matplotlib       | Data visualization        |
| Seaborn          | Statistical visualization |
| Scikit-learn     | Machine Learning          |
| Jupyter Notebook | Development               |
| Git & GitHub     | Version control           |

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Missing Value Checking
   ↓
Duplicate Checking
   ↓
EDA & Visualization
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Feature Importance
   ↓
Employee Attrition Prediction
```

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Checked dataset shape
* Checked column names
* Checked missing values
* Checked duplicate records
* Removed unnecessary columns
* Converted `Attrition` into numerical values
* Converted `OverTime` into numerical values
* Applied One-Hot Encoding to categorical variables
* Converted data into integer format
* Split data into training and testing sets

### Removed Columns

```text
EmployeeCount
EmployeeNumber
Over18
StandardHours
```

---

# 📊 Exploratory Data Analysis

Several visualizations were created to understand employee attrition.

## 1. Employee Attrition Distribution

This visualization shows the number of employees who stayed and left the company.

## 2. Overtime vs Attrition

This analysis helps understand the relationship between overtime work and employee attrition.

## 3. Department vs Attrition

This visualization compares attrition across different departments.

## 4. Job Role vs Attrition

This helps identify differences in attrition among various job roles.

## 5. Monthly Income vs Attrition

A box plot is used to compare employee income between employees who stayed and those who left.

## 6. Age vs Attrition

This visualization analyzes the relationship between employee age and attrition.

## 7. Job Satisfaction vs Attrition

This helps analyze whether job satisfaction is associated with employee attrition.

## 8. Years at Company vs Attrition

This visualization compares employee tenure with attrition.

---

# 🤖 Machine Learning Models

## 1. Logistic Regression

Logistic Regression is used as a baseline classification model.

```python
log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)
```

---

## 2. Random Forest Classifier

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees.

```python
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
```

---

# 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### Model Comparison

| Model               |        Accuracy |
| ------------------- | --------------: |
| Logistic Regression | Add your result |
| Random Forest       | Add your result |

> Replace **"Add your result"** with the actual accuracy obtained after running the notebook.

---

# 🔍 Feature Importance

Random Forest feature importance is used to identify which features contribute most to the model's predictions.

```python
importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})
```

The project displays the **Top 10 Important Features** using a visualization.

---

# 🎯 Prediction

The trained Random Forest model can predict whether an employee is likely to leave.

```text
0 → Employee is likely to stay
1 → Employee is likely to leave
```

Example:

```python
prediction = rf_model.predict(
    X_test.iloc[[0]]
)
```

---

# 💡 Business Insights

The analysis can help organizations understand potential factors associated with employee attrition.

Possible insights include:

* Overtime may be associated with higher attrition.
* Employee income can be related to retention.
* Attrition can vary across job roles.
* Job satisfaction may influence employee retention.
* Years spent at the company can be associated with attrition.
* Employee characteristics can help HR teams identify higher-risk groups.

> These insights should be interpreted based on the actual EDA results and model output.

---

# 🚀 Future Improvements

The project can be improved by adding:

* Hyperparameter tuning
* Cross-validation
* SMOTE for class imbalance
* ROC-AUC curve
* Precision-Recall curve
* XGBoost
* Gradient Boosting
* Explainable AI
* Streamlit web application
* Interactive HR dashboard
* Employee risk prediction system

---

# 📁 Project Structure

```text
Employee-Attrition-Prediction/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── notebooks/
│   └── Employee_Attrition_Prediction.ipynb
│
├── images/
│   ├── attrition.png
│   ├── overtime.png
│   ├── department.png
│   ├── jobrole.png
│   ├── income.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── src/
│   └── employee_attrition.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ How to Run the Project

## Step 1: Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

## Step 2: Open Project Folder

```bash
cd Employee-Attrition-Prediction
```

## Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

## Step 4: Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/Employee_Attrition_Prediction.ipynb
```

---

# 📦 Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

# 📌 Key Skills Demonstrated

This project demonstrates practical knowledge of:

```text
Python
Data Cleaning
Data Preprocessing
Exploratory Data Analysis
Data Visualization
Feature Engineering
One-Hot Encoding
Classification
Logistic Regression
Random Forest
Model Evaluation
Confusion Matrix
Feature Importance
Machine Learning
```

---

# 👨‍💻 Author

**Praful Bowade**

### Career Goal

Aspiring **Data Scientist / Machine Learning Engineer**

### Interests

* Data Science
* Machine Learning
* Python
* Data Analysis
* Artificial Intelligence

---

# ⭐ Project Highlights

```text
✔ Real-world HR dataset
✔ Data cleaning and preprocessing
✔ Exploratory Data Analysis
✔ Multiple visualizations
✔ Logistic Regression
✔ Random Forest Classifier
✔ Model comparison
✔ Confusion Matrix
✔ Feature Importance
✔ Employee Attrition Prediction
```

---

## ⭐ If you found this project useful

Feel free to ⭐ **Star** the repository and explore the project.
