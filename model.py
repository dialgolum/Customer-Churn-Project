import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load the dataset

df = pd.read_csv('churn_data.csv')

# # show basic info
# print(df.head())
# print("\nDataset shape:", df.shape)
# print("\nColumns:", df.columns.tolist())

# 1. Drop CustomerID
df.drop("customerID", axis=1, inplace=True)

# 2. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# 3. Handle missing values (There will be some NaN in TotalCharges )
df.dropna(inplace=True)

# 4. Encode categorical variables
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn", "gender"]
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1})

# 5. One hot encode remaining categorical features
df = pd.get_dummies(df, drop_first=True)

# 6. Separete features and target variable
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 7. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("Data preprocessing complete.")
# print("Training set shape:", X_train.shape)
# print("Testing set shape:", X_test.shape)

# 8. Initialize and Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9. Make predictions
y_pred = model.predict(X_test)

# 10. Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))