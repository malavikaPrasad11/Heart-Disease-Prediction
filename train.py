import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("C:/Users/ADMIN/OneDrive/Desktop/internship/Heart Disease Prediction/data/dataset.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Basic models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# Hyperparameter tuning for Random Forest
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "bootstrap": [True, False]
}

print("Tuning Random Forest...")
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring="accuracy"
)
grid_search.fit(X_train, y_train)
rf_best = grid_search.best_estimator_
rf_preds = rf_best.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds)

# Train XGBoost
print("Training XGBoost...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds)

# Evaluate all models
best_model = None
best_score = 0

print("\nModel Evaluation Results:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"Model: {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")

    if acc > best_score:
        best_score = acc
        best_model = model
        best_name = name

# Check RF (tuned) vs XGBoost
print("Tuned Random Forest:")
print(f"  Accuracy : {rf_acc:.4f}")
print(f"  F1 Score : {rf_f1:.4f}\n")

print("XGBoost:")
print(f"  Accuracy : {xgb_acc:.4f}")
print(f"  F1 Score : {xgb_f1:.4f}\n")

# Final best model selection
if rf_acc > best_score and rf_acc >= xgb_acc:
    best_model = rf_best
    best_name = "RandomForestClassifier (Tuned)"
elif xgb_acc > best_score:
    best_model = xgb
    best_name = "XGBoostClassifier"

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")
print(f"\nBest model saved: {best_name}")
