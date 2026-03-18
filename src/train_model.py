
# ===============================
# Product Category Classification
# Train Model Script
# ===============================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report


# ===============================
# 1. Load dataset
# ===============================
print("Loading dataset...")

df = pd.read_csv("../data/products.csv")

print("Dataset shape:", df.shape)


# ===============================
# 2. Clean column names
# ===============================
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.lower()


# ===============================
# 3. Drop missing values
# ===============================
df = df.dropna()

print("After cleaning shape:", df.shape)


# ===============================
# 4. Prepare text data
# ===============================
df["product_title_cleaned"] = (
    df["product_title"]
    .astype(str)
    .str.lower()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.replace(r"\d+", "", regex=True)
    .str.strip()
)


# ===============================
# 5. Define X and y
# ===============================
x = df["product_title_cleaned"]
y = df["category_label"]


# ===============================
# 6. Train-test split
# ===============================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


# ===============================
# 7. Define models
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": LinearSVC()
}


# ===============================
# 8. Train & evaluate
# ===============================
best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    print("\n==============================")
    print(f"Model: {name}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", model)
    ])

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)

    print(classification_report(y_test, y_pred))

    score = pipeline.score(x_test, y_test)

    if score > best_score:
        best_score = score
        best_model = pipeline
        best_name = name


# ===============================
# 9. Save best model
# ===============================
print("\nBest model:", best_name)
print("Accuracy:", best_score)

model_dir = "../model"

os.makedirs(model_dir, exist_ok=True)

# Save best model
joblib.dump(best_model, f"{model_dir}/product_classifier.pkl")

print("Model and vectorizer saved in root /model folder!")
