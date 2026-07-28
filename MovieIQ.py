from scipy.stats import ttest_ind
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("movies.csv")

print(df.head())
# Dataset Information
print("\nDataset Info:")
print(df.info())

# Shape
print("\nShape:")
print(df.shape)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

df["success"] = (df["revenue"] > df["budget"]).astype(int)

print(df.head())

# Check Budget = 0
print("\nBudget = 0")
print((df["budget"] == 0).sum())

# Check Revenue = 0
print("\nRevenue = 0")
print((df["revenue"] == 0).sum())

# Remove invalid rows
df = df[(df["budget"] > 0) & (df["revenue"] > 0)]

print("\nDataset Shape After Cleaning:")
print(df.shape)

# Success Count
print("\nSuccess Count:")
print(df["success"].value_counts())

# Budget vs Revenue Scatter Plot

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="budget",
    y="revenue",
    hue="success"
)

plt.title("Budget vs Revenue")
plt.xlabel("Budget")
plt.ylabel("Revenue")

plt.savefig("budget_vs_revenue.png")
plt.show()

# Extract Genre Name

df["genre_name"] = df["genres"].str.extract(r"'name': '([^']+)'")

print(df[["genres", "genre_name"]].head())

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="genre_name",
    order=df["genre_name"].value_counts().index
)

plt.title("Number of Movies by Genre")
plt.xlabel("Genre")
plt.ylabel("Count")

plt.xticks(rotation=45)

plt.savefig("genre_distribution.png")
plt.show()

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="genre_name",
    hue="success"
)

plt.title("Movie Success by Genre")
plt.xlabel("Genre")
plt.ylabel("Count")

plt.xticks(rotation=45)

plt.savefig("genre_success.png")
plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------

plt.figure(figsize=(8,6))

correlation = df[
    ["budget",
     "revenue",
     "popularity",
     "runtime",
     "vote_average",
     "success"]
].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.savefig("correlation_heatmap.png")

plt.show()


# -----------------------------
# T-Test
# -----------------------------

success_movies = df[df["success"] == 1]["popularity"]
failure_movies = df[df["success"] == 0]["popularity"]

t_stat, p_value = ttest_ind(success_movies, failure_movies)

print("\nT-Test Results")
print("T-Statistic :", t_stat)
print("P-Value :", p_value)

if p_value < 0.05:
    print("Result : Significant Difference")
else:
    print("Result : No Significant Difference")

# -----------------------------
# Machine Learning
# -----------------------------

# Features
X = df[[
    "budget",
    "popularity",
    "runtime",
    "vote_average"
]]

# Target
y = df["success"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Data :", X_train.shape)
print("Testing Data :", X_test.shape)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nAccuracy :", accuracy_score(y_test, y_pred))

print("\nPrecision :", precision_score(y_test, y_pred))

print("\nRecall :", recall_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("Class Weights:", class_weights)
# -----------------------------
# Feature Importance
# -----------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(8,5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")

plt.savefig("feature_importance.png")

plt.show()

import joblib

joblib.dump(model, "movie_model.pkl")

print("Model Saved Successfully")