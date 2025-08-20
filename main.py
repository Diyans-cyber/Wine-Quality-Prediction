# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

# 2. Load Dataset
df = pd.read_csv("wine.csv")   # Ensure the file is in the same folder

# 3. Explore Data
print("Dataset Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())
print(df.info())
print(df.describe())

# Check missing values
print("Missing Values:\n", df.isnull().sum())

# 4. Data Visualization
plt.figure(figsize=(8,6))
sns.countplot(x='quality', data=df, palette="viridis", legend=False)
plt.title("Distribution of Wine Quality")
plt.show()

plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot (optional - heavy)
# sns.pairplot(df, hue="quality")

# 5. Feature & Target
X = df.drop("quality", axis=1)
y = df["quality"]

# Optional: convert quality into binary (good vs bad wine)
# y = y.apply(lambda x: 1 if x >= 7 else 0)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SGD Classifier": SGDClassifier(random_state=42),
    "Support Vector Classifier": SVC(kernel='rbf', random_state=42)
}

# 9. Train & Evaluate
for name, model in models.items():
    print("\n==============================")
    print(f"Model: {name}")
    print("==============================")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
