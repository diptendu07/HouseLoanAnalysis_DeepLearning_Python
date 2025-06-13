import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Paths
DATA_PATH = "data/loan_data.csv"
MODEL_PATH = "models/loan_default_model.h5"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load dataset
df = pd.read_csv(DATA_PATH)

# Step 2: Check and handle null values
nulls = df.isnull().sum()
print("Null Values:\n", nulls)

# Drop columns with more than 50% missing values
threshold = 0.5
df = df.loc[:, df.isnull().mean() < threshold]

# Fill numeric NaNs with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical NaNs with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Step 3: Default percentage
default_pct = df['TARGET'].value_counts(normalize=True) * 100
print("Default Distribution:\n", default_pct)

# Step 4: Handle imbalance with SMOTE
X = df.drop(columns=['TARGET'])
y = df['TARGET']

# Encode categorical features
X = pd.get_dummies(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance dataset
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

# Step 5: Plot class distribution
sns.countplot(x=y_bal)
plt.title('Balanced Class Distribution')
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png")
plt.close()

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

# Step 7: Deep Learning Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

# Save model
model.save("models/loan_default_model.keras")

# Save model summary
with open(f"{OUTPUT_DIR}/model_summary.txt", "w", encoding='utf-8') as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# Step 8: Evaluation
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype("int32")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()

# Sensitivity
TP = cm[1, 1]
FN = cm[1, 0]
sensitivity = TP / (TP + FN)

# ROC AUC
auc = roc_auc_score(y_test, y_pred_probs)
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)

plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
plt.close()

# Save metrics
with open(f"{OUTPUT_DIR}/metrics.txt", "w") as f:
    f.write(f"Sensitivity: {sensitivity:.4f}\n")
    f.write(f"AUC: {auc:.4f}\n")

print("Model training and evaluation complete. Outputs saved.")
