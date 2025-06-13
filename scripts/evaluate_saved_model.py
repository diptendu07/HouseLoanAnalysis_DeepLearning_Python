import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Config
DATA_PATH = "data/loan_data.csv"
MODEL_H5_PATH = "models/loan_default_model.h5"
MODEL_KERAS_PATH = "models/loan_default_model.keras"
OUTPUT_DIR = "outputs"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and preprocess data
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['TARGET'])
y = df['TARGET']

# Encode categorical features
X = pd.get_dummies(X)

# Impute missing values
X.fillna(X.median(), inplace=True)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance using SMOTE
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

# Choose model path: 'h5' or 'keras'
model_path = MODEL_KERAS_PATH  # Change to MODEL_H5_PATH to load .h5 model

# Load the saved model
model = load_model(model_path)

# Predict
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype("int32")

# Evaluate
cm = confusion_matrix(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_probs)
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)

# Sensitivity
TP = cm[1, 1]
FN = cm[1, 0]
sensitivity = TP / (TP + FN)

# Print results
print(f"Model: {os.path.basename(model_path)}")
print("Confusion Matrix:")
print(cm)
print(f"Sensitivity: {sensitivity:.4f}")
print(f"AUC: {auc:.4f}")

# Save ROC curve
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Loaded Model")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/roc_loaded_model.png")
plt.close()

# Save predictions to CSV (optional)
np.savetxt(f"{OUTPUT_DIR}/y_pred_probs.csv", y_pred_probs, delimiter=",", fmt="%.4f")
np.savetxt(f"{OUTPUT_DIR}/y_pred_labels.csv", y_pred, delimiter=",", fmt="%d")

