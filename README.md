# 🏠 House Loan Default Analysis using Deep Learning

A comprehensive machine learning pipeline to predict loan default risk using a deep neural network with Keras. The project includes preprocessing, model training, evaluation, and result visualization. It also demonstrates saving and reloading models for future inference.

---

## 📁 Project Structure
```
house_loan_analysis/
├── data/
│   └── loan_data.csv           # Main dataset    
├── outputs/
│   ├── model_summary.txt       # Model architecture summary
│   ├── confusion_matrix.png    # Confusion matrix plot
│   ├── roc_curve.png           # ROC-AUC curve
│   └── metrics.txt             # Accuracy, Sensitivity, AUC values
├── models/
│   └── loan_default_model.h5   # Trained Keras model
├── scripts/
│   ├── evaluate_saved_model.py   # Python file to run after generating models (.h5/.keras)
│   └── loan_default_analysis.py  # Main script
└── README.md
```
---

## How to Run:

### Run Python venv enviornment:
```
py -3.10 -m venv venv
```

### Activate the enviornment:
```
venv\Scripts\activate
```

### Install 
```
pip install -r requirements.txt
```

### Train the Model:

```
python scripts/loan_default_analysis.py
```

---

## ✅ 5. Output Saved
### All outputs will be saved in the outputs/ folder:
```
confusion_matrix.png – Confusion matrix

roc_curve.png – ROC AUC curve

metrics.txt – Contains sensitivity and AUC score

model_summary.txt – Keras model architecture

loan_default_model.h5 – Trained model
```
---

## Models

###  Models will be saved in the /models directory.
```
Loan_deafult_model.h5

loan_default_model.keras
```
### Evaluate Saved Models:

```
python scripts/evaluate_saved_model.py
```

*Note: Run in project root directory*

---

## ✅ 4. Libraries Required
```
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow
```
**Note: All of these are included in requirements.txt. Hence running *pip* install -r requirements.txt* will install them all.**

---

## 📌 Notes

### Model files (.h5 and .keras) are automatically saved after training.

### Evaluation results will be stored in the /outputs folder for easy access and reporting.

---

# Output Screenshots:

![](/images/class_distribution.png)

![](/images/confusion_matrix.png)

![](/images/roc_curve.png)

---

## Output Results:

### metrics.txt

```
Sensitivity: 0.8329
AUC: 0.9303
```

### model_summary.txt

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 128)                 │          24,832 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 64)                  │           8,256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 1)                   │              65 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 99,461 (388.52 KB)
 Trainable params: 33,153 (129.50 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 66,308 (259.02 KB)
```
---

