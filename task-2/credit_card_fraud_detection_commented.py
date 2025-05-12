import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE

start_time = time.time()  # Track execution time

print("Step 1: Loading Dataset...")
data = pd.read_csv('creditcard.csv')
print(f"✅ Dataset loaded. Shape: {data.shape}\n")

# Checking class distribution
print("\nStep 2: Checking Class Distribution:")
print(data['Class'].value_counts())

# Visualizing original class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data)
plt.title("Original Class Distribution")
plt.show(block=False)  # Prevents blocking
plt.pause(2)  # Keeps the plot visible for 2 seconds
plt.close()  # Closes automatically after display

# Checking for missing values efficiently
print("\nStep 3: Checking for Missing Values...")
total_missing = data.isnull().sum().sum()
print(f"✅ Total missing values: {total_missing}\n")

# Normalizing the Amount column
print("Step 4: Normalizing Amount Column...")
scaler = StandardScaler()
data['NormalizedAmount'] = scaler.fit_transform(data[['Amount']])
data.drop(['Amount', 'Time'], axis=1, inplace=True)
print("✅ Normalization complete.\n")

# Handling class imbalance with SMOTE **(LIMITED DATA SIZE FOR SPEED)**
print("Step 5: Handling Class Imbalance with SMOTE...")
X = data.drop('Class', axis=1)
y = data['Class']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# **Limit dataset size for faster processing**
print("Reducing dataset size to 30,000 rows for quicker execution...")
X_resampled = X_resampled.sample(30000, random_state=42)
y_resampled = y_resampled.loc[X_resampled.index]

print(f"✅ Class distribution after SMOTE: {np.bincount(y_resampled)}\n")

# Visualizing balanced class distribution after SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled)
plt.title("Balanced Class Distribution After SMOTE")
plt.show(block=False)  # Prevents blocking
plt.pause(2)  # Keeps the plot visible for 2 seconds
plt.close()  # Closes automatically after display

# Splitting dataset **(REDUCED TEST SIZE TO 15% FOR SPEED)**
print("Step 6: Splitting Data into Training and Testing Sets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.15, random_state=42)
print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")

# **Training models (Reduced Random Forest trees for speed)**
print("Step 7: Training Models...")
lr_model = LogisticRegression()
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)  # Optimized for speed

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
print("✅ Model training completed.\n")

# Evaluating models
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n--- {model_name} Evaluation ---")
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("\n")

# Predictions
print("Step 8: Evaluating Models...")
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

evaluate_model(y_test, y_pred_lr, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# **Visualizing Confusion Matrix for Random Forest**
print("Displaying Confusion Matrix for Random Forest...")
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test)
plt.title("Confusion Matrix - Random Forest")
plt.show(block=False)  # Prevents blocking
plt.pause(2)  # Keeps the plot visible for 2 seconds
plt.close()  # Closes automatically after display

end_time = time.time()
print(f"✅ Total Execution Time: {round(end_time - start_time, 2)} seconds")