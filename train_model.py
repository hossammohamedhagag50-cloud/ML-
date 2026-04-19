import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from imblearn.over_sampling import SMOTE

# 1. Load Data
print("Loading data...")
df = pd.read_csv('creditcard.csv')

# 2. Preprocessing
# Scale Amount and Time (or drop Time if not useful)
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
# Usually Time is less useful or needs specific engineering, let's keep it for now but scale it
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

X = df.drop('Class', axis=1)
y = df['Class']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Handle Imbalance with SMOTE (on training data only)
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 5. Train Model
print("Training Random Forest model...")
# Using a faster config for RF to avoid timeout, but still robust
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train_res, y_train_res)

# 6. Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Save Model and Scaler
print("Saving model and scaler...")
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Done!")
