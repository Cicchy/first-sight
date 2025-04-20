import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import joblib
import sys
import os

from src.xgbclassifier.preprocessing.preprocess import preprocess

# Importing dataset

data = pd.read_csv('data/frc-match-history.csv')

#Split data
X_train, X_test, y_train, y_test = preprocess(data)

# Train the XGBClassifier model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions with XGBCLassifier
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# Check performance
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)


print(f'Accuracy: {accuracy}')
print(f'ROC AUC: {roc_auc}')

# Save the model
dir = "src/xgbclassifier/model"
joblib.dump(xgb_model, f"{dir}/xgbclassifier.joblib")
