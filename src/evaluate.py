import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import load_and_preprocess_data
from sklearn.metrics import roc_auc_score, average_precision_score

X_train, X_test, y_train, y_test = load_and_preprocess_data()

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fraud_detection_rf.joblib"

model_bundle = joblib.load(MODEL_PATH)
model = model_bundle['model']
threshold = model_bundle['threshold']

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("PR-AUC:", round(average_precision_score(y_test, y_prob), 4))

