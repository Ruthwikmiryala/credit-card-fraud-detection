import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_and_preprocess_data

X_train, X_test, y_train, y_test = load_and_preprocess_data()

model = RandomForestClassifier(
    n_estimators=113,
    max_depth=10,
    max_features='log2',
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fraud_detection_rf.joblib"

joblib.dump({
    'model': model,
    'threshold': 0.35
}, MODEL_PATH)

print("Model trained and saved successfully!")