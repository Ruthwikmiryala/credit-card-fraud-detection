import joblib
import pandas as pd
from pathlib import Path

# Load trained model
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fraud_detection_rf.joblib"

model_bundle = joblib.load(MODEL_PATH)

model = model_bundle['model']
threshold = model_bundle['threshold']

print("Fraud Detection Model Loaded Successfully!")

def predict_transaction(transaction):

    data = pd.DataFrame([transaction])

    data = data[model.feature_names_in_]

    probability = model.predict_proba(data)[0][1]

    prediction = int(probability >= threshold)

    if prediction == 1:
        result = "Fraudulent Transaction"
    else:
        result = "Legitimate Transaction"

    return {
        "Prediction": result,
        "Fraud Probability": round(probability * 100, 2)
    }

if __name__ == "__main__":

    DATA_PATH = Path(__file__).resolve().parent.parent / "dataset" / "creditcard.csv"

    df = pd.read_csv(DATA_PATH)

    # Select one legitimate and one fraudulent transaction
    legitimate = df[df['Class'] == 0].iloc[0]
    fraudulent = df[df['Class'] == 1].iloc[0]

    print("\nLegitimate Transaction:")
    print(predict_transaction(legitimate.drop('Class').to_dict()))

    print("\nFraudulent Transaction:")
    print(predict_transaction(fraudulent.drop('Class').to_dict()))