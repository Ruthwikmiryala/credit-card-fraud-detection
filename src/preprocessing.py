import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_and_preprocess_data():

    DATA_PATH = Path(__file__).resolve().parent.parent / "dataset" / "creditcard.csv"

    df = pd.read_csv(DATA_PATH)

    df = df.drop_duplicates()

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test