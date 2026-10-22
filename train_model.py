from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

def main():
    data = pd.read_csv(DATA_PATH)
    target = "maintenance_label"
    X = data.drop(columns=[target])
    y = data[target]
    categorical = ["vehicle_type"]
    numeric = [col for col in X.columns if col not in categorical]
    preprocessor = ColumnTransformer([
        ("vehicle_type", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numbers", "passthrough", numeric),
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        )),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Dataset rows:", len(data))
    print("Train rows:", len(X_train), "| Test rows:", len(X_test))
    print("Demo test accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("\nClassification report (synthetic data only):")
    print(classification_report(y_test, predictions, zero_division=0))
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")
    print("Reminder: metrics measure agreement with synthetic labels, not real-world diagnostic accuracy.")

if __name__ == "__main__":
    main()
