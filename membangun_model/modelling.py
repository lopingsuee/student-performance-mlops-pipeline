import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import os

def load_data(path):
    if not os.path.exists(path):
        print(f"❌ File tidak ditemukan: {path}")
        return None
    data = pd.read_csv(path)
    print(f"✅ Data berhasil dimuat. Jumlah data: {data.shape}")
    return data

def train_model(data):
    X = data.drop(columns=["pass_status"])
    y = data["pass_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("student-performance")
    mlflow.sklearn.autolog()  

    with mlflow.start_run(run_name="logistic_regression"):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\n=== HASIL EVALUASI MODEL ===")
        print(f"Akurasi     : {acc:.4f}")
        print(f"Presisi     : {prec:.4f}")
        print(f"Recall      : {rec:.4f}")
        print(f"F1-Score    : {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    print("\n✅ Model berhasil dilatih dan dicatat di MLflow!")

if __name__ == "__main__":
    data = load_data("siswa_clean.csv")
    if data is not None:
        train_model(data)
