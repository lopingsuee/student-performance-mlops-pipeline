import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_data(path):
    if not os.path.exists(path):
        print(f"File tidak ditemukan: {path}")
        return None
    data = pd.read_csv(path)
    print(f"Data berhasil dimuat. Shape: {data.shape}")
    return data

def split_data(data):
    X = data.drop(columns=["pass_status"])
    y = data["pass_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def tune_logistic_regression(data):
    X_train, X_test, y_train, y_test = split_data(data)
    mlflow.set_experiment("student-performance-tuning")
    C_values = [0.01, 0.1, 1.0, 10.0]
    best_f1 = -1.0
    best_params = None
    best_run_id = None

    for C in C_values:
        with mlflow.start_run(run_name=f"logreg_C_{C}") as run:
            model = LogisticRegression(C=C, max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("C", C)
            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, "model")

            print(f"\nRun untuk C={C}")
            print(f"Akurasi  : {acc:.4f}")
            print(f"Presisi  : {prec:.4f}")
            print(f"Recall   : {rec:.4f}")
            print(f"F1-Score : {f1:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))

            if f1 > best_f1:
                best_f1 = f1
                best_params = {"C": C, "max_iter": 1000}
                best_run_id = run.info.run_id

    print("\n=== HASIL TUNING TERBAIK ===")
    print(f"Run ID terbaik : {best_run_id}")
    print(f"F1-Score terbaik: {best_f1:.4f}")
    print(f"Parameter terbaik: {best_params}")

if __name__ == "__main__":
    data_path = "siswa_clean.csv"
    data = load_data(data_path)
    if data is not None:
        tune_logistic_regression(data)
