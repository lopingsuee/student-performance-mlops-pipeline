import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

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

def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def save_predictions_csv(y_true, y_pred, y_proba, path):
    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def train_and_log_to_dagshub(data):
    mlflow.set_tracking_uri("https://dagshub.com/lopingsuee/eksperimen_SML_aditya.mlflow")
    mlflow.set_experiment("student-performance-dagshub")
    X_train, X_test, y_train, y_test = split_data(data)
    C_value = 10.0

    with mlflow.start_run(run_name=f"logreg_dagshub_C_{C_value}"):
        model = LogisticRegression(C=C_value, max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C_value)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        report = classification_report(y_test, y_pred)
        report_path = "artifacts/classification_report.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        cm_path = "artifacts/confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)

        preds_path = "artifacts/predictions.csv"
        save_predictions_csv(y_test.to_numpy(), y_pred, y_proba, preds_path)
        mlflow.log_artifact(preds_path)

        model_path = "artifacts/model.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print("\n=== HASIL EVALUASI MODEL (DAGSHUB) ===")
        print(f"Akurasi  : {acc:.4f}")
        print(f"Presisi  : {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print("\nClassification Report:")
        print(report)

if __name__ == "__main__":
    data_path = "siswa_clean.csv"
    data = load_data(data_path)
    if data is not None:
        train_and_log_to_dagshub(data)
