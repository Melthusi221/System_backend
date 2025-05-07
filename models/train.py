import os
import joblib
import pandas as pd
import traceback
from datetime import datetime
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import json

from data.preprocessing import build_preprocessor
from data.extraction import extract_data  # Ensure this works in Render

# Load environment variables
MODEL_PARAMS = json.loads(os.getenv("MODEL_PARAMS", "{}"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
MODEL_DIR = Path("/data/models")  # Persistent storage location
REPORT_DIR = Path("/data/reports")  # Ensure reports are saved persistently

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and visualize results."""
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba)
        }

        plt.figure(figsize=(12, 5))

        # Precision-Recall Curve
        plt.subplot(1, 2, 1)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve (AP={metrics["pr_auc"]:.2f})')

        # Class Distribution
        plt.subplot(1, 2, 2)
        pd.Series(y_test).value_counts().plot(kind='bar')
        plt.title('Class Distribution')

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(REPORT_DIR / 'validation_plots.png')
        plt.close()

        print("\n=== Model Evaluation ===")
        print(f"AUC-ROC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return metrics

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()
        raise

def train_model(X: pd.DataFrame, y: pd.Series):
    """Train model with validation."""
    try:
        if len(X) != len(y):
            raise ValueError(f"Feature-target length mismatch. X: {len(X)}, y: {len(y)}")

        class_distribution = y.value_counts(normalize=True)
        print(f"\nClass Distribution:\n{class_distribution}")

        preprocessor = build_preprocessor()

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(**MODEL_PARAMS))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y, shuffle=True
        )

        model.fit(X_train, y_train)
        return model, X_test, y_test

    except Exception as e:
        print(f"Training failed: {str(e)}")
        traceback.print_exc()
        raise

def save_model(model):
    """Save trained model with persistent storage."""
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = MODEL_DIR / f"model_{version}.pkl"

        joblib.dump(model, model_path)
        print(f"✅ Model saved to: {model_path.resolve()}")
        return model_path
    except Exception as e:
        print(f"❌ Model save failed: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        df = extract_data()

        required_features = FEATURES['numeric'] + FEATURES['categorical']
        missing = list(set(required_features) - set(df.columns))
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X = df[required_features]
        y = df['Defaulted']

        model, X_test, y_test = train_model(X, y)
        metrics = evaluate_model(model, X_test, y_test)
        model_path = save_model(model)

        with open(REPORT_DIR / "latest_metrics.json", "w") as f:
            json.dump(metrics, f)

    except Exception as e:
        print(f"\n❌ Critical failure: {str(e)}")
        traceback.print_exc()
