from pathlib import Path
import joblib
import pandas as pd
import sys
import traceback
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import json
from data.preprocessing import build_preprocessor
from config.settings import MODEL_PARAMS, FEATURES, TEST_SIZE, RANDOM_STATE


def evaluate_model(model, X_test, y_test):
    """Enhanced evaluation with visualization and metrics"""
    try:
        # Generate predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba)
        }

        # Generate plots
        plt.figure(figsize=(12, 5))
        
        # Precision-Recall Curve
        plt.subplot(1, 2, 1)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve (AP={metrics["pr_auc"]:.2f})')

        # Feature Distribution
        plt.subplot(1, 2, 2)
        pd.Series(y_test).value_counts().plot(kind='bar')
        plt.title('Class Distribution')
        
        plt.savefig('reports/validation_plots.png')
        plt.close()

        # Print key metrics
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
    """Enhanced training with validation"""
    try:
        # Validate input shapes
        if len(X) != len(y):
            raise ValueError(f"Feature-target length mismatch. X: {len(X)}, y: {len(y)}")

        # Check class balance
        class_distribution = y.value_counts(normalize=True)
        print(f"\nClass Distribution:\n{class_distribution}")

        preprocessor = build_preprocessor()
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(**MODEL_PARAMS))
        ])

        # Stratified split with random state
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
            shuffle=True
        )

        # Train with validation
        model.fit(X_train, y_train)
        return model, X_test, y_test
    except Exception as e:
        print(f"Training failed: {str(e)}")
        traceback.print_exc()
        raise


def save_model(model, base_path: str):
    """Versioned model saving with validation"""
    try:
        # Create versioned filename
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(base_path).parent
        model_path = model_dir / f"model_{version}.pkl"

        # Ensure directory exists
        model_dir.mkdir(parents=True, exist_ok=True)

        # Temporary save path
        temp_path = model_path.with_suffix('.tmp')

        # Save with validation
        with open(temp_path, 'wb') as f:
            joblib.dump(model, f, protocol=4)

        # Validate saved model
        try:
            with open(temp_path, 'rb') as f:
                joblib.load(f)  # Test load
        except Exception as e:
            raise ValueError(f"Model validation failed: {str(e)}")

        # Atomic replacement
        if model_path.exists():
            model_path.unlink()
        temp_path.rename(model_path)

        print(f"✅ Model saved to: {model_path.resolve()}")
        return model_path
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"❌ Model save failed: {str(e)}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        # Configure paths
        Path("reports").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)

        # Data loading with validation
        from data.extraction import extract_data
        df = extract_data()

        # Feature validation
        required_features = FEATURES['numeric'] + FEATURES['categorical']
        missing = list(set(required_features) - set(df.columns))
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Data preparation
        X = df[required_features]
        y = df['Defaulted']

        # Training pipeline
        model, X_test, y_test = train_model(X, y)
        metrics = evaluate_model(model, X_test, y_test)

        model_path = save_model(model, "models/production_model.pkl")

        # Save metrics
        with open("reports/latest_metrics.json", "w") as f:
            json.dump(metrics, f)

    except Exception as e:
        print(f"\n❌ Critical failure: {str(e)}")
        traceback.print_exc()
        sys.exit(1)