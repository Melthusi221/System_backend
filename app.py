import sys
import os
import threading
import joblib
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent)) 

# Import local modules
from config.config import load_environment
from data.extraction import extract_data
from models.train import train_model, evaluate_model, save_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Persistent storage path (Render-compatible)
MODEL_DIR = Path("/data/models")  # Ensure persistent storage
MODEL_PATH = MODEL_DIR / "latest_model.pkl"
model = None
model_lock = threading.Lock()

# Configuration
FEATURES = {
    'numeric': ['Difference', 'Tenure', 'Loan amount', 'Interest rate', 'Client age', 'Loan Cycle'],
    'categorical': ['Loan Type', 'Collateral_Type', 'Client gender'],
    'target': 'Defaulted'
}

def create_defaulted_target(df):
    """Generate target column based on client status."""
    return df['Client status (on date)'].str.contains(
        'INACTIVE|INARREARS|BLACKLISTED', case=False, na=False
    ).astype(int)

def validate_input_data(df):
    """Ensure required features are present."""
    required_columns = FEATURES['numeric'] + FEATURES['categorical'] + ['Client status (on date)']
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return True

def train_model_async():
    """Background training function with forced model loading."""
    global model
    try:
        print("🚀 Starting model training...")
        
        df = extract_data()
        df['Defaulted'] = create_defaulted_target(df)

        validate_input_data(df)

        X = df[FEATURES['numeric'] + FEATURES['categorical']]
        y = df['Defaulted']

        trained_model, X_test, y_test = train_model(X, y)
        evaluate_model(trained_model, X_test, y_test)

        # Ensure persistent storage
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(trained_model, MODEL_PATH)
        print(f"✅ Model training completed and saved at {MODEL_PATH.resolve()}")

        # 🔹 Force model loading right after training 🔹
        with model_lock:
            model = joblib.load(MODEL_PATH)
        print(f"🔄 Model reloaded successfully.")

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")

@app.route('/train', methods=['POST'])
def trigger_training():
    """Endpoint to start model training."""
    threading.Thread(target=train_model_async, daemon=True).start()
    return jsonify({"status": "Training started"}), 202
from datetime import datetime

training_status = {
    "status": "idle", 
    "start_time": None,
    "end_time": None,
    "error": None
}

@app.route('/training-status', methods=['GET'])
def get_training_status():
    return jsonify(training_status)

def train_model_async():
    global training_status
    try:
        training_status = {
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "error": None
        }
        # ... training logic ...
        training_status["status"] = "completed"
        training_status["end_time"] = datetime.now().isoformat()
    except Exception as e:
        training_status = {
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        }

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using trained model."""
    global model
    if not model:
        return jsonify({"error": "Model not trained yet"}), 503

    try:
        data = request.json
        input_df = pd.DataFrame([data])

        missing_features = set(FEATURES['numeric'] + FEATURES['categorical']) - set(input_df.columns)
        if missing_features:
            return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

        with model_lock:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "interpretation": "high risk" if prediction else "low risk"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Directly verify model file exists."""
    model_ready = MODEL_PATH.exists()  # Check file presence
    model_version = MODEL_PATH.stat().st_mtime if model_ready else None

    return jsonify({
        "status": "ready" if model_ready else "not_ready",
        "model_version": model_version,
        "features": FEATURES
    })


def load_existing_model():
    """Load model at startup if available."""
    global model
    try:
        if MODEL_PATH.exists():
            with model_lock:
                model = joblib.load(MODEL_PATH)
            print(f"✅ Loaded existing model from {MODEL_PATH.resolve()}")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")

if __name__ == '__main__':
    # Load environment variables
    load_environment()

    # Create persistent storage
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing model if available
    load_existing_model()

    # Start Flask app with dynamic port assignment
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 10000)), threaded=True)
