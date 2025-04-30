import sys
from pathlib import Path
import threading
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # Added CORS import

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import local modules
from config.config import load_environment
from data.extraction import extract_data
from models.train import train_model, evaluate_model, save_model

app = Flask(__name__)
CORS(app)  # Added CORS initialization
model = None
model_lock = threading.Lock()

# Configuration
FEATURES = {
    'numeric': ['Difference', 'Tenure'],
    'categorical': ['Loan Type', 'Employment Status'],
    'target': 'Defaulted'
}

def create_defaulted_target(df):
    """Create target variable based on client status"""
    return df['Client status (on date)'].str.contains(
        'INACTIVE|INARREARS|BLACKLISTED', 
        case=False, 
        na=False
    ).astype(int)

def validate_input_data(df):
    """Validate input data structure"""
    required_columns = FEATURES['numeric'] + FEATURES['categorical'] + ['Client status (on date)']
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return True

def train_model_async():
    """Background training task"""
    global model
    try:
        print("Starting model training...")
        
        # Extract and prepare data
        df = extract_data()
        df['Defaulted'] = create_defaulted_target(df)
        
        # Validate data
        validate_input_data(df)
        
        # Split data
        X = df[FEATURES['numeric'] + FEATURES['categorical']]
        y = df['Defaulted']
        
        # Train model
        trained_model, X_test, y_test = train_model(X, y)
        evaluate_model(trained_model, X_test, y_test)
        
        # Update model
        with model_lock:
            model = trained_model
            
        # Save model
        save_model(model, "models/latest_model.pkl")
        print("Training completed successfully")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")

@app.route('/train', methods=['POST'])
def trigger_training():
    """Endpoint to start model training"""
    if not request.json or 'force' not in request.json:
        return jsonify({"error": "Missing force parameter"}), 400
    
    # Start training in background
    threading.Thread(target=train_model_async, daemon=True).start()
    return jsonify({"status": "Training started"}), 202

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using current model"""
    global model
    if not model:
        return jsonify({"error": "Model not trained yet"}), 503
    
    try:
        # Get input data
        data = request.json
        input_df = pd.DataFrame([data])
        
        # Validate input
        missing_features = set(FEATURES['numeric'] + FEATURES['categorical']) - set(input_df.columns)
        if missing_features:
            return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400
        
        # Make prediction
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
    """System health status"""
    model_version = None
    if model and Path("models/latest_model.pkl").exists():
        model_version = Path("models/latest_model.pkl").stat().st_mtime
    
    return jsonify({
        "status": "ready" if model else "not_ready",
        "model_version": model_version,
        "features": FEATURES
    })

def load_existing_model():
    """Load model at startup if available"""
    global model
    try:
        model_path = "models/latest_model.pkl"
        if Path(model_path).exists():
            with model_lock:
                model = joblib.load(model_path)
            print(f"Loaded existing model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

if __name__ == '__main__':
    # Load environment variables
    load_environment()
    
    # Create required directories
    Path("models").mkdir(exist_ok=True)
    
    # Load existing model
    load_existing_model()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=10000, threaded=True)