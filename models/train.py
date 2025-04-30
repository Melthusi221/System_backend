from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from data.preprocessing import build_preprocessor

from config.settings import MODEL_PARAMS, FEATURES, TEST_SIZE, RANDOM_STATE

def validate_model_file(path: str) -> bool:
    """Check if file exists and is a valid model file"""
    path = Path(path)
    if not path.exists():
        print(f"File {path} does not exist")
        return False
    
    # Basic file validation
    if path.stat().st_size == 0:
        print(f"File {path} is empty")
        return False
    
    return True

def train_model(X: pd.DataFrame, y: pd.Series):
    """Train and evaluate the logistic regression model"""
    preprocessor = build_preprocessor()

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(**MODEL_PARAMS))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nAUC-ROC Score:", roc_auc_score(y_test, y_proba))

def save_model(model, path: str):
    """Safely save model with atomic write"""
    path = Path(path)
    temp_path = path.with_suffix('.tmp')
    
    try:
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with highest protocol and in binary mode
        with open(temp_path, 'wb') as f:
            joblib.dump(model, f, protocol=4)
        
        # Verify the saved file
        if not validate_model_file(temp_path):
            raise ValueError("Saved file failed validation")
            
        # Test load the model
        with open(temp_path, 'rb') as f:
            joblib.load(f)  # Test loading
            
        # Replace old file if everything succeeded
        if path.exists():
            path.unlink()
        temp_path.rename(path)
        
        print(f"✅ Model successfully saved to {path.absolute()}")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"❌ Failed to save model: {str(e)}")

if __name__ == "__main__":
    # Load your actual data
    try:
        df = pd.read_excel("C:/Users/HP/OneDrive/Documents/Python Scripts/MFI Credit Scoring Model/loan_data.xlsx")
    except Exception as e:
        raise FileNotFoundError("❌ Failed to load data. Make sure 'loan_data.xlsx' exists.") from e

    if df.empty:
        raise ValueError("❌ Loaded data is empty! Please check your dataset.")

    # Create Defaulted target variable
    df['Defaulted'] = df['Client status (on date)'].str.contains(
        'INACTIVE|INARREARS|BLACKLISTED', 
        case=False, 
        na=False
    ).astype(int)

    # Check for required columns
    required_columns = FEATURES['numeric'] + FEATURES['categorical'] + ['Client status (on date)']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"❌ Missing required columns: {missing_columns}")

    # Prepare features and target
    X = df[FEATURES['numeric'] + FEATURES['categorical']]
    y = df['Defaulted']

    # Train and evaluate
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)
    
    # Save model
    model_path = "models/model.pkl"
    save_model(model, model_path)
    
    # Final verification
    try:
        with open(model_path, 'rb') as f:
            loaded_model = joblib.load(f)
        print("\n✅ Final verification: Model loaded successfully!")
    except Exception as e:
        print(f"\n❌ Final verification failed: {str(e)}")
        if Path(model_path).exists():
            print(f"File size: {Path(model_path).stat().st_size} bytes")