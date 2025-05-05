import pandas as pd
from pathlib import Path

EXCEL_PATH = r"C:\Users\HP\OneDrive\Documents\Python Scripts\MFI Credit Scoring Model\loan_data.xlsx"

def extract_data(use_excel=True) -> pd.DataFrame:
    """Main extraction pipeline with enhanced error handling"""
    try:
        if use_excel:
            df = _extract_from_excel()
        else:
            df = _extract_from_api()
            
        df = _add_defaulted_target(df)
        
        if df is None:
            raise ValueError("Data extraction returned None")
        if df.empty:
            raise ValueError("Empty DataFrame after processing")
            
        return df
        
    except Exception as e:
        print(f"CRITICAL EXTRACTION ERROR: {str(e)}")
        raise

def _extract_from_excel() -> pd.DataFrame:
    """Load data from Excel with validation"""
    try:
        path = Path(EXCEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Excel file missing at {EXCEL_PATH}")
            
        raw_df = pd.read_excel(path, engine='openpyxl')
        print(f"Raw data loaded: {raw_df.shape[0]} records")
        
        return clean_data(raw_df)
        
    except Exception as e:
        print(f"EXCEL EXTRACTION FAILED: {str(e)}")
        raise

def _add_defaulted_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create Defaulted column with validation"""
    if df is None:
        raise ValueError("Cannot process None DataFrame")
        
    required_col = 'Client status (on date)'
    if required_col not in df.columns:
        raise KeyError(f"Missing critical column: {required_col}")
        
    df['Defaulted'] = df[required_col].str.contains(
        r'(?:INACTIVE|INARREARS|BLACKLISTED)',  # Non-capturing group
        case=False, 
        na=False, 
        regex=True
    ).astype(int)
    
    # Validate target creation
    if 'Defaulted' not in df.columns:
        raise RuntimeError("Failed to create Defaulted column")
    if df['Defaulted'].isnull().any():
        raise ValueError("Null values in Defaulted column")
        
    return df
#Data Cleaning
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep ALL required columns including Loan amount"""
    keep_columns = [
        'Client status (on date)',
        'Difference', 
        'Tenure',
        'Loan purpose',           # Raw column name from Excel
        'Loan collateral types',  # Raw column name from Excel
        'Loan amount',
        'Interest rate' ,
        'Client age',
        'Loan Cycle',
        'Client gender'                                                                                                                                                                                                        # MUST BE PRESENT <--- Add this line
    ]
    
    clean_df = df[keep_columns].copy()
    
    # Rename to match FEATURES config
    clean_df = clean_df.rename(columns={
        'Loan purpose': 'Loan Type',
        'Loan collateral types': 'Collateral_Type'
    })
    
    # Convert numeric columns
    for col in ['Difference', 'Tenure', 'Loan amount', 'Interest rate', 'Client age', 'Loan Cycle']:
        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
    
    return clean_df.dropna()