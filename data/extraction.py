# app/data/extraction.py
import requests
import pandas as pd
from datetime import date
from config.config import load_environment

def fetch_api_data(payload: dict) -> requests.Response:
    """Make authenticated API request"""
    credentials = load_environment()
    url = 'https://wisrod.instafin.com/submit/cube.LoanAnalysisFromConfiguration'
    return requests.post(
        url,
        auth=(credentials['API_USERNAME'], credentials['API_PASSWORD']),
        json=payload
    )

def process_response(response: requests.Response) -> pd.DataFrame:
    """Process API response into DataFrame"""
    data = response.json()
    results = data.get('results', [])
    
    if not results:
        raise ValueError("No results found in API response")
    
    df = pd.DataFrame(results)
    df.columns = df.iloc[0]
    df = df[1:]
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform raw data"""
    df['Maturity date date'] = pd.to_datetime(df['Maturity date date'], format='%Y-%m-%d', errors='coerce')
    df['Schedule start date'] = pd.to_datetime(df['Schedule start date'], format='%Y-%m-%d', errors='coerce')
    df['Difference'] = (df['Maturity date date'] - df['Schedule start date']).dt.days
    df['Difference'] = df['Difference'].fillna(0)
    df['Tenure'] = df['Difference']/31
    df = df[df['Tenure'] != 0]
    
    return df

def extract_data() -> pd.DataFrame:
    """Main extraction pipeline"""
    payload = {
        "date": date.today().strftime("%Y-%m-%d"),
        "configurationID": "30166d2d-ebdd-435c-8756-a74d00c4418f",
        # ... rest of payload ...
    }
    
    response = fetch_api_data(payload)
    if response.status_code != 200:
        raise ConnectionError(f"API request failed with status {response.status_code}")
    
    raw_df = process_response(response)
    return clean_data(raw_df)