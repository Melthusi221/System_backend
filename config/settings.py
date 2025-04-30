# Configuration constants
MODEL_PARAMS = {
    'solver': 'lbfgs',
    'max_iter': 1000,
    'class_weight': 'balanced',
    'random_state': 42
}

FEATURES = {
    'numeric': ['Loan amount', 'Interest rate', 'Tenure', 'Client age', 'Loan Cycle'],
    'categorical': ['Client gender', 'Loan purpose'],
    'target': 'Defaulted'
}

TEST_SIZE = 0.3
RANDOM_STATE = 42