import pandas as pd
import numpy as np

def generate_fraud_data(n_samples=100000):
    np.random.seed(42)
    data = {
        'transaction_amount': np.random.lognormal(mean=4, sigma=1, size=n_samples),
        'time_of_day': np.random.randint(0, 24, size=n_samples),
        'is_weekend': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'location_match': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
        'unusual_frequency': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'high_risk_merchant': np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03]),
    }
    
    df = pd.DataFrame(data)
    
    fraud_prob = (
        (df['transaction_amount'] > np.percentile(df['transaction_amount'], 95)) * 0.3 +
        (df['time_of_day'].between(0, 5)) * 0.2 +
        (df['is_weekend'] == 1) * 0.1 +
        (df['location_match'] == 0) * 0.2 +
        (df['unusual_frequency'] == 1) * 0.3 +
        (df['high_risk_merchant'] == 1) * 0.4
    )
    
    df['is_fraud'] = (np.random.random(n_samples) < fraud_prob).astype(int)
    
    return df