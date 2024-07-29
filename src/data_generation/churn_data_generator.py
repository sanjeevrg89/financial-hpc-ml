import pandas as pd
import numpy as np

def generate_churn_data(n_samples=100000):
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 80, size=n_samples),
        'tenure': np.random.randint(0, 10, size=n_samples),
        'balance': np.random.lognormal(mean=8, sigma=1, size=n_samples),
        'num_products': np.random.randint(1, 5, size=n_samples),
        'credit_score': np.random.randint(300, 850, size=n_samples),
        'is_active': np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8]),
        'estimated_salary': np.random.lognormal(mean=10, sigma=0.5, size=n_samples),
    }
    
    df = pd.DataFrame(data)
    
    churn_prob = (
        (df['age'] > 60) * 0.2 +
        (df['tenure'] < 2) * 0.3 +
        (df['balance'] < 1000) * 0.2 +
        (df['num_products'] == 1) * 0.2 +
        (df['credit_score'] < 600) * 0.2 +
        (df['is_active'] == 0) * 0.4 +
        (df['estimated_salary'] < np.percentile(df['estimated_salary'], 25)) * 0.2
    )
    
    df['churned'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    return df