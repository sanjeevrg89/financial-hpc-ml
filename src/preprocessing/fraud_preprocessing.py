from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_fraud_data(df):
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test