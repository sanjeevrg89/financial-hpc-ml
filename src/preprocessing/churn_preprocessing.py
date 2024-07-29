from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_churn_data(df):
    X = df.drop('churned', axis=1)
    y = df['churned']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test