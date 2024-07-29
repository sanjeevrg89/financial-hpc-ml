import pandas as pd
import numpy as np

def generate_sentiment_data(n_samples=10000):
    np.random.seed(42)
    
    positive_headlines = [
        "Company X reports record profits",
        "Stock market reaches all-time high",
        "New technology breakthrough boosts investor confidence",
        "Unemployment rate drops to historic low",
        "Merger creates new market leader"
    ]

    negative_headlines = [
        "Economic recession fears grow",
        "Company Y announces major layoffs",
        "Trade tensions escalate between major economies",
        "Cybersecurity breach impacts millions of customers",
        "Regulatory crackdown hits industry hard"
    ]

    neutral_headlines = [
        "Federal Reserve maintains current interest rates",
        "Quarterly earnings report in line with expectations",
        "New CEO appointed at Company Z",
        "Industry conference discusses future trends",
        "Market awaits upcoming economic data release"
    ]

    headlines = []
    sentiments = []

    for _ in range(n_samples):
        sentiment = np.random.choice(['positive', 'negative', 'neutral'])
        if sentiment == 'positive':
            headline = np.random.choice(positive_headlines)
            sentiments.append(2)
        elif sentiment == 'negative':
            headline = np.random.choice(negative_headlines)
            sentiments.append(0)
        else:
            headline = np.random.choice(neutral_headlines)
            sentiments.append(1)
        headlines.append(headline)

    df = pd.DataFrame({
        'text': headlines,
        'sentiment': sentiments
    })

    return df