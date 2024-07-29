import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import yaml
import os

from src.data_generation.sentiment_data_generator import generate_sentiment_data
from src.preprocessing.sentiment_preprocessing import preprocess_sentiment_data
from src.models.sentiment_analysis import SentimentAnalyzer
from src.utils.data_loader import load_data

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'sentiment_config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Generate data
    data = generate_sentiment_data(config['data_samples'])
    data.to_csv('sentiment_data.csv', index=False)

    # Load and preprocess data
    df = load_data('/tmp/sentiment_data.csv')
    train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels = preprocess_sentiment_data(df)

    # Initialize Ray
    ray.init()

    # Set up the analyzer
    analyzer = SentimentAnalyzer.remote(train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels)

    # Run hyperparameter tuning
    scheduler = ASHAScheduler(
        metric="f1",
        mode="max",
        max_t=config['max_epochs'],
        grace_period=1,
        reduction_factor=2
    )

    result = tune.run(
        lambda trial_config: analyzer.train_model.remote(trial_config),
        resources_per_trial={"cpu": 2, "gpu": config['gpus_per_trial']},
        config=config['hyperparameters'],
        num_samples=config['num_samples'],
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("f1", "max", "last")
    print("Best trial config:", best_trial.config)
    print("Best trial final F1 score:", best_trial.last_result["f1"])

if __name__ == "__main__":
    main()