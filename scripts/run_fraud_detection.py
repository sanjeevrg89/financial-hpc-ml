import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import yaml
import os

from src.data_generation.fraud_data_generator import generate_fraud_data
from src.preprocessing.fraud_preprocessing import preprocess_fraud_data
from src.models.fraud_detection import FraudDetectionTrainer
from src.utils.data_loader import load_data

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'fraud_config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Generate data
    data = generate_fraud_data(config['data_samples'])
    data.to_csv('fraud_data.csv', index=False)

    # Load and preprocess data
    df = load_data('/tmp/fraud_data.csv')
    X_train, X_test, y_train, y_test = preprocess_fraud_data(df)

    # Initialize Ray
    ray.init()

    # Set up the trainer
    trainer = FraudDetectionTrainer.remote(X_train, X_test, y_train, y_test)

    # Run hyperparameter tuning
    scheduler = ASHAScheduler(
        metric="f1",
        mode="max",
        max_t=config['max_epochs'],
        grace_period=1,
        reduction_factor=2
    )

    result = tune.run(
        lambda trial_config: trainer.train_model.remote(trial_config),
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