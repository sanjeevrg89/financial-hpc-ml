import ray
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@ray.remote
class ChurnPredictor:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, config):
        model = Sequential([
            Dense(config["units_1"], activation="relu", input_shape=(self.X_train.shape[1],)),
            Dropout(config["dropout_1"]),
            Dense(config["units_2"], activation="relu"),
            Dropout(config["dropout_2"]),
            Dense(1, activation="sigmoid")
        ])

        model.compile(optimizer=Adam(learning_rate=config["lr"]),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

        model.fit(
            self.X_train, self.y_train,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            validation_split=0.2,
            verbose=0
        )

        y_pred = (model.predict(self.X_test) > 0.5).astype(int)
        
        return {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred)
        }