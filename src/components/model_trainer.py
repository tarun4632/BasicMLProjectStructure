import os
import sys
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', "model.pkl")
    report_path: str = os.path.join('artifacts', "report.txt")


class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_data_path: str, test_data_path: str):
        """
        Initiates the model training process by reading the train and test data,
        training the model, evaluating it, and saving the trained model and report.

        Parameters:
        train_data_path (str): The path to the training data file.
        test_data_path (str): The path to the testing data file.

        Returns:
        str: Path to the saved trained model file.
        """
        logging.info("Entered the model training method")
        try:
            # Read the train and test datasets
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info(f'Train and test datasets read from {train_data_path} and {test_data_path}')

            # Assuming the last column is the target variable
            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            # Initialize and train the model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            logging.info("Model training completed")

            # Evaluate the model
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            logging.info(f"Model evaluation completed with accuracy: {accuracy}")

            # Save the evaluation report
            with open(self.trainer_config.report_path, 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write("Classification Report:\n")
                f.write(report)

            # Save the trained model to a .pkl file
            os.makedirs(os.path.dirname(self.trainer_config.model_path), exist_ok=True)
            with open(self.trainer_config.model_path, 'wb') as file:
                pickle.dump(model, file)
            logging.info(f"Trained model saved at {self.trainer_config.model_path}")

            return self.trainer_config.model_path

        except Exception as e:
            raise CustomException(e, sys)