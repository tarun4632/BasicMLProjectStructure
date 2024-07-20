import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', "raw.csv")
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    test_size: float = 0.2  # Default test size for splitting the dataset


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, data_source: str):
        """
        Initiates the data ingestion process by reading the dataset from the specified data source,
        splitting it into training and testing sets, and saving these sets to CSV files.

        Parameters:
        data_source (str): The path to the input data file (e.g., a CSV file).

        Returns:
        tuple: Paths to the training and testing data files.
        """
        logging.info("Entered the data ingestion method")
        try:
            # Read the dataset from the provided data source
            df = pd.read_csv(data_source)
            logging.info(f'Dataset read from {data_source}')

            # Ensure the artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")

            # Split the dataset into train and test sets
            train_df, test_df = train_test_split(df, test_size=self.ingestion_config.test_size, random_state=42)
            logging.info("Train-test split done")

            # Save the train and test sets to CSV files
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data saved")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
