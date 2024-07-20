import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    transformed_train_data_path: str = os.path.join('artifacts', "transformed_train.csv")
    transformed_test_data_path: str = os.path.join('artifacts', "transformed_test.csv")
    scaler_path: str = os.path.join('artifacts', "scaler.pkl")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_data_path: str, test_data_path: str):
        """
        Initiates the data transformation process by reading the train and test data,
        applying preprocessing steps, and saving the transformed data.

        Parameters:
        train_data_path (str): The path to the training data file.
        test_data_path (str): The path to the testing data file.

        Returns:
        tuple: Paths to the transformed training and testing data files.
        """
        logging.info("Entered the data transformation method")
        try:
            # Read the train and test datasets
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info(f'Train and test datasets read from {train_data_path} and {test_data_path}')

            # Define numerical and categorical columns
            numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

            # Define transformers for numerical and categorical features
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine transformers into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            # Fit the preprocessor on the training data and transform both train and test data
            train_transformed = preprocessor.fit_transform(train_df)
            test_transformed = preprocessor.transform(test_df)
            logging.info("Data transformation completed")

            # Ensure the artifacts directory exists
            os.makedirs(os.path.dirname(self.transformation_config.transformed_train_data_path), exist_ok=True)

            # Convert transformed arrays back to dataframes with appropriate column names
            transformed_train_df = pd.DataFrame(train_transformed, columns=preprocessor.get_feature_names_out())
            transformed_test_df = pd.DataFrame(test_transformed, columns=preprocessor.get_feature_names_out())

            # Save the transformed datasets
            transformed_train_df.to_csv(self.transformation_config.transformed_train_data_path, index=False, header=True)
            transformed_test_df.to_csv(self.transformation_config.transformed_test_data_path, index=False, header=True)
            logging.info("Transformed data saved")

            return self.transformation_config.transformed_train_data_path, self.transformation_config.transformed_test_data_path

        except Exception as e:
            raise CustomException(e, sys)
