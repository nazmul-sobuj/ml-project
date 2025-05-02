import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessing_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numeric_columns = ['reading score', 'writing score']
            categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy= 'median')),
                    ('scaler', StandardScaler(with_mean=False))
                    
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

                ]
            )
            logging.info(f"Numerical columns are {numeric_columns}")
            logging.info(f"Categorical columns are {categorical_columns}")

            preprocessor = ColumnTransformer(transformers=
                [
                    ('num_pipeline', num_pipeline, numeric_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = "math score"
            numeric_columns = ['reading score', 'writing score']
            input_feature_train_df = train_df.drop(columns= [target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns= [target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Applying preprocessing object on traning data frame and test data frame")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info("Save preprocessing object")


            save_object(

                file_path=  self.data_transformation_config.preprocessing_obj_file_path,
                obj= preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)




        
