import sys
import pickle
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Update these paths based on your project structure
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def predict(self, features):
        try:
            # Load preprocessor and model
            with open(self.preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)

            # Preprocess input features
            transformed_features = preprocessor.transform(features)

            # Make prediction
            prediction = model.predict(transformed_features)

            return prediction

        except Exception as e:
            raise Exception(f"Prediction pipeline failed: {e}")


class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    

    def get_data_as_dataframe(self):
        try:
            data = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score]
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise Exception(f"Error creating dataframe: {e}")



