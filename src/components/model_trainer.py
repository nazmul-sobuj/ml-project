import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import save_object, evaluate_models
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from dataclasses import dataclass
@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
         try:
             logging.info("Splitting train and test data")
             X_train, y_train, X_test, y_test =(
                  train_array[:, :-1],
                  train_array[:, -1],
                  test_array[:, :-1],
                  test_array[:, -1]
             )
           
             models = {
                 "AdaboostBoost": AdaBoostRegressor(),
                 "GradientBoost": GradientBoostingRegressor(),
                 "Xgboost": XGBRegressor(),
                 "RandomForest": RandomForestRegressor(),
                 "Catboost": CatBoostRegressor(verbose=False),
                 "LinearModel": LinearRegression(),
                 "Kneighbors": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
             }

             model_report:dict= evaluate_models(X_train=X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)
             best_model_score = max(model_report.values())
             best_model_name = list(model_report.keys())[
             list(model_report.values()).index(best_model_score)]
        
             best_model = models[best_model_name]

             if best_model_score < 0.6:
                  raise CustomException("No suitable model found (score < 0.6)", sys)

                 
            

             logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
        
        # ✅ Train the best model now
             best_model.fit(X_train, y_train)

             save_object(
             file_path=self.model_trainer_config.trained_model_file_path,
             obj=best_model
             )
        

             predicted = best_model.predict(X_test)
             r2_square = r2_score(y_test, predicted)
             return r2_square
         except Exception as e:
             raise CustomException

    

            
           
            
        
       
        