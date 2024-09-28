import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.utils import save_object,evaluate_models
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','Model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and test input data")
            x_train,y_train,x_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models={
                "Linear Regression": LinearRegression(),
                # "KNearest Neighbour": KNeighborsRegressor(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "ElasticNet":ElasticNet(),                
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "Catboost": CatBoostRegressor(verbose=False),
                "Adaboost": AdaBoostRegressor()
            }

            params={
                "Linear Regression":{},
                "Decision Tree":{
                    "criterion":['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "splitter":['best','random'],
                    "max_features":['sqrt','log2']
                },
                "Ridge":{
                    'alpha': [0.1, 1, 10, 100, 1000]
                },
                "Lasso":{
                    'alpha': [0.1, 1, 10, 100, 1000]
                },
                "ElasticNet":{
                    'alpha': [0.1, 1, 10, 100],  # Regularization strength
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0]  # Balance between L1 and L2 regularization
                },                
                "Random Forest":{
                    'n_estimators': [10,20,50,100,200]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [10,20,50,100,200],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]
                },
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [10,20,50,100,200]
                },
                "Catboost":{
                    'depth': [5,10,15],
                    'learning_rate': [0.1,0.05,0.01,.001],
                    'iterations': [10,20,50,100]
                },
                "Adaboost":{
                    'n_estimators': [10,20,50,100,200],
                    'learning_rate':[0.1,0.05,0.01,.001]
                }
            }

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                             models=models,param=params)
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("best model found on traing and testing data")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2_score_pred=r2_score(y_test,predicted)
            return best_model_name,r2_score_pred
        except Exception as e:
            raise CustomException(e,sys)    

