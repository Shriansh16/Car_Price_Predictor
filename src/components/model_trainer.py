import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
sys.path.insert(0, 'D:\Car_Price_Prediction\src')
from logger import logging
from exception import CustomException
from utils import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

@dataclass
class ModelTrainerConfig:
    model_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            X_train,y_train=train_arr[:,:-1],train_arr[:,-1]
            X_test,y_test=test_arr[:,:-1],test_arr[:,-1]
            logging.info("TRAINING STARTED")
            models={'Linear_Regression':LinearRegression(),'Random_Forest_Regressor':RandomForestRegressor(),
                    'Adaboost_regressor':AdaBoostRegressor(),'Gradientboosting_Regressor':GradientBoostingRegressor(),
                    'Decision_Tree_Regressor':DecisionTreeRegressor()}
            report=model_evaluation(models,X_train,y_train,X_test,y_test)
            logging.info('TRAINING COMPLETED')
            logging.info(f'training report: {report}')
            logging.info("SELECTING BEST MODEL")
            best_model_score=max(sorted(list(report.values())))
            best_model_name=list(report.keys())[list(report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            best_model.fit(X_train,y_train)
            logging.info(f'best model is {best_model_name} which r2 score {best_model_score}')
            logging.info('saving the best model')
            save_object(self.model_trainer_config.model_path,best_model)
        except Exception as e:
            logging.info("ERROR OCCURED DURING MODEL TRAINING")
            raise CustomException(e,sys)

