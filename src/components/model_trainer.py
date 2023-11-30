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
            