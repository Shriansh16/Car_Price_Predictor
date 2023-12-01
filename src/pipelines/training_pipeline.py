import os
import sys
sys.path.insert(0, 'D:\Car_Price_Prediction\src')
from logger import logging
from exception import CustomException
from utils import *
from components.data_ingestion import *
from components.data_transformation import *
from components.model_trainer import *

try:
    obj1=DataIngestion()
    train_path,test_path=obj1.initiate_data_ingestion()
    obj2=DataTransformation()
    train_arr,test_arr=obj2.initiate_data_transformation(train_path,test_path)
    obj3=ModelTrainer()
    obj3.initiate_model_trainer(train_arr,test_arr)
except Exception as e:
    logging.info("error occured in traning pipeline")
    raise CustomException(e,sys)