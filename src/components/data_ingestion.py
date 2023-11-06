import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
sys.path.insert(0, 'D:\Car_Price_Prediction\src')
from logger import logging
from exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path=os.path.join('artifacts','raw.csv')
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            raw=pd.read_csv('D:\Car_Price_Prediction\notebooks\cleaned_data_final.csv')
            logging.info('dataset is taken')
            train_data,test_data=train_test_split(raw,test_size=0.20,random_state=42)
            raw.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            logging.info('raw dataset is successfully saved')
            train_data.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            logging.info('train dataset is succerssfully saved')
            test_data.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            logging.info('test dataset is succerssfully saved')
        except Exception as e:
            logging.info('ERROR OCCURED DURING DATA INGESTION')
            raise CustomException(e,sys)
        

