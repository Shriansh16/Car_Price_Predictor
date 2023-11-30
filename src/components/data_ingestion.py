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
            logging.info("TAKING RAW DATASET")
            raw_data=pd.read_csv('D:\Car_Price_Prediction/notebooks/cleaned_data_final.csv')
            logging.info("RAW DATA SET IS READ AS DF")
            train,test=train_test_split(raw_data,test_size=0.20,random_state=42)
            logging.info("DATASET IS SPLITTED INFO TRAIN AND TEST DATA")
            raw_data.drop('Unnamed: 0',axis=1,inplace=True)
            train.drop('Unnamed: 0',axis=1,inplace=True)
            test.drop('Unnamed: 0',axis=1,inplace=True)
            logging.info(train.head())
            logging.info("saving raw,test and train dataset")
            raw_data.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            train.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            logging.info("all dataset saved successfully")
            return(self.data_ingestion_config.train_data_path,self.data_ingestion_config.test_data_path)
        except Exception as e:
            logging.info("error occured during data ingestion")
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()



        

