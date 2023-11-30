import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'D:\Car_Price_Prediction\src')
from logger import logging
from exception import CustomException
from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.sparse import csr_matrix
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    data_transformation_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformation(self):
        try:

            logging.info('preparing preprocessor')
            num_columns=['year', 'kms_covered']
            cat_columns=['company', 'fuel_type']
            preprocessor=ColumnTransformer([('ohe',OneHotEncoder(),cat_columns),('standardScaler',StandardScaler(),num_columns)])
            logging.info('preprocessor is successfully prepared')
            return preprocessor
        except Exception as e:
            logging.info("ERROR OCCURED DURING PREPARATION OF PREPROCESSOR")
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path)
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            input_train_features=train_df.drop('price',axis=1)
            target_train=train_df['price']
            input_test_features=test_df.drop('price',axis=1)
            target_test=test_df['price']
            logging.info('data transformation started')
            pre_obj=self.get_data_transformation()
            input_train_tr=pre_obj.fit_transform(input_train_features)
            input_test_tr=pre_obj.transform(input_test_features)
            input_train_tr=input_train_tr.toarray()
            input_test_tr=input_test_tr.toarray()
            logging.info("preprocessor done")
            train_arr=np.c_[input_train_tr,np.array(target_train)]
            test_arr=np.c_[input_test_tr,np.array(target_test)]
            save_object(self.data_transformation_config.data_transformation_path,pre_obj)
            return (train_arr,test_arr)
        except Exception as e:
            logging.info("error occured in preproceesing step")
            raise CustomException(e,sys)


            
            
