import os
import sys
import pandas as pd
sys.path.insert(0, 'D:\Car_Price_Prediction\src')
from logger import logging
from exception import CustomException
from utils import *


class Predict_Pipeline:
    def __init__(self):
        pass
    def predict(self,feature):
        try:

            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            scaled_data=preprocessor.transform(feature)
            pred=model.predict(scaled_data)
            return pred
        except Exception as e:
            logging.info('ERROR OCCURED DURING PREDICTION')
            raise CustomException(e,sys)

class CustomData:
        def __init__(self,
                 company:str,
                 year:int,
                 kms_covered:int,
                 fuel_type:str,
                 ):
        
            self.company=company
            self.year=year
            self.kms_covered=kms_covered
            self.fuel_type=fuel_type
            

        def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                'company':[self.company],
                'year':[self.year],
                'kms_covered':[self.kms_covered],
                'fuel_type':[self.fuel_type],
                
                 }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise CustomException(e,sys)