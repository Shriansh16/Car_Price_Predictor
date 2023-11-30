import pandas as pd
import os
import sys
sys.path.insert(0, 'D:\Car_Price_Prediction\src')
from logger import logging
from exception import CustomException
import pickle


def save_object(path,object):
    try:
        dir_path=os.path.dirname(path)
        os.makedirs(dir_path,exist_ok=True)
        with open(path,'wb') as file_obj:
           pickle.dump(object,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(path,object):
    try:
        with open(path,'wb') as file_obj:
            return pickle.load(path)
    except Exception as e:
        raise CustomException(e,sys)