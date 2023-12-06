import pandas as pd
import os
import sys
sys.path.insert(0, 'D:\Car_Price_Prediction\src')
from logger import logging
from exception import CustomException
import pickle
from sklearn.metrics import r2_score


def save_object(path,object):
    try:
        dir_path=os.path.dirname(path)
        os.makedirs(dir_path,exist_ok=True)
        with open(path,'wb') as file_obj:
           pickle.dump(object,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(path):
    try:
        with open(path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('ERROR OCCURRED DURING LOADING THE OBJECT')
        raise CustomException(e, sys)
def model_evaluation(models,X_train,y_train,X_test,y_test):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            pred=model.predict(X_test)
            report[list(models.keys())[i]]=r2_score(y_test,pred)
        return report
    except Exception as e:
        logging.info("error occured during model evaluation")
        raise CustomException(e,sys)
