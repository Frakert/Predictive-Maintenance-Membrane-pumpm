# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:17:27 2023

@author: klabbf
"""
import os
import pandas as pd

from Membrane_Model_Class import Membrane_Model

if __name__ ==  '__main__':
    # Import data from csv and fix little formating
    test_data=pd.read_csv(os.getcwd()+'\\Test_Data.csv',parse_dates=[1],index_col=[0])
    test_data.rename(columns={'0':'Date'},inplace=True)
    test_data.head()

    # 3 lines to iniate the model, clean the data, and make predictions
    Membrane_Model=Membrane_Model(test_data)
    Membrane_Model.clean_data()
    Membrane_Model.predict()
    
    # Get variables from the class
    X=Membrane_Model.X
    model=Membrane_Model.model
    predictions=Membrane_Model.predictions