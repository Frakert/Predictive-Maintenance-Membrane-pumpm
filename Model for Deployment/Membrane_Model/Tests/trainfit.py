# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:43:27 2023

@author: klabbf
"""

#%%

import os
import pandas as pd

from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
curr_path = dirname(__file__)
dependencies_path = abspath(join(curr_path, '..', 'Dependencies'))
class_path = abspath(join(curr_path, '..'))
database_path = abspath(join(curr_path, '..', 'Output'))
sys.path.append(class_path)

import pyodbc


if __name__ ==  '__main__':
    # Import data from csv and fix little formating
    test_data=pd.read_csv("C:/Users\klabbf/OneDrive - Canon Production Printing Netherlands B.V/Documents/Data-Excel/Python scripting"+'\\Data_%5IAL%301%_from_2022-04-10_12_00_until_2023-06-05_12_00.csv',parse_dates=[1],sep=":",index_col=[0])
    test_data.rename(columns={'0':'Date'},inplace=True)
    test_data.head()
    print('Data imported')

#%%

if __name__ ==  '__main__':
    
    
    from Membrane_Model_Class import Membrane_Model
    
    # 3 lines to iniate the model, clean the data, and make predictions
    Membrane_Model=Membrane_Model(test_data)
    Membrane_Model.clean_data()
    Membrane_Model.predict()
    print('Predictions made')
    
    # Get variables from the class
    X=Membrane_Model.X
    model=Membrane_Model.model
    predictions=Membrane_Model.predictions
    X.to_excel('C:/Users/klabbf/Downloads/X_fiter.xlsx')

    # Write to SQL database
    SQL_conn_string = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                 r'DBQ=%s\Predictions_Membrane_Model.accdb;')%format(database_path)
    Membrane_Model.predictions_to_SQL(SQL_conn_string,'BatchName,[DateTime],Prediction')

    

#%%
if __name__ ==  '__main__':
    import sqlalchemy # Dependency: sqlalchemy-access
    
    from sqlalchemy import create_engine


    con_string="access+pyodbc://%s" %('MS AC')
    engine = create_engine(con_string)
    

    predictions.to_sql('Train_set',con=engine,index=False,if_exists='replace')
#%%
