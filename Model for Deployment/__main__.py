# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:17:27 2023

@author: klabbf
"""
import os
import pandas as pd

import pyodbc

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
    
    
    
#%%
import os
import pandas as pd

import pyodbc

database_string = os.getcwd() + '\Predictions_Membrane_Model.accdb'
print(database_string)

conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=%s\Predictions_Membrane_Model.accdb;')%format(os.getcwd())
conn = pyodbc.connect(conn_str)

#%%

cursor = conn.cursor()
for i in cursor.tables(tableType='TABLE'):
    print(i.table_name)

cursor.execute(
    """
    INSERT INTO Predictions (BatchName,[DateTime],Prediction)
    VALUES ('KP5512307602','2023-03-06 12:01:00','1');
    """
    )

cursor.commit()




#%%
#predictions.to_sql('Predictions',con=conn,index=False,if_exists='append')