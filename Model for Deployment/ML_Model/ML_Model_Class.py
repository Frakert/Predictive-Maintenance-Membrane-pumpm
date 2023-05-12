# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:51:18 2023

@author: klabbf
"""


class ML_Model:
    """
    @author: klabbf aka Freek Klabbers, graduation intern at CPP M2 Maintenance
    
    This Class violates some OOP rules im sure but is designed to be an encapsulaton and abstraction for my ML model.
    It needs only 3 lines for full functionality, call (in order):
        - Membrane_Model(raw_data) - Iniaties model with the data 
        - .clean_data() - Cleanes the data in a way the model will accept
        - .predict() - fits cleaned data on the model.
        
    Then info can be gotten from the public properties:
        - .predictions
        - .model

    Most properties have a default setting so that this class should work out of the box.
    The code will automaticly look for a few files in its own directory (XGBoost_Membrane_Model.json, normalisation_values.csv)
    If you wish to change this filepath, the property 'current_path' can be edited.
    
    If at any point the model is retrained or changed the normalisation values and model.json file can easaly be changed.
    """
    
    def __init__(self, raw_data):
        """
        Iniate the class with raw data
        """
        import os
        self.current_path=os.getcwd()
        self.raw_data=raw_data
        print(os.getcwd())
        
    def clean_data(self):
        """Clean Data placeholder"""
        pass

    def predict(self):
        """
        Load the model from the filepath specefied
        """
        self.predictions=[0,0,0,1,0]
    
    def predictions_to_SQL(self,SQL_conn_string, database_header_names='BatchName,[DateTime],Prediction'):
        import pyodbc
        
        self.SQL_conn_string = SQL_conn_string
        self.database_header_names = database_header_names
        predictions = self.predictions        

        conn_str = (SQL_conn_string)
        
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        for index, row in predictions.iterrows():
            batchname=predictions.loc[index,'BatchName']
            date=predictions.loc[index,'DateTime']
            predict=predictions.loc[index,'Failure within 20 Batches']
            
            querry="""
                INSERT INTO Predictions (%s)
                VALUES ('%s','%s','%s');
                """%(database_header_names,batchname,date,predict)
            
            cursor.execute(querry)
            cursor.commit()
        
        
        conn.close()
        