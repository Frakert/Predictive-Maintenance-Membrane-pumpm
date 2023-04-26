# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:17:27 2023

@author: klabbf
"""

class Membrane_Model:
    def __init__(self, raw_data):
        import os
        
        self.raw_data=raw_data
        self.model_path=os.getcwd()
        self.output_path=os.getcwd()
        
    def clean_data(self):
        import pandas as pd
        import numpy as np
        import os
        
        
        raw_data=self.raw_data
        raw_data['5IAL_3_301.BatchName']= raw_data['5IAL_3_301.BatchName'].fillna('No Batch Specified')
        
        
        # Filter out everything that is not a batch
        COLUMN_NAME = '5IAL_3_301.BatchName'
        mask = (raw_data[COLUMN_NAME].str.len() >= 11) & (raw_data[COLUMN_NAME].str.len() <= 12)
        data_selected = raw_data[mask]
        data=data_selected
        
        
        """
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Normalise All data
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        """
        
        batch_names=data['5IAL_3_301.BatchName']
        dates=raw_data['Date']
        
        data.drop(['5IAL_3_301.BatchName','Date'],axis=1,inplace=True)
        
        normalisation_values=pd.read_csv(os.getcwd()+'normalisation_values.csv')
        
        data_norm=pd.DataFrame()
        for col in data:
            zval=(data[col]-np.mean(data[col]))/np.std(data[col])
            data_norm=pd.concat([data_norm,zval],axis=1)
        
            
        data_norm['5IAL_3_301.BatchName']=batch_names
        data['5IAL_3_301.BatchName']=batch_names
         
        
        """ 
       -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
       Feature Creation
       -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
       """
       