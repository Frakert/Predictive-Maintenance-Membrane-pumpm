# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:22:05 2023

@author: klabbf
"""
import sys
import os

# Add the ML_Model folder to the places python looks for packages and modules.
# Do note that this is relative to this folder.
from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
curr_path = dirname(__file__)
dependencies_path = curr_path + '\\Dependencies'
parrent_class_path = abspath(join(curr_path, '..','ML_Model'))
sys.path.append(parrent_class_path)

from ML_Model_Class import ML_Model


class Membrane_Model(ML_Model):
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
        Iniate the child class by calling the parent class
        """
        ML_Model.__init__(self, raw_data)
        
    def clean_data(self):
        """
        Clean data in a way that the model will accept.
        This is fully model and training method dependend and thus not very reusable.
        """
        
        # TO DO: check for incomplete batches (batches that start or end outside the reach of the dataset.) 
        # TO DO: See if certain aspects (like One hot encoding) can be made into reusable methods.
        # TO DO: Rename current_path to a more logical name
        # TO DO: Cleanup code where possible
        
        import tsfresh
        import pandas as pd
        import numpy as np

        raw_data=self.raw_data
        raw_data['5IAL_3_301.BatchName']= raw_data['5IAL_3_301.BatchName'].fillna('No Batch Specified')
        
        # Check for Columns that are not supposed to be in the dataset and if all columns are there
        NAME_LIST=[
        "Date","5IAL_3_TT301.50","5IAL_3_QIT301.52","5IAL_3_PIT 301.55","5IAL_3_PIT301.63","5IAL_3_QIT301.57","5IAL_3_PIT301.60","5IAL_3_FIT301.61MF","5IAL_3_FIT301.61VF","5IAL_3_FIT301.61D","5IAL_3_P301.70","5IAL_3_R301.71","5IAL_3_P301.72","5IAL_3_301.BatchName","5IAL_3_XPV301.05","5IAL_3_XPV301.06","5IAL_3_XPV301.08","5IAL_3_XPV301.09","5IAL_3_XPV301.22","5IAL_3_XPV301.35","5IAL_3_XPV301.36","5IAL_3_XPV301.42","5IAL_3_XPV301.43","5IAL_3_XPV301.46","5IAL_3_XPV301.53","5IAL_3_XPV301.54","5IAL_3_XPV301.63","5IAL_3_LSL301.51","5IAL_3_LSL301.53","5IAL_3_GSC301.44","5IAL_3_GSO301.44","5IAL_3_LSL301.64","5IAL_3_LSL301.68","5IAL_3_LSLL301.69","5IAL_3_301.OCCUPIED","5IAL_3_LIT301.54","5IAL_3_LSH301.56","5IAL_3_XPV301.13","5IAL_3_WY301.54"
        ]# All names that should be in columns

        list_extra=list(filter(lambda a: a not in NAME_LIST, test_data))
        list_missing =list(filter(lambda a: a not in test_data, NAME_LIST))

        if list_extra:
            warnings.warn("The input pandas dataframe has too many columns, the following columns shouldnt be there: " + ' '.join(str(a) for a in list_extra) +". "+"These columns will now be removed automaticly and will not be used for making predictions." )
            print(" ")
            test_data=raw_data.drop(list_extra,axis=1)

        if list_missing:
            raise Exception("The input pandas dataframe is missing the folowing columns: " + ' '.join(str(a) for a in list_missing))

        
        
        #Check first and last datapoint and check if the dataset cuts off a batch, if so filter it out.
        batchnames=raw_data['5IAL_3_301.BatchName']
        batch_list=[batchnames.iloc[0],batchnames.iloc[-1]]
        blacklist = [x for x in batch_list if x != 'No Batch Specified']

        # Loop over blacklist, filterout all the blacklist items.
        for i in range(len(blacklist)):
            raw_data.drop(raw_data[raw_data['5IAL_3_301.BatchName'] == blacklist[i]].index, inplace=True)
        
        
        
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
        #dates=raw_data['Date']
        
        data.drop(['5IAL_3_301.BatchName','Date'],axis=1,inplace=True)
        
        normalisation_values=pd.read_csv(self.current_path+'\\Dependencies\\normalisation_values.csv')
        
        data_norm=pd.DataFrame()
        for col in data:
            zval=(data[col]-normalisation_values[col][0])/normalisation_values[col][1]
            data_norm=pd.concat([data_norm,zval],axis=1)
        
            
        data_norm['5IAL_3_301.BatchName']=batch_names
        data['5IAL_3_301.BatchName']=batch_names
         
        
        """ 
       -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
       Feature Creation
       -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
       """
        data_norm['frac']=data_norm['5IAL_3_P301.70']/data_norm['5IAL_3_FIT301.61MF']

        mean_grouped=data_norm.groupby(pd.Grouper(key="5IAL_3_301.BatchName")).mean()
        mean_grouped=mean_grouped.add_suffix('_mean')
        
        std_grouped=data_norm.groupby(pd.Grouper(key="5IAL_3_301.BatchName")).std()
        std_grouped=std_grouped.add_suffix('_std')
    
        max_grouped=data_norm.groupby(pd.Grouper(key="5IAL_3_301.BatchName")).max()
        max_grouped=max_grouped.add_suffix('_max')
    
        min_grouped=data_norm.groupby(pd.Grouper(key="5IAL_3_301.BatchName")).min()
        min_grouped=min_grouped.add_suffix('_min')
    
        var_grouped=data_norm.groupby(pd.Grouper(key="5IAL_3_301.BatchName")).var()
        var_grouped=var_grouped.add_suffix('_var')
    
        sum_grouped=data_norm.groupby(pd.Grouper(key="5IAL_3_301.BatchName")).sum()
        sum_grouped=sum_grouped.add_suffix('_sum')
        
        Batch_duration=data.groupby(data['5IAL_3_301.BatchName']).size()
        Batch_duration=pd.DataFrame(Batch_duration, columns=['Batch_duration'])
        
        Batch_Names=data['5IAL_3_301.BatchName'][mask]
        unique_names=pd.DataFrame(Batch_Names.unique(),columns=['5IAL_3_301.BatchName'])
        
        batch_types_list=[]
        
        # for every unique name, check length and choose wether to take the first 3 or 4 letters
        for i in range(len(unique_names)):
            if len(unique_names.loc[i][0]) == 11:
                substring=unique_names.loc[i][0][0:3]
            else:
                substring=unique_names.loc[i][0][0:4]
            batch_types_list.append(substring.upper())
        
        series=pd.Series(batch_types_list)
        series= series.replace({'IU7':'IU70'})
        
        all_ink_types=pd.Series(['YP58','CD1','KD1', 'MD1', 'YD1', 'YP70', 'KP70', 'CP70', 'MP70', 'YB2', 'CB2',
                         'KC1', 'KB2', 'MB2', 'MP55', 'CP55', 'KP55', 'YP55', 'KP58', 'CP58', 'MP58', 'YC1',
                         'CC1', 'MC1', 'IU70', 'CGB2', 'CGD1'])
        inks_not_in_dataset=all_ink_types[~(all_ink_types.isin(series.unique()))]
        empty_one_hot_inktypes = pd.DataFrame(0, index=np.arange(len(series)), columns=inks_not_in_dataset)

        one_hot=pd.concat([pd.get_dummies(series),empty_one_hot_inktypes],axis=1)
        one_hot=pd.get_dummies(series)
        one_hot=pd.concat([one_hot,unique_names],axis=1)
        one_hot.set_index(['5IAL_3_301.BatchName'])
        
        # TO DO: Rethink implementation of tsfresh string later
        tsfresh_fc_parameters={'5IAL_3_WY301.54': {'absolute_maximum': None, 'maximum': None, 'mean_n_absolute_max': [{'number_of_maxima': 7}], 'quantile': [{'q': 0.9}, {'q': 0.8}]}, '5IAL_3_TT301.50': {'fft_coefficient': [{'attr': 'angle', 'coeff': 0}, {'attr': 'real', 'coeff': 1}], 'index_mass_quantile': [{'q': 0.1}, {'q': 0.2}], 'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 0}, {'num_segments': 10, 'segment_focus': 4}, {'num_segments': 10, 'segment_focus': 3}], 'variation_coefficient': None, 'change_quantiles': [{'f_agg': 'mean', 'isabs': True, 'qh': 0.4, 'ql': 0.0}, {'f_agg': 'mean', 'isabs': True, 'qh': 0.2, 'ql': 0.0}], 'agg_linear_trend': [{'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'min'}, {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'min'}], 'linear_trend': [{'attr': 'rvalue'}, {'attr': 'slope'}], 'ar_coefficient': [{'coeff': 0, 'k': 10}], 'number_crossing_m': [{'m': 1}]}, '5IAL_3_FIT301.61MF': {'agg_linear_trend': [{'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'max'}]}, '5IAL_3_QIT301.52': {'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 7}, {'num_segments': 10, 'segment_focus': 6}, {'num_segments': 10, 'segment_focus': 8}], 'index_mass_quantile': [{'q': 0.2}], 'c3': [{'lag': 1}]}, '5IAL_3_FIT301.61D': {'quantile': [{'q': 0.9}, {'q': 0.8}, {'q': 0.7}, {'q': 0.4}, {'q': 0.6}], 'median': None, 'mean_n_absolute_max': [{'number_of_maxima': 7}], 'matrix_profile': [{'feature': 'max', 'threshold': 0.98}, {'feature': '75', 'threshold': 0.98}]}, '5IAL_3_PIT301.63': {'cwt_coefficients': [{'coeff': 6, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 1, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 1, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 20, 'widths': (2, 5, 10, 20)}], 'quantile': [{'q': 0.8}, {'q': 0.9}, {'q': 0.3}, {'q': 0.2}, {'q': 0.1}, {'q': 0.4}, {'q': 0.6}, {'q': 0.7}], 'agg_linear_trend': [{'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'min'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'min'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'min'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'mean'}], 'linear_trend': [{'attr': 'intercept'}], 'maximum': None, 'mean_n_absolute_max': [{'number_of_maxima': 7}], 'absolute_maximum': None, 'c3': [{'lag': 1}, {'lag': 2}, {'lag': 3}], 'minimum': None, 'root_mean_square': None, 'mean': None, 'median': None, 'abs_energy': None}, '5IAL_3_XPV301.13': {'minimum': None}, '5IAL_3_R301.71': {'number_cwt_peaks': [{'n': 5}]}, '5IAL_3_P301.72': {'agg_linear_trend': [{'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'min'}, {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'min'}]}, 'frac': {'cwt_coefficients': [{'coeff': 7, 'w': 5, 'widths': (2, 5, 10, 20)}]}}
        
        data_norm_imp=data_norm.fillna(0)
        
        tsfresh_features=tsfresh.extract_features(data_norm_imp, column_id="5IAL_3_301.BatchName", kind_to_fc_parameters=tsfresh_fc_parameters)
        
        """
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Create X
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        """ 
        Eind=pd.merge(pd.merge(mean_grouped,std_grouped,on=['5IAL_3_301.BatchName']), unique_names, on=['5IAL_3_301.BatchName'])
    
        # Add in Batch duration
        Eind=pd.merge(Eind,Batch_duration,on=['5IAL_3_301.BatchName'])
        Eind=pd.merge(Eind,max_grouped,on=['5IAL_3_301.BatchName'])
        Eind=pd.merge(Eind,min_grouped,on=['5IAL_3_301.BatchName'])
        Eind=pd.merge(Eind,var_grouped,on=['5IAL_3_301.BatchName'])
        Eind=pd.merge(Eind,sum_grouped,on=['5IAL_3_301.BatchName'])
    
        Eind=pd.merge(Eind,Batch_duration,on=['5IAL_3_301.BatchName'])
       
        #X_t = Eind.drop(['5IAL_3_301.BatchName'],axis=1)
        X_t=Eind
        X_t=X_t.fillna(0)
        
        X=pd.merge(X_t,one_hot,on=['5IAL_3_301.BatchName'])
        X=X.set_index(['5IAL_3_301.BatchName'])
        
        X=X.join(tsfresh_features)
        X=X.fillna(0)
        
        self.X=X

    def predict(self):
        """
        Load the model from the filepath specefied
        """
        import xgboost as xgb
        import pandas as pd
        
        # TO DO: make model filename customizable
        
        model=xgb.XGBClassifier(colsample_bytree=0.8, learning_rate=0.075, max_depth= 3, n_estimators= 500)
        model.load_model(self.current_path+'\\Dependencies\\XGBoost_Membrane_Model.json')
        self.model=model
        
        X=self.X
        
        #XGBoost finds input order of features very important, use this to align X with the order the model expects.
        cols_when_model_builds = model.get_booster().feature_names
        y_pred=model.predict(X[cols_when_model_builds])
        
    
        batch_names=X.index.to_series(name='BatchName')
        
        raw_data=self.raw_data
        date_list=[]
        for name in batch_names:
            pd_data=(raw_data['5IAL_3_301.BatchName']==name)
            batch_date=(raw_data['Date'][pd_data[pd_data].index[0]])
            date_list.append(batch_date)
        
        batch_dates=pd.Series(date_list,name='DateTime')
        
        predictions=pd.concat([batch_dates,batch_names.reset_index()['BatchName']],axis=1)
        predictions=pd.concat([predictions,pd.Series(y_pred,name='Failure within 20 Batches')],axis=1)
        
        self.predictions=predictions
    
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
        