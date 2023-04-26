# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:17:27 2023

@author: klabbf
"""

class Membrane_Model:
    def __init__(self, raw_data):
        import os
        self.current_path=os.getcwd()
        self.raw_data=raw_data

        
    def clean_data(self):
        import pandas as pd
        import tsfresh

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
        
        normalisation_values=pd.read_csv(self.current_path+'\\normalisation_values.csv')
        
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
        
        One_hot=pd.get_dummies(series)
        
        # Rethink implementation of tsfresh string later
        tsfresh_fc_parameters={'5IAL_3_WY301.54': {'absolute_maximum': None, 'maximum': None, 'mean_n_absolute_max': [{'number_of_maxima': 7}], 'quantile': [{'q': 0.9}, {'q': 0.8}]}, '5IAL_3_TT301.50': {'fft_coefficient': [{'attr': 'angle', 'coeff': 0}, {'attr': 'real', 'coeff': 1}], 'index_mass_quantile': [{'q': 0.1}, {'q': 0.2}], 'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 0}, {'num_segments': 10, 'segment_focus': 4}, {'num_segments': 10, 'segment_focus': 3}], 'variation_coefficient': None, 'change_quantiles': [{'f_agg': 'mean', 'isabs': True, 'qh': 0.4, 'ql': 0.0}, {'f_agg': 'mean', 'isabs': True, 'qh': 0.2, 'ql': 0.0}], 'agg_linear_trend': [{'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'min'}, {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'min'}], 'linear_trend': [{'attr': 'rvalue'}, {'attr': 'slope'}], 'ar_coefficient': [{'coeff': 0, 'k': 10}], 'number_crossing_m': [{'m': 1}]}, '5IAL_3_FIT301.61MF': {'agg_linear_trend': [{'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'max'}]}, '5IAL_3_QIT301.52': {'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 7}, {'num_segments': 10, 'segment_focus': 6}, {'num_segments': 10, 'segment_focus': 8}], 'index_mass_quantile': [{'q': 0.2}], 'c3': [{'lag': 1}]}, '5IAL_3_FIT301.61D': {'quantile': [{'q': 0.9}, {'q': 0.8}, {'q': 0.7}, {'q': 0.4}, {'q': 0.6}], 'median': None, 'mean_n_absolute_max': [{'number_of_maxima': 7}], 'matrix_profile': [{'feature': 'max', 'threshold': 0.98}, {'feature': '75', 'threshold': 0.98}]}, '5IAL_3_PIT301.63': {'cwt_coefficients': [{'coeff': 6, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 1, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 1, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 20, 'widths': (2, 5, 10, 20)}], 'quantile': [{'q': 0.8}, {'q': 0.9}, {'q': 0.3}, {'q': 0.2}, {'q': 0.1}, {'q': 0.4}, {'q': 0.6}, {'q': 0.7}], 'agg_linear_trend': [{'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'min'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'min'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'min'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'mean'}], 'linear_trend': [{'attr': 'intercept'}], 'maximum': None, 'mean_n_absolute_max': [{'number_of_maxima': 7}], 'absolute_maximum': None, 'c3': [{'lag': 1}, {'lag': 2}, {'lag': 3}], 'minimum': None, 'root_mean_square': None, 'mean': None, 'median': None, 'abs_energy': None}, '5IAL_3_XPV301.13': {'minimum': None}, '5IAL_3_R301.71': {'number_cwt_peaks': [{'n': 5}]}, '5IAL_3_P301.72': {'agg_linear_trend': [{'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'min'}, {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'min'}]}, 'frac': {'cwt_coefficients': [{'coeff': 7, 'w': 5, 'widths': (2, 5, 10, 20)}]}}
        
        tsfresh_features=tsfresh.extract_features(data_norm, column_id="5IAL_3_301.BatchName", default_fc_parameters=tsfresh_fc_parameters)
        
        """
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Create X and y
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
   
        X=pd.concat([X_t,tsfresh_features],axis=1)
        X=pd.concat([X,One_hot],axis=1)
        
        self.__X=X
        
#%% Unit Test!
import os
import pandas as pd
import numpy as np


test_data=pd.read_csv(os.getcwd()+'\Test_Data.csv',parse_dates=[1],index_col=[0])
test_data.rename(columns={'0':'Date'},inplace=True)
test_data.head()

Membrane_Model=Membrane_Model(test_data)
Membrane_Model.clean_data()
print(Membrane_Model.X)