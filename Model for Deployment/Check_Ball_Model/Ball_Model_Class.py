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

class Ball_Model(ML_Model):
        
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
        import warnings
    
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

        tsfresh_fc_parameters={'5IAL_3_QIT301.57': {'ratio_beyond_r_sigma': [{'r': 6}, {'r': 5}, {'r': 3}, {'r': 2.5}, {'r': 0.5}, {'r': 1}], 'large_standard_deviation': [{'r': 0.2}, {'r': 0.25}, {'r': 0.30000000000000004}, {'r': 0.15000000000000002}, {'r': 0.35000000000000003}], 'change_quantiles': [{'f_agg': 'var', 'isabs': False, 'qh': 1.0, 'ql': 0.8}, {'f_agg': 'mean', 'isabs': False, 'qh': 1.0, 'ql': 0.8}, {'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.8}, {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.8}, {'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.2}, {'f_agg': 'var', 'isabs': False, 'qh': 1.0, 'ql': 0.2}, {'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.4}, {'f_agg': 'mean', 'isabs': False, 'qh': 1.0, 'ql': 0.6}, {'f_agg': 'var', 'isabs': False, 'qh': 1.0, 'ql': 0.4}, {'f_agg': 'mean', 'isabs': False, 'qh': 1.0, 'ql': 0.2}, {'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.6}, {'f_agg': 'mean', 'isabs': False, 'qh': 1.0, 'ql': 0.4}, {'f_agg': 'var', 'isabs': False, 'qh': 1.0, 'ql': 0.6}, {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.2}, {'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.0}, {'f_agg': 'var', 'isabs': False, 'qh': 1.0, 'ql': 0.0}, {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.0}, {'f_agg': 'var', 'isabs': True, 'qh': 0.6, 'ql': 0.2}, {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.4}, {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.6}, {'f_agg': 'var', 'isabs': False, 'qh': 0.6, 'ql': 0.2}, {'f_agg': 'var', 'isabs': True, 'qh': 0.6, 'ql': 0.4}, {'f_agg': 'var', 'isabs': False, 'qh': 0.6, 'ql': 0.4}], 'last_location_of_maximum': None, 'first_location_of_maximum': None, 'skewness': None, 'cwt_coefficients': [{'coeff': 1, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 0, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 1, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 0, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 0, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 2, 'widths': (2, 5, 10, 20)}], 'index_mass_quantile': [{'q': 0.2}, {'q': 0.3}, {'q': 0.1}, {'q': 0.4}, {'q': 0.6}], 'binned_entropy': [{'max_bins': 10}], 'variation_coefficient': None, 'kurtosis': None, 'autocorrelation': [{'lag': 4}, {'lag': 9}, {'lag': 3}, {'lag': 8}, {'lag': 5}, {'lag': 7}, {'lag': 6}, {'lag': 2}, {'lag': 1}], 'agg_linear_trend': [{'attr': 'slope', 'chunk_len': 50, 'f_agg': 'var'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'var'}, {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'var'}, {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'mean'}, {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'var'}, {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'var'}, {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'var'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'var'}, {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'var'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'var'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'max'}], 'fft_aggregated': [{'aggtype': 'skew'}, {'aggtype': 'centroid'}], 'fft_coefficient': [{'attr': 'imag', 'coeff': 12}, {'attr': 'real', 'coeff': 18}, {'attr': 'real', 'coeff': 2}, {'attr': 'real', 'coeff': 19}, {'attr': 'real', 'coeff': 4}, {'attr': 'imag', 'coeff': 11}, {'attr': 'imag', 'coeff': 7}, {'attr': 'imag', 'coeff': 6}, {'attr': 'abs', 'coeff': 16}, {'attr': 'real', 'coeff': 23}, {'attr': 'abs', 'coeff': 8}, {'attr': 'real', 'coeff': 1}, {'attr': 'real', 'coeff': 14}, {'attr': 'abs', 'coeff': 26}, {'attr': 'imag', 'coeff': 13}, {'attr': 'real', 'coeff': 3}, {'attr': 'abs', 'coeff': 17}, {'attr': 'abs', 'coeff': 15}, {'attr': 'abs', 'coeff': 19}, {'attr': 'abs', 'coeff': 12}, {'attr': 'abs', 'coeff': 10}, {'attr': 'abs', 'coeff': 5}, {'attr': 'abs', 'coeff': 29}, {'attr': 'abs', 'coeff': 18}, {'attr': 'abs', 'coeff': 13}, {'attr': 'imag', 'coeff': 9}, {'attr': 'abs', 'coeff': 28}, {'attr': 'abs', 'coeff': 9}, {'attr': 'abs', 'coeff': 25}, {'attr': 'abs', 'coeff': 23}, {'attr': 'abs', 'coeff': 37}, {'attr': 'abs', 'coeff': 22}, {'attr': 'abs', 'coeff': 27}, {'attr': 'real', 'coeff': 22}, {'attr': 'abs', 'coeff': 14}, {'attr': 'abs', 'coeff': 20}, {'attr': 'real', 'coeff': 15}, {'attr': 'abs', 'coeff': 50}, {'attr': 'abs', 'coeff': 30}, {'attr': 'abs', 'coeff': 36}, {'attr': 'imag', 'coeff': 8}, {'attr': 'abs', 'coeff': 7}, {'attr': 'abs', 'coeff': 31}, {'attr': 'real', 'coeff': 12}, {'attr': 'abs', 'coeff': 35}, {'attr': 'real', 'coeff': 13}, {'attr': 'abs', 'coeff': 33}, {'attr': 'abs', 'coeff': 24}, {'attr': 'abs', 'coeff': 40}, {'attr': 'abs', 'coeff': 39}, {'attr': 'abs', 'coeff': 34}, {'attr': 'abs', 'coeff': 32}, {'attr': 'real', 'coeff': 16}, {'attr': 'abs', 'coeff': 38}, {'attr': 'abs', 'coeff': 6}, {'attr': 'abs', 'coeff': 21}, {'attr': 'abs', 'coeff': 11}, {'attr': 'imag', 'coeff': 5}], 'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 7}, {'num_segments': 10, 'segment_focus': 0}, {'num_segments': 10, 'segment_focus': 6}, {'num_segments': 10, 'segment_focus': 8}, {'num_segments': 10, 'segment_focus': 5}], 'absolute_maximum': None, 'maximum': None, 'symmetry_looking': [{'r': 0.05}], 'lempel_ziv_complexity': [{'bins': 10}, {'bins': 5}], 'partial_autocorrelation': [{'lag': 2}, {'lag': 1}], 'max_langevin_fixed_point': [{'m': 3, 'r': 30}], 'cid_ce': [{'normalize': True}, {'normalize': False}], 'friedrich_coefficients': [{'coeff': 3, 'm': 3, 'r': 30}, {'coeff': 2, 'm': 3, 'r': 30}], 'absolute_sum_of_changes': None, 'agg_autocorrelation': [{'f_agg': 'var', 'maxlag': 40}, {'f_agg': 'median', 'maxlag': 40}, {'f_agg': 'mean', 'maxlag': 40}], 'mean_n_absolute_max': [{'number_of_maxima': 7}], 'mean_abs_change': None, 'standard_deviation': None, 'variance': None, 'ar_coefficient': [{'coeff': 10, 'k': 10}, {'coeff': 0, 'k': 10}], 'augmented_dickey_fuller': [{'attr': 'pvalue', 'autolag': 'AIC'}, {'attr': 'teststat', 'autolag': 'AIC'}], 'c3': [{'lag': 1}, {'lag': 2}, {'lag': 3}], 'time_reversal_asymmetry_statistic': [{'lag': 1}, {'lag': 2}, {'lag': 3}], 'linear_trend': [{'attr': 'stderr'}, {'attr': 'intercept'}], 'root_mean_square': None, 'count_below_mean': None, 'abs_energy': None, 'approximate_entropy': [{'m': 2, 'r': 0.5}], 'longest_strike_below_mean': None, 'mean': None}, '5IAL_3_FIT301.61MF': {'mean_n_absolute_max': [{'number_of_maxima': 7}], 'absolute_maximum': None, 'maximum': None, 'quantile': [{'q': 0.9}, {'q': 0.8}, {'q': 0.7}, {'q': 0.6}], 'cwt_coefficients': [{'coeff': 14, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 5, 'widths': (2, 5, 10, 20)}], 'median': None}, '5IAL_3_PIT301.63': {'agg_linear_trend': [{'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'min'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'min'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'min'}], 'cwt_coefficients': [{'coeff': 1, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 1, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 0, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 0, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 1, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 0, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 2, 'widths': (2, 5, 10, 20)}], 'root_mean_square': None, 'mean': None, 'quantile': [{'q': 0.6}, {'q': 0.9}, {'q': 0.8}, {'q': 0.4}, {'q': 0.3}, {'q': 0.2}, {'q': 0.7}, {'q': 0.1}], 'c3': [{'lag': 1}, {'lag': 2}, {'lag': 3}], 'maximum': None, 'absolute_maximum': None, 'mean_n_absolute_max': [{'number_of_maxima': 7}], 'median': None, 'linear_trend': [{'attr': 'intercept'}], 'minimum': None, 'sum_of_reoccurring_values': None, 'fft_coefficient': [{'attr': 'real', 'coeff': 50}, {'attr': 'real', 'coeff': 28}], 'change_quantiles': [{'f_agg': 'mean', 'isabs': False, 'qh': 1.0, 'ql': 0.4}]}, '5IAL_3_PIT 301.55': {'maximum': None, 'absolute_maximum': None, 'quantile': [{'q': 0.9}, {'q': 0.8}, {'q': 0.7}, {'q': 0.6}], 'mean_n_absolute_max': [{'number_of_maxima': 7}], 'cid_ce': [{'normalize': False}], 'agg_linear_trend': [{'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'var'}, {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'var'}], 'cwt_coefficients': [{'coeff': 6, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 1, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 2, 'widths': (2, 5, 10, 20)}], 'absolute_sum_of_changes': None, 'number_crossing_m': [{'m': 1}], 'change_quantiles': [{'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.0}, {'f_agg': 'var', 'isabs': False, 'qh': 1.0, 'ql': 0.0}], 'time_reversal_asymmetry_statistic': [{'lag': 2}, {'lag': 3}]}, '5IAL_3_QIT301.52': {'absolute_maximum': None, 'maximum': None, 'mean_n_absolute_max': [{'number_of_maxima': 7}], 'cwt_coefficients': [{'coeff': 3, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 0, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 1, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 5, 'widths': (2, 5, 10, 20)}], 'approximate_entropy': [{'m': 2, 'r': 0.1}], 'skewness': None, 'index_mass_quantile': [{'q': 0.3}, {'q': 0.1}, {'q': 0.2}, {'q': 0.4}], 'agg_linear_trend': [{'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'var'}, {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'var'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'var'}, {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'var'}, {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'var'}, {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'var'}], 'change_quantiles': [{'f_agg': 'mean', 'isabs': False, 'qh': 1.0, 'ql': 0.2}, {'f_agg': 'mean', 'isabs': False, 'qh': 1.0, 'ql': 0.4}, {'f_agg': 'mean', 'isabs': False, 'qh': 1.0, 'ql': 0.8}, {'f_agg': 'mean', 'isabs': False, 'qh': 1.0, 'ql': 0.6}, {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.8}], 'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 0}, {'num_segments': 10, 'segment_focus': 7}, {'num_segments': 10, 'segment_focus': 8}, {'num_segments': 10, 'segment_focus': 6}], 'sample_entropy': None, 'time_reversal_asymmetry_statistic': [{'lag': 3}, {'lag': 2}], 'percentage_of_reoccurring_values_to_all_values': None, 'matrix_profile': [{'feature': '75', 'threshold': 0.98}, {'feature': 'median', 'threshold': 0.98}, {'feature': 'mean', 'threshold': 0.98}, {'feature': 'max', 'threshold': 0.98}], 'permutation_entropy': [{'dimension': 7, 'tau': 1}, {'dimension': 6, 'tau': 1}, {'dimension': 5, 'tau': 1}, {'dimension': 4, 'tau': 1}], 'linear_trend': [{'attr': 'intercept'}, {'attr': 'rvalue'}, {'attr': 'slope'}], 'fft_coefficient': [{'attr': 'real', 'coeff': 1}, {'attr': 'imag', 'coeff': 1}, {'attr': 'imag', 'coeff': 6}], 'quantile': [{'q': 0.9}], 'longest_strike_below_mean': None, 'count_below_mean': None, 'variance_larger_than_standard_deviation': None}, '5IAL_3_TT301.50': {'fft_coefficient': [{'attr': 'abs', 'coeff': 39}, {'attr': 'abs', 'coeff': 44}, {'attr': 'abs', 'coeff': 40}, {'attr': 'abs', 'coeff': 45}, {'attr': 'abs', 'coeff': 34}, {'attr': 'abs', 'coeff': 43}, {'attr': 'abs', 'coeff': 49}, {'attr': 'abs', 'coeff': 35}, {'attr': 'abs', 'coeff': 36}, {'attr': 'abs', 'coeff': 38}, {'attr': 'abs', 'coeff': 41}, {'attr': 'abs', 'coeff': 47}, {'attr': 'abs', 'coeff': 37}, {'attr': 'abs', 'coeff': 51}, {'attr': 'abs', 'coeff': 50}, {'attr': 'abs', 'coeff': 33}, {'attr': 'abs', 'coeff': 42}, {'attr': 'abs', 'coeff': 48}, {'attr': 'abs', 'coeff': 46}, {'attr': 'abs', 'coeff': 29}, {'attr': 'real', 'coeff': 49}, {'attr': 'abs', 'coeff': 28}, {'attr': 'abs', 'coeff': 32}, {'attr': 'imag', 'coeff': 12}, {'attr': 'abs', 'coeff': 30}], 'cwt_coefficients': [{'coeff': 3, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 1, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 5, 'widths': (2, 5, 10, 20)}], 'quantile': [{'q': 0.4}, {'q': 0.3}, {'q': 0.2}, {'q': 0.6}, {'q': 0.7}, {'q': 0.1}, {'q': 0.8}], 'median': None, 'partial_autocorrelation': [{'lag': 2}], 'mean': None, 'root_mean_square': None, 'c3': [{'lag': 1}, {'lag': 3}, {'lag': 2}], 'fft_aggregated': [{'aggtype': 'kurtosis'}, {'aggtype': 'skew'}], 'change_quantiles': [{'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.0}, {'f_agg': 'var', 'isabs': False, 'qh': 1.0, 'ql': 0.0}], 'agg_linear_trend': [{'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'mean'}], 'ar_coefficient': [{'coeff': 10, 'k': 10}]}, '5IAL_3_PIT301.60': {'number_crossing_m': [{'m': 0}], 'absolute_maximum': None, 'absolute_sum_of_changes': None, 'cid_ce': [{'normalize': False}, {'normalize': True}], 'maximum': None, 'symmetry_looking': [{'r': 0.55}, {'r': 0.65}, {'r': 0.6000000000000001}, {'r': 0.25}, {'r': 0.5}, {'r': 0.45}, {'r': 0.4}, {'r': 0.8500000000000001}, {'r': 0.9500000000000001}, {'r': 0.35000000000000003}, {'r': 0.9}, {'r': 0.8}, {'r': 0.30000000000000004}, {'r': 0.7000000000000001}, {'r': 0.75}, {'r': 0.2}, {'r': 0.15000000000000002}, {'r': 0.1}, {'r': 0.05}], 'large_standard_deviation': [{'r': 0.05}], 'approximate_entropy': [{'m': 2, 'r': 0.9}, {'m': 2, 'r': 0.5}, {'m': 2, 'r': 0.3}, {'m': 2, 'r': 0.1}, {'m': 2, 'r': 0.7}], 'agg_linear_trend': [{'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'var'}, {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'var'}, {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'mean'}, {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'mean'}, {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'var'}, {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'var'}, {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'var'}, {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'mean'}, {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'var'}, {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'var'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'var'}, {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'var'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'var'}, {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'var'}], 'spkt_welch_density': [{'coeff': 8}, {'coeff': 5}, {'coeff': 2}], 'first_location_of_maximum': None, 'fft_coefficient': [{'attr': 'angle', 'coeff': 2}, {'attr': 'angle', 'coeff': 1}, {'attr': 'angle', 'coeff': 4}, {'attr': 'angle', 'coeff': 3}, {'attr': 'abs', 'coeff': 51}, {'attr': 'imag', 'coeff': 2}, {'attr': 'angle', 'coeff': 5}, {'attr': 'imag', 'coeff': 1}, {'attr': 'imag', 'coeff': 4}, {'attr': 'abs', 'coeff': 50}, {'attr': 'imag', 'coeff': 3}, {'attr': 'angle', 'coeff': 6}, {'attr': 'abs', 'coeff': 3}, {'attr': 'abs', 'coeff': 4}, {'attr': 'abs', 'coeff': 5}, {'attr': 'abs', 'coeff': 7}, {'attr': 'abs', 'coeff': 0}, {'attr': 'abs', 'coeff': 6}, {'attr': 'abs', 'coeff': 8}, {'attr': 'abs', 'coeff': 2}, {'attr': 'angle', 'coeff': 7}, {'attr': 'abs', 'coeff': 9}, {'attr': 'abs', 'coeff': 10}, {'attr': 'abs', 'coeff': 1}, {'attr': 'imag', 'coeff': 5}, {'attr': 'abs', 'coeff': 11}, {'attr': 'abs', 'coeff': 43}, {'attr': 'abs', 'coeff': 12}, {'attr': 'angle', 'coeff': 8}, {'attr': 'abs', 'coeff': 27}, {'attr': 'abs', 'coeff': 49}, {'attr': 'abs', 'coeff': 28}, {'attr': 'abs', 'coeff': 36}, {'attr': 'abs', 'coeff': 42}, {'attr': 'angle', 'coeff': 9}, {'attr': 'abs', 'coeff': 26}, {'attr': 'angle', 'coeff': 10}, {'attr': 'imag', 'coeff': 6}, {'attr': 'abs', 'coeff': 25}, {'attr': 'abs', 'coeff': 13}, {'attr': 'angle', 'coeff': 11}, {'attr': 'abs', 'coeff': 24}, {'attr': 'abs', 'coeff': 40}, {'attr': 'abs', 'coeff': 23}, {'attr': 'abs', 'coeff': 29}, {'attr': 'abs', 'coeff': 14}, {'attr': 'abs', 'coeff': 41}, {'attr': 'abs', 'coeff': 37}, {'attr': 'abs', 'coeff': 30}, {'attr': 'abs', 'coeff': 15}, {'attr': 'angle', 'coeff': 12}, {'attr': 'abs', 'coeff': 16}, {'attr': 'abs', 'coeff': 38}, {'attr': 'abs', 'coeff': 48}, {'attr': 'imag', 'coeff': 7}, {'attr': 'abs', 'coeff': 20}, {'attr': 'abs', 'coeff': 31}, {'attr': 'abs', 'coeff': 39}, {'attr': 'abs', 'coeff': 35}, {'attr': 'abs', 'coeff': 32}, {'attr': 'abs', 'coeff': 21}, {'attr': 'abs', 'coeff': 19}, {'attr': 'abs', 'coeff': 22}, {'attr': 'abs', 'coeff': 33}, {'attr': 'abs', 'coeff': 17}, {'attr': 'imag', 'coeff': 8}, {'attr': 'abs', 'coeff': 18}, {'attr': 'abs', 'coeff': 44}, {'attr': 'angle', 'coeff': 13}, {'attr': 'imag', 'coeff': 9}, {'attr': 'abs', 'coeff': 34}, {'attr': 'abs', 'coeff': 47}, {'attr': 'imag', 'coeff': 10}, {'attr': 'real', 'coeff': 0}], 'permutation_entropy': [{'dimension': 7, 'tau': 1}, {'dimension': 6, 'tau': 1}, {'dimension': 5, 'tau': 1}, {'dimension': 3, 'tau': 1}, {'dimension': 4, 'tau': 1}], 'change_quantiles': [{'f_agg': 'var', 'isabs': False, 'qh': 1.0, 'ql': 0.0}, {'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.0}, {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.0}, {'f_agg': 'var', 'isabs': False, 'qh': 1.0, 'ql': 0.2}, {'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.2}, {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.2}, {'f_agg': 'var', 'isabs': False, 'qh': 1.0, 'ql': 0.8}, {'f_agg': 'var', 'isabs': False, 'qh': 1.0, 'ql': 0.6}, {'f_agg': 'var', 'isabs': False, 'qh': 1.0, 'ql': 0.4}, {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.8}, {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.4}, {'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.6}, {'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.6}, {'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.4}, {'f_agg': 'var', 'isabs': True, 'qh': 1.0, 'ql': 0.8}], 'mean_abs_change': None, 'mean_n_absolute_max': [{'number_of_maxima': 7}], 'abs_energy': None, 'cwt_coefficients': [{'coeff': 10, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 5, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 4, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 1, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 0, 'w': 20, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 2, 'widths': (2, 5, 10, 20)}], 'count_above_mean': None, 'longest_strike_above_mean': None, 'root_mean_square': None, 'binned_entropy': [{'max_bins': 10}], 'standard_deviation': None, 'variance': None, 'ratio_beyond_r_sigma': [{'r': 0.5}, {'r': 1}, {'r': 1.5}, {'r': 2}, {'r': 2.5}, {'r': 3}], 'agg_autocorrelation': [{'f_agg': 'var', 'maxlag': 40}, {'f_agg': 'median', 'maxlag': 40}], 'linear_trend': [{'attr': 'stderr'}, {'attr': 'pvalue'}, {'attr': 'intercept'}, {'attr': 'slope'}, {'attr': 'rvalue'}], 'count_below': [{'t': 0}], 'count_below_mean': None, 'longest_strike_below_mean': None, 'kurtosis': None, 'sum_values': None}, '5IAL_3_WY301.54': {'change_quantiles': [{'f_agg': 'var', 'isabs': False, 'qh': 0.4, 'ql': 0.0}, {'f_agg': 'var', 'isabs': True, 'qh': 0.4, 'ql': 0.0}], 'cwt_coefficients': [{'coeff': 1, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 2, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 0, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 2, 'widths': (2, 5, 10, 20)}], 'quantile': [{'q': 0.8}, {'q': 0.9}], 'mean_n_absolute_max': [{'number_of_maxima': 7}], 'absolute_maximum': None, 'maximum': None, 'cid_ce': [{'normalize': False}], 'last_location_of_minimum': None, 'benford_correlation': None, 'percentage_of_reoccurring_values_to_all_values': None, 'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 0}]}, '5IAL_3_LIT301.54': {'cid_ce': [{'normalize': False}], 'cwt_coefficients': [{'coeff': 14, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 1, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 11, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 6, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 10, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 0, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 7, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 0, 'w': 5, 'widths': (2, 5, 10, 20)}], 'agg_linear_trend': [{'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'var'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'var'}], 'partial_autocorrelation': [{'lag': 4}], 'approximate_entropy': [{'m': 2, 'r': 0.9}], 'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 0}], 'time_reversal_asymmetry_statistic': [{'lag': 2}, {'lag': 3}], 'change_quantiles': [{'f_agg': 'var', 'isabs': True, 'qh': 0.6, 'ql': 0.0}]}, '5IAL_3_FIT301.61D': {'quantile': [{'q': 0.4}, {'q': 0.6}, {'q': 0.3}], 'median': None, 'minimum': None}, '5IAL_3_FIT301.61VF': {'cwt_coefficients': [{'coeff': 14, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 14, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 9, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 5, 'widths': (2, 5, 10, 20)}, {'coeff': 13, 'w': 10, 'widths': (2, 5, 10, 20)}, {'coeff': 8, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 3, 'w': 2, 'widths': (2, 5, 10, 20)}, {'coeff': 12, 'w': 10, 'widths': (2, 5, 10, 20)}], 'benford_correlation': None, 'quantile': [{'q': 0.7}]}}

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
        import pandas as pd
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        
        
        # TO DO: make model filename customizable
        
       
        model = joblib.load(self.current_path+'\\Dependencies\\check_valve_model.joblib')
        self.model=model
        
        X=self.X
        f_names = model.feature_names


        y_pred=model.predict(X[f_names])
        
    
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
        