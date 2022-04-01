#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 22:12:51 2022

@author: zengke
"""


from sklearn.exceptions import NotFittedError
from BDMLtools.exception import DataTypeError,XyIndexError,yValueError
from pandas.api.types import is_array_like
from warnings import warn
import pandas as pd
import numpy as np

class Base:
    
    def _check_is_fitted(self):
        
        if not self._is_fitted:
            
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))
            
            
    def _check_X(self,X):
 
        if not isinstance(X,pd.core.frame.DataFrame):
                
            raise DataTypeError("X is pd.core.frame.DataFrame")
            
        if not X.index.is_unique:
                
            raise XyIndexError("X.index is not unique")
            
            
    def _check_data(self,X,y):
        
        if not isinstance(X,pd.core.frame.DataFrame):
            
            raise DataTypeError("X is pd.core.frame.DataFrame")
            
        if not isinstance(y,pd.core.frame.Series):
            
            raise DataTypeError("y is pd.core.frame.Series")
            
        if not X.index.is_unique:
            
            raise XyIndexError("X.index is not unique")
            
        if not y.index.is_unique:
            
            raise XyIndexError("y.index is not unique")
            
        if not X.index.equals(y.index):
        
            raise XyIndexError("X's index not equal to y")            

        if not np.isin(y.unique(),[0,1]).any():
        
            raise yValueError("vals of y in [0,1] and 0(no-event),1(event)")    


    def _check_ws(self,y,sample_weight):

        if sample_weight is not None:
            
            if not isinstance(sample_weight,pd.Series):
                
                raise DataTypeError("sample_weight is not pandas.Series.") 
                
            if not y.index.equals(sample_weight.index):
                
                raise XyIndexError("index of sample_weight not equal to y_pred or y_true")             

    
    def _check_param_dtype(self,dtype):
        
        if not dtype in ('float32','float64'):
            
            raise ValueError("dtype in ('float32','float64')")



class BaseEval:    
    
    
    def _check_plot_params(self,show_plot,pred_desc):
        
        if not isinstance(show_plot ,tuple):
            
            raise ValueError("show_plot is tuple")  
            
        if not np.isin(show_plot,('ks', 'lift', 'gain', 'roc', 'lz', 'pr', 'f1', 'density')).any() :
            
            raise ValueError("show_plot in ('ks', 'lift', 'gain', 'roc', 'lz', 'pr', 'f1', 'density')")  
            
        if pred_desc is not None:
            
            if not isinstance(pred_desc,bool):
                
                raise ValueError("pred_desc is bool type.")  
    
    
    def _check_params(self,y_pred,y_true,group,sample_weight):
        
        if not isinstance(y_pred,pd.Series):
            
            raise DataTypeError("y_pred is not pandas.Series.")   
            
        if not isinstance(y_true,pd.Series):
            
            raise DataTypeError("y_true is not pandas.Series.")  
            
        if not y_pred.index.equals(y_true.index):
            
            raise XyIndexError("index of y_true not equal to y_pred")  
            
        if sample_weight is not None:
            
            if not isinstance(sample_weight,pd.Series):
                
                raise DataTypeError("sample_weight is not pandas.Series.") 
                
            if not y_pred.index.equals(sample_weight.index):
                
                raise XyIndexError("index of sample_weight not equal to y_pred or y_true") 
            
        if group is not None:    
            
            if not is_array_like(group):
                
                raise DataTypeError("group is not array.")              


    def _check_values(self,y_pred,y_true,group,sample_weight):   
        
        if pd.isnull(y_pred).any():
            
            warn("y_pred contains NAN and will be droped.")   
            
        if pd.isnull(y_true).any():
            
            raise yValueError("y contains NAN")  
            
        if not y_pred.size==y_true.size:
            
            raise ValueError("length of y_true and y_pred not same.")  
        
        if group is not None: 
            
            if pd.isnull(group).any():
            
                warn("group contains NAN and will be droped.")  

        if not np.isin(y_true.unique(),[0,1]).any():
        
            raise yValueError("vals of y in [0,1] and 0(no-event),1(event).")   
            
        if sample_weight is not None:          
                            
            if sample_weight.size!=y_pred.size or sample_weight.size!=y_true.size:
                
                raise ValueError("length of sample_weight not equal to y_true and y_pred") 
            
            if (sample_weight<0).any():
                
                raise ValueError("vals of sample_weight should be non-negative")  
                
            if pd.isnull(sample_weight).any():
                
                raise ValueError("sample_weight contains NAN")        
                
            