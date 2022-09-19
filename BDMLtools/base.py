#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 22:12:51 2022

@author: zengke
"""


from sklearn.exceptions import NotFittedError
from BDMLtools.exception import DataTypeError,XyIndexError,yValueError
import pandas as pd
import numpy as np
from itertools import combinations

class Base:
    
    def _check_is_fitted(self):
        
        if not self._is_fitted:
            
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))
            
    def _check_x(self,x):
        
        if not isinstance(x,pd.core.frame.Series):
            
            raise DataTypeError("x is pd.core.frame.Series")
            
        if x.dtype in [np.dtype(np.float32),np.dtype(np.float16)]:
            
            raise DataTypeError("x's float dtype must be float64")
            

    def _check_ind(self,lst):
        
        if not np.all([np.all(np.equal(com[0],com[1])) for com in combinations([x.index for x in lst],2)]):
            
            raise XyIndexError("All data's index must be equal")
            
        if not np.all([(x.index.is_unique) for x in lst]):
            
            raise XyIndexError("All data's index must be unique")      
            
            
    def _check_X(self,X,check_dtype=True):
 
        if not isinstance(X,pd.core.frame.DataFrame):
                
            raise DataTypeError("X is pd.core.frame.DataFrame")
            
        if not X.index.is_unique:
                
            raise XyIndexError("X.index is not unique")
            
        if check_dtype:    
            
            if X.dtypes.isin([np.dtype(np.float32),np.dtype(np.float16)]).any():
                
                raise DataTypeError("X's float dtype must be float64")
            
            
    def _check_colname(self,X):
        
        if not X.columns.is_unique:
            
            raise XyIndexError("duplicated colname in X")
            
            
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

        if not np.isin(y.unique(),[0,1]).all():
        
            raise yValueError("vals of y in [0,1] and 0(no-event),1(event)")  
            
        if X.dtypes.isin([np.dtype(np.float32),np.dtype(np.float16)]).any():
            
            raise DataTypeError("X's float dtype must be float64")


    def _check_ws(self,y,sample_weight):

        if sample_weight is not None:
            
            if not isinstance(sample_weight,pd.Series):
                
                raise DataTypeError("sample_weight is not pandas.Series.") 
                
            if sample_weight.dtypes in [np.dtype(np.float32),np.dtype(np.float16)]:
                
                raise DataTypeError("sample_weight's float dtype must be float64")
                
            if not y.index.equals(sample_weight.index):
                
                raise XyIndexError("index of sample_weight not equal to y_pred or y_true")          


    def _check_yname(self,y):
        
        if y.name is None:
            
            raise ValueError("Series y must have a name!")
            
            
    def _check_breaks(self,breaks_list,digit=3):
                   
        def is_numeric_list(breaks):
            
            return all(np.isreal(i) for i in breaks)
        
        def get_breaks(breaks,digit=digit):
        
            if is_numeric_list(breaks):
        
                return np.round(breaks,digit).tolist()
        
            else:
                
                return breaks       
        
        if isinstance(breaks_list,dict):
            
            return {key:get_breaks(breaks_list[key]) for key in breaks_list}
        
        elif isinstance(breaks_list,list):
            
            return get_breaks(breaks_list)
        
        else:
            
            raise ValueError("breaks_list is dict or list")
                
        
class BaseEval:    
    
    
    def _check_plot_params(self,show_plot,pred_desc):
        
        if not isinstance(show_plot ,tuple):
            
            raise ValueError("show_plot is tuple")  
            
        if not np.isin(show_plot,('ks', 'lift', 'gain', 'roc', 'lz', 'pr', 'f1', 'density')).all() :
            
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
            
        if group is not None:    
            
            if not isinstance(group,pd.Series):
                
                raise DataTypeError("group is is not pandas.Series.")      
                
            if not y_pred.index.equals(group.index):
                    
                raise XyIndexError("index of group not equal to y_pred or y_true")     
            
        if sample_weight is not None:
            
            if not isinstance(sample_weight,pd.Series):
                
                raise DataTypeError("sample_weight is not pandas.Series.") 
                
            if not y_pred.index.equals(sample_weight.index):
                
                raise XyIndexError("index of sample_weight not equal to y_pred or y_true") 
            



    def _check_values(self,y_pred,y_true,group,sample_weight):   
        
        if pd.isnull(y_pred).any():
            
            raise ValueError("y_pred contains NAN")   
            
        if pd.isnull(y_true).any():
            
            raise yValueError("y contains NAN")  
            
        if not y_pred.size==y_true.size:
            
            raise ValueError("length of y_true and y_pred not same.")  
        
        if group is not None: 
            
            if pd.isnull(group).any():
            
                raise ValueError("group contains NAN")  

        if not np.isin(y_true.unique(),[0,1]).all():
        
            raise yValueError("vals of y in [0,1] and 0(no-event),1(event).")   
            
        if sample_weight is not None:          
                            
            if sample_weight.size!=y_pred.size or sample_weight.size!=y_true.size:
                
                raise ValueError("length of sample_weight not equal to y_true and y_pred") 
            
            if (sample_weight<0).any():
                
                raise ValueError("vals of sample_weight should be non-negative")  
                
            if pd.isnull(sample_weight).any():
                
                raise ValueError("sample_weight contains NAN")        
                
            