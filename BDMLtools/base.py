#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 22:12:51 2022

@author: zengke
"""


from sklearn.exceptions import NotFittedError
from BDMLtools.exception import DataDtypesError,DataTypeError,XyIndexError,yValueError
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

        if X.select_dtypes(exclude=['number','object']).size:
            
            raise DataDtypesError("dtypes not in ('number' or 'object')")
            
            
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
            
        # if not np.equal(y.unique(),[0,1]).all():
            
        #     raise yValueError("vals of y in [0,1] and 0(no-event),1(event)")
            
        if X.select_dtypes(exclude=['number','object']).columns.size:
            
            raise DataDtypesError("dtypes not in ('number' or 'object')")
    
    def _check_param_dtype(self,dtype):
        
        if not dtype in ('float32','float64'):
            
            raise ValueError("dtype in ('float32','float64')")
        