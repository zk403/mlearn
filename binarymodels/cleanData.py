#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:25:57 2020

@author: kezeng
"""


# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.base import TransformerMixin

class OutliersTransformer(TransformerMixin):
    
    def __init__(self,columns=None,quantile_range=(0.01,0.99),na_option='keep'):
        """ 
        分位数替代法处理异常值
        Params:
        ------
            columns:list,替代法的列名list,默认为全部数值列
            quantile_range:list,分位数上下限阈值
            na_option:str,{'keep'},缺失值处理方式,默认为keep即保留缺失值
        Returns
        ------
        pandas.dataframe
            已经处理好异常值的数据框
        Examples
        ------        
        """
        self.columns=columns
        self.quantile_range = quantile_range
        
    def fit(self,X, y=None):        
        quantile_range=self.quantile_range
        if self.columns:
            self.quantile_data=X[self.columns].quantile([min(quantile_range),max(quantile_range)])        
        else:
            self.quantile_data=X.select_dtypes('number').quantile([min(quantile_range),max(quantile_range)])        
        return self
    
    def transform(self,X):
        quantile_range=self.quantile_range        
        pd.options.mode.chained_assignment = None
        X=X.copy()
        for column in self.quantile_data.columns:
            X[column][X[column]<self.quantile_data.loc[min(quantile_range),column]]=self.quantile_data.loc[min(quantile_range),column]
            X[column][X[column]>self.quantile_data.loc[max(quantile_range),column]]=self.quantile_data.loc[max(quantile_range),column]        
        return X