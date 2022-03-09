#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 21:58:45 2021

@author: zengke
"""

from sklearn.base import BaseEstimator
import pandas as pd
import shap
from scipy.stats import pearsonr


class shapCheck(BaseEstimator):
    
    """ 
    对树模型结果进行woe/shap相关性校验,注意校验的模型与建模数据都需进行woe编码，若为原始数据模型则本校验将无意义
        
    Params:
    ------
    Estimator:lightgbm或xgboost模型对象
    limit:相关系数阈值,保留高于阈值的列
    p_value:相关系数显著性阈值,保留低于阈值的列,与limit为and关系
    Attributes:
    -------
    keep_col:list,保留的列的列名
    result:pd.DataFrame,shap-woe相关性分析结果
    """    
    
    def __init__(self,Estimator,limit=0,p_value=0.05):
        
        self.Estimator=Estimator
        self.limit=limit
        self.p_value=p_value
        
    def transform(self,X,y=None):
        """ 
        WOE转换
        """
        return X[self.keep_col]
            
          
    def fit(self,X,y=None):
        
        explainer = shap.TreeExplainer(self.Estimator)
        shap_values = explainer.shap_values(X)
        X_shap=pd.DataFrame(shap_values[1],columns=X.columns)

        result=self._check(X, X_shap)
        self.keep_col=result[result['pearsonr'].gt(self.limit) & result['p-value'].lt(self.p_value)].index.tolist()
        self.result=result
        
        return self   
    
    def _check(self,X,X_shap):
    
        index=[]
        corrs=[]
        ps=[]
        
        for col in X.columns:
            
            index.append(col)
            corr,p=pearsonr(X[col],X_shap[col])
            corrs.append(corr)    
            ps.append(p)   
            
        shap_woe_corr=pd.Series(corrs,index=index,name='pearsonr')
        shap_woe_corr_p=pd.Series(ps,index=index,name='p-value')        
        result=pd.concat([shap_woe_corr,shap_woe_corr_p],axis=1)     
        
        return result
      