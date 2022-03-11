#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:03:34 2022

@author: zengke
"""

from sklearn.feature_selection import RFECV,SequentialFeatureSelector
from lightgbm import LGBMClassifier
import pandas as pd
from joblib import effective_n_jobs
from BDMLtools.base import Base
from BDMLtools.tuner.base import BaseTunner
from sklearn.model_selection import RepeatedStratifiedKFold


class LgbmRFECVSelector(Base,BaseTunner):
    
    '''
    使用LightGBM进行基于交叉验证的递归式特征消除(Recursive feature elimination with CV)
    
    RFECV算法介绍:
    https://scikit-learn.org/stable/modules/feature_selection.html#rfe  
    
    目前不支持sample_weight
    
    Parameters:
    --
        step:int,float(0-1),RFE中每次消除特征的个数(int)/百分比(float)
        min_features_to_select:int,最少选择的特征个数
        clf_params:dict,LGBMClassifier的超参数设置,{}代表默认参数
        scoring:str,寻优准则,可选'auc','ks','lift','neglogloss'
        cv:int,RepeatedStratifiedKFold交叉验证的折数
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        random_state:int,随机种子
        n_jobs,int,运行交叉验证时的joblib的并行数,默认-1
        verbose,int,并行信息输出等级
    
    Attribute:    
    --
        keep_cols:list,保存的列名
        clf:RFECV对象
        
    Method:    
    --        
        fit(X,y,categorical_feature):拟合模型，categorical_feature为分类列列名list,默认None
        transform(X):对X进行特征筛选，返回筛选后的数据
        
    '''     

    def __init__(self,step=1,min_features_to_select=1,clf_params={},scoring='ks',cv=5,repeats=1,
                 categorical_feature=None,random_state=123,n_jobs=-1,verbose=0):
        
        self.step=step
        self.min_features_to_select=min_features_to_select
        self.clf_params=clf_params
        self.scoring=scoring
        self.cv=cv
        self.repeats=repeats
        #self.sample_weight=sample_weight
        self.random_state=random_state
        self.n_jobs=n_jobs
        self.verbose=verbose
        
        self._is_fitted=False
        
    def transform(self,X,y=None):
        
        """
        
        Parameters:
        --
            X:pd.DataFrame,训练数据X
        """

        self._check_X(X)
        self._check_is_fitted()

        return X[self.keep_cols]
            
    
    def fit(self,X,y,categorical_feature=None):
        
        """
        
        Parameters:
        --
            X:pd.DataFrame,训练数据X
            y:pd.Series,训练数据y
            categorical_feature:list,分类列列名,默认None即无分类特征

        """
                

        self._check_data(X, y)
        
        if categorical_feature:
        
            X=X.apply(lambda col:pd.Series(pd.Categorical(col),index=col.index) if col.name in categorical_feature else col)
        
        if self.scoring in ['ks','auc','lift','neglogloss']:
            
            scorer=self._get_scorer[self.scoring]
            
        else:
            
            raise ValueError('scoring not understood,should be "ks","auc","lift","neglogloss")')
            
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state)       
        
        n_jobs=effective_n_jobs(self.n_jobs)
        
        lgbm=LGBMClassifier(**self.clf_params)
        
        rfe_clf = RFECV(lgbm,step=self.step,
                        min_features_to_select=self.min_features_to_select,
                        importance_getter='feature_importances_',
                        n_jobs=n_jobs,cv=cv,scoring=scorer)
        
        rfe_clf.fit(X, y)
        
        self.keep_cols=rfe_clf.feature_names_in_[rfe_clf.get_support()].tolist()
        
        self.clf=rfe_clf
             
        self._is_fitted=True
    
        return self

    
    
class LgbmSeqSelector(Base,BaseTunner):
    
    '''
    使用LightGBM进行基于交叉验证的序列特征消除(Sequential Feature Selection with CV)
    
    RFECV算法介绍:
    https://scikit-learn.org/stable/modules/feature_selection.html#sequential-feature-selection
    
    目前不支持sample_weight
    
    Parameters:
    --
        direction:str,逐步法方向,可选'forward','backward'
        n_features_to_select:int,选择的特征个数
        clf_params:dict,LGBMClassifier的超参数设置,{}代表默认参数
        scoring:str,寻优准则,可选'auc','ks','lift','neglogloss'
        cv:int,RepeatedStratifiedKFold交叉验证的折数
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        random_state:int,随机种子
        n_jobs,int,运行交叉验证时的joblib的并行数,默认-1
        verbose,int,并行信息输出等级
    
    Attribute:    
    --
        keep_cols:list,保存的列名
        clf:SequentialFeatureSelector对象
        
    Method:    
    --        
        fit(X,y,categorical_feature):拟合模型，categorical_feature为分类列列名list,默认None
        transform(X):对X进行特征筛选，返回筛选后的数据
        
    '''     

    def __init__(self,direction='forward',n_features_to_select=5,clf_params={},scoring='ks',cv=5,repeats=1,
                 categorical_feature=None,random_state=123,n_jobs=-1,verbose=0):
        
        self.direction=direction
        self.n_features_to_select=n_features_to_select
        self.clf_params=clf_params
        self.scoring=scoring
        self.cv=cv
        self.repeats=repeats
        #self.sample_weight=sample_weight
        self.random_state=random_state
        self.n_jobs=n_jobs
        self.verbose=verbose
        
        self._is_fitted=False
        
    def transform(self,X,y=None):
        
        """
        
        Parameters:
        --
            X:pd.DataFrame,训练数据X
        """

        self._check_X(X)
        self._check_is_fitted()

        return X[self.keep_cols]
            
    
    def fit(self,X,y,categorical_feature=None):
        
        """
        
        Parameters:
        --
            X:pd.DataFrame,训练数据X
            y:pd.Series,训练数据y
            categorical_feature:list,分类列列名,默认None即无分类特征

        """
                

        self._check_data(X, y)
        
        if categorical_feature:
        
            X=X.apply(lambda col:pd.Series(pd.Categorical(col),index=col.index) if col.name in categorical_feature else col)
        
        if self.scoring in ['ks','auc','lift','neglogloss']:
            
            scorer=self._get_scorer[self.scoring]
            
        else:
            
            raise ValueError('scoring not understood,should be "ks","auc","lift","neglogloss")')
            
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state)       
        
        n_jobs=effective_n_jobs(self.n_jobs)
        
        lgbm=LGBMClassifier(**self.clf_params)
        
        rfe_clf = SequentialFeatureSelector(lgbm,
                        n_features_to_select=self.n_features_to_select,
                        direction=self.direction,
                        n_jobs=n_jobs,cv=cv,scoring=scorer)
        
        rfe_clf.fit(X, y)
        
        self.keep_cols=rfe_clf.feature_names_in_[rfe_clf.get_support()].tolist()
        
        self.clf=rfe_clf
             
        self._is_fitted=True
    
        return self    
    
