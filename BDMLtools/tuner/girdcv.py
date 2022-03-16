#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 09:32:09 2021

@author: zengke
"""

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from BDMLtools.base import Base
from BDMLtools.tuner.base import BaseTunner
from joblib import effective_n_jobs
from lightgbm.sklearn import LGBMClassifier
from catboost.core import CatBoostClassifier

class girdTuner(Base,BaseTunner,BaseEstimator):
    
    '''
    Xgb与Lgbm的网格搜索与随机搜索
    Parameters:
    --
        Estimator:拟合器,XGBClassifier、LGBMClassifier或CatBoostClassifier
        method:str,可选"gird"或"random_gird"
        cat_features:list or None,分类特征名,仅当Estimator为LGBMClassifier或CatBoostClassifier时启用
        para_space:dict,参数空间,注意随机搜索与网格搜索对应不同的dict结构,参数空间写法见后                        
        n_iter:随机网格搜索迭代次数,当method="gird"时该参数会被忽略
        scoring:str,寻优准则,可选'auc','ks','lift','neglogloss'
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        cv:int,交叉验证的折数
        n_jobs,int,运行交叉验证时的joblib的并行数,默认-1
        verbose,int,并行信息输出等级
        random_state:随机种子
        sample_weight:样本权重
        calibration:使用sklearn的CalibratedClassifierCV对refit=True下的模型进行概率校准
        cv_calibration:CalibratedClassifierCV的交叉验证数
        
        """参数空间写法
            当Estimator=XGBClassifier,method="gird":
                
                param_space={
                     'n_estimators':[100]
                     'learning_rate':[0.1],
                    
                     'max_depth':[3],
                     'gamma': [0,10],
                     'min_child_weight': 
                     
                     'subsample':[0.6,0.8],
                     'colsample_bytree' :[0.6,0.8],
                     'reg_lambda':[0,10], 
                     'scale_pos_weight':[1,10],
                     'max_delta_step':[0]
                     }
            
            当Estimator=XGBClassifier,method="random_gird":     
               
               from scipy.stats import randint as sp_randint
               from scipy.stats import uniform as sp_uniform 
               
               param_distributions={
                     'n_estimators':sp_randint(low=60,high=120),#迭代次数
                     'learning_rate':sp_uniform(loc=0.05,scale=0.15), #学习率
                    
                     'max_depth':sp_randint(low=2,high=4),
                     'gamma': sp_uniform(loc=0,scale=21),
                     'min_child_weight': sp_uniform(loc=0,scale=21),
                     
                     'subsample':sp_uniform(loc=0.5,scale=0.5),
                     'colsample_bytree' :sp_uniform(loc=0.5,scale=0.5),
                     
                     'reg_lambda':sp_randint(low=0,high=1), 
                     'scale_pos_weight':sp_uniform(loc=1,scale=0), 
                     'max_delta_step':sp_uniform(loc=0,scale=0)
                     } 
              
             当Estimator=LGBMClassifier,method="gird": 
                 
                para_space={
                     'boosting_type':['gbdt','goss'], 
                     'n_estimators':[100],
                     'learning_rate':[0.1], 
                    
                     'max_depth':[3],#[0,∞],
                     'min_split_gain': [0],
                     'min_child_weight':[0],
                     
                     'scale_pos_weight':[1],
                     'subsample':[0.6,0.8],
                     'colsample_bytree' :[0.6,0.8],
                     'reg_lambda':[0,10], 
                     }
                
             当Estimator=LGBMClassifier,method="random_gird": 
                 
                 from scipy.stats import randint as sp_randint
                 from scipy.stats import uniform as sp_uniform 
                 
                 para_space={
                     'boosting_type':['gbdt','goss'], #'goss','gbdt'
                     'n_estimators':sp_randint(low=100,high=110),
                     'learning_rate':sp_uniform(loc=0.1,scale=0), 
                    
                     'max_depth':sp_randint(low=2,high=4),#[0,∞],
                     'min_split_gain': sp_uniform(loc=0,scale=0),
                     'min_child_weight': sp_uniform(loc=0,scale=0),
                     
                     'scale_pos_weight':[1,11],
                     'subsample':sp_uniform(loc=0.5,scale=0.5),
                     'colsample_bytree' :sp_uniform(loc=0.5,scale=0.5),
                     'reg_lambda':sp_uniform(loc=0,scale=20),
                     }
                     
            当Estimator=CatBoostClassifier,method="gird": 
            
                    para_space={
                     'nan_mode':['Min'],
                     'n_estimators': [80, 100],
                     'learning_rate': [0.03,0.05, 0.1],
                     'max_depth': [2,3],
                     'scale_pos_weight': [1],
                     'subsample': [1],
                     'colsample_bylevel': [1],
                     'reg_lambda': [0]}
            
            
            当Estimator=CatBoostClassifier,method="random_gird": 


                 from scipy.stats import randint as sp_randint
                 from scipy.stats import uniform as sp_uniform 
                 
                 para_space={         
                     'nan_mode':['Min'],
                     'n_estimators':sp_randint(low=100,high=110),
                     'learning_rate':sp_uniform(loc=0.1,scale=0),                     
                     'max_depth':sp_randint(low=2,high=4),#[0,∞],                     
                     'scale_pos_weight':[1],
                     'subsample':sp_uniform(loc=0.5,scale=0.5),
                     'colsample_bylevel' :sp_uniform(loc=0.5,scale=0.5),
                     'reg_lambda':sp_uniform(loc=0,scale=20),
                     }
                
        """   
    
    Attribute:    
    --
        cv_result:交叉验证结果,需先使用fit
        params_best:最优参数组合,需先使用fit
        gird_res:method="gird"下的网格优化结果,需先使用fit
        r_gird_res:method="random_gird"下的网格优化结果,需先使用fit
        model_refit:最优参数下的模型,需先使用fit
        
    Examples
    --        

    '''    
    
    
    def __init__(self,Estimator,para_space,cat_features=None,method='random_gird',n_iter=10,scoring='auc',repeats=1,cv=5,
                 n_jobs=-1,verbose=0,random_state=123,sample_weight=None,calibration=False,cv_calibration=5):
       
        self.Estimator=Estimator
        self.para_space=para_space
        self.cat_features=cat_features
        self.method=method
        self.n_iter=n_iter
        self.scoring=scoring
        self.cv=cv
        self.repeats=repeats
        self.n_jobs=n_jobs
        self.verbose=verbose     
        self.random_state=random_state
        self.sample_weight=sample_weight
        self.calibration=calibration
        self.cv_calibration=cv_calibration
        
        self._is_fitted=False

        
    def predict_proba(self,X,y=None):
        '''
        最优参数下的模型的预测
        Parameters:
        --
        X:pd.DataFrame对象
        '''      
        self._check_is_fitted()
        self._check_X(X)
        
        pred = self.model_refit.predict_proba(self.transform(X))[:,1]        
        return pred
    
    def predict_score(self,X,y=None,PDO=75,base=660,ratio=1/15):
        '''
        最优参数下的模型的预测
        Parameters:
        --
        X:pd.DataFrame对象
        '''      
        self._check_is_fitted()
        self._check_X(X)
        
        pred = self.model_refit.predict_proba(self.transform(X))[:,1]  
        pred = self._p_to_score(pred,PDO,base,ratio)
        
        return pred
    
    def transform(self,X,y=None):  
        
        self._check_is_fitted()
        self._check_X(X)
        
        if self.Estimator is CatBoostClassifier:
            
            out=X.apply(lambda col:col.astype('str') if col.name in self.cat_features else col) if self.cat_features else X
            
        elif self.Estimator is LGBMClassifier:
        
            out=X.apply(lambda col:col.astype('category') if col.name in self.cat_features else col) if self.cat_features else X
            
        else:
            
            out=X

        return out
          
    def fit(self,X,y):
        '''
        进行参数优化
        Parameters:
        --
        X:pd.DataFrame对象
        y:目标变量,pd.Series对象
        '''   
        
        self._check_data(X, y)     

        if self.Estimator is CatBoostClassifier:
            
            X=X.apply(lambda col:col.astype('str') if col.name in self.cat_features else col) if self.cat_features else X
            
        elif self.Estimator is LGBMClassifier:
        
            X=X.apply(lambda col:col.astype('category') if col.name in self.cat_features else col) if self.cat_features else X
            
        else:
            
            X=X   

        
        if self.method=='gird':
            
            self._gird_search(X,y,self.sample_weight)
            #输出最优参数组合
            self.params_best=self.gird_res.best_params_
            self.cv_result=self._cvresult_to_df(self.gird_res.cv_results_)
            self.model_refit = self.gird_res.best_estimator_  
            
            if self.calibration:
                
                self.model_refit=CalibratedClassifierCV(self.model_refit,cv=self.cv_calibration,
                                                     n_jobs=self.n_jobs).fit(X,y,sample_weight=self.sample_weight)
        
        elif self.method=='random_gird':
            
            self._random_search(X,y,self.sample_weight)
            #输出最优参数组合
            self.params_best=self.r_gird_res.best_params_
            self.cv_result=self._cvresult_to_df(self.r_gird_res.cv_results_)
            self.model_refit = self.r_gird_res.best_estimator_
            
            if self.calibration:
                
                self.model_refit=CalibratedClassifierCV(self.model_refit,cv=self.cv_calibration,
                                                     n_jobs=self.n_jobs).fit(X,y,sample_weight=self.sample_weight)
            
        else:
            
            raise ValueError('method should be "gird" or "random_gird".')
            
        self._is_fitted=True
            
        #交叉验证结果保存             
        
        return self    
    
    def _gird_search(self,X,y,sample_weight):          
        '''
        网格搜索
        '''  
        if self.scoring in ['ks','auc','lift','neglogloss']:
            
            scorer=self._get_scorer[self.scoring]
            
        else:
            
            raise ValueError('scoring not understood,should be "ks","auc","lift","neglogloss")')
            
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state) 
        
        n_jobs=effective_n_jobs(self.n_jobs)              
        
        gird=GridSearchCV(self.Estimator(random_state=self.random_state,verbose=0),self.para_space,cv=cv,
                          n_jobs=n_jobs,
                          refit=True,
                          verbose=self.verbose,
                          scoring=scorer,error_score=0)    
        
        if self.Estimator is CatBoostClassifier:
        
            self.gird_res=gird.fit(X,y,sample_weight=sample_weight,cat_features=self.cat_features)
            
        else:
            
            self.gird_res=gird.fit(X,y,sample_weight=sample_weight)
        
        return self
        
        
    def _random_search(self,X,y,sample_weight):          
        '''
        随机网格搜索
        '''         
        
        if self.scoring in ['ks','auc','lift','neglogloss']:
            
            scorer=self._get_scorer[self.scoring]
            
        else:
            
            raise ValueError('scoring not understood,should be "ks","auc","lift","neglogloss")')
        
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state) 
        
        n_jobs=effective_n_jobs(self.n_jobs) 
        
        r_gird=RandomizedSearchCV(self.Estimator(random_state=self.random_state,verbose=0),self.para_space,cv=cv,
                                  n_jobs=n_jobs,verbose=self.verbose,refit=True,
                                  random_state=self.random_state,
                                  scoring=scorer,error_score=0,n_iter=self.n_iter)
        
        if self.Estimator is CatBoostClassifier:
        
            self.r_gird_res=r_gird.fit(X,y,sample_weight=sample_weight,cat_features=self.cat_features)
            
        else:
            
            self.r_gird_res=r_gird.fit(X,y,sample_weight=sample_weight)
        
        return self   