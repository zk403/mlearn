#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 09:32:09 2021

@author: zengke
"""

from sklearn.base import BaseEstimator
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd

class girdTunner(BaseEstimator):
    
    def __init__(self,Estimator,para_space,method='random_gird',n_iter=10,scoring='auc',cv=5):
        '''
        Xgb与Lgbm的网格搜索与随机搜索
        Parameters:
        --
            Estimator:拟合器,XGBClassifier或LGBMClassifier
            method:str,可选"gird"或"random_gird"
            para_space:dict,参数空间,注意随机搜索与网格搜索对应不同的dict结构,参数空间写法见后
                        
            n_iter:随机网格搜索迭代次数,当method="gird"时该参数会被忽略
            scoring:str,寻优准则,可选'auc','ks','lift'
            cv:int,交叉验证的折数
            
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
                         'scale_pos_weight':[1,11],
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
        self.Estimator=Estimator
        self.para_space=para_space
        self.method=method
        self.n_iter=n_iter
        self.scoring=scoring
        self.cv=cv

        
    def predict_proba(self,X,y=None):
        '''
        最优参数下的模型的预测
        Parameters:
        --
        X:pd.DataFrame对象
        '''      
        pred = self.model_refit.predict_proba(X)[:,1]        
        return pred
    
    def transform(self,X,y=None):   
         
        return self
          
    def fit(self,X,y):
        '''
        进行参数优化
        Parameters:
        --
        X:pd.DataFrame对象
        y:目标变量,pd.Series对象
        '''   
        
        self.X=X
        self.y=y
        
        if self.method=='gird':
            
            self.gird_search()
            #输出最优参数组合
            self.params_best=self.gird_res.best_params_
            self.cv_result=self.cvresult_to_df(self.gird_res.cv_results_)
        
        elif self.method=='random_gird':
            
            self.random_search()
            #输出最优参数组合
            self.params_best=self.r_gird_res.best_params_
            self.cv_result=self.cvresult_to_df(self.r_gird_res.cv_results_)
            
        else:
            raise IOError('method should be "gird" or "random_gird".')
            
        
        if self.Estimator is XGBClassifier:
            
            self.model_refit = XGBClassifier(
                colsample_bytree=self.params_best['colsample_bytree'],
                gamma=self.params_best['gamma'],
                scale_pos_weight=self.params_best['scale_pos_weight'],
                learning_rate=self.params_best['learning_rate'],          
                max_delta_step=self.params_best['max_delta_step'],
                max_depth=self.params_best['max_depth'],
                min_child_weight=self.params_best['min_child_weight'],
                n_estimators=self.params_best['n_estimators'],
                reg_lambda=self.params_best['reg_lambda'],
                subsample=self.params_best['subsample']
                ).fit(X,y)          
            
        elif self.Estimator is LGBMClassifier:
            
            self.model_refit = LGBMClassifier(
                boosting_type=self.params_best['boosting_type'],
                n_estimators=self.params_best['n_estimators'],
                learning_rate=self.params_best['learning_rate'],
                max_depth=self.params_best['max_depth'],          
                min_split_gain=self.params_best['min_split_gain'],
                min_child_weight=self.params_best['min_child_weight'],              
                subsample=self.params_best['subsample'],
                colsample_bytree=self.params_best['colsample_bytree'],
                scale_pos_weight=self.params_best['scale_pos_weight'],
                reg_lambda=self.params_best['reg_lambda']
                ).fit(X,y)    
        
        else:
            raise IOError('Estimator should be XGBClassifier or LGBMClassifier.')
            
        #交叉验证结果保存
             
        
        return self    
    
    def gird_search(self):          
        '''
        网格搜索
        '''  
        if self.scoring=='ks':
            scorer=metrics.make_scorer(self.custom_score_KS,greater_is_better=True,needs_proba=True)
        elif self.scoring=='auc':
            scorer=metrics.make_scorer(self.custom_score_AUC,greater_is_better=True,needs_proba=True)
        elif self.scoring=='lift':
            scorer=metrics.make_scorer(self.custom_score_Lift,greater_is_better=True,needs_proba=True)
        else:
            raise IOError('scoring not understood,should be "ks","auc","lift")')
        
        gird=GridSearchCV(self.Estimator(random_state=123),self.para_space,cv=self.cv,
                                scoring=scorer,error_score=0)    
        
        self.gird_res=gird.fit(self.X,self.y)
        
        return self
        
        
    def random_search(self):  
        '''
        随机网格搜索
        '''         
        
        if self.scoring=='ks':
            scorer=metrics.make_scorer(self.custom_score_KS,greater_is_better=True,needs_proba=True)
        elif self.scoring=='auc':
            scorer=metrics.make_scorer(self.custom_score_AUC,greater_is_better=True,needs_proba=True)
        elif self.scoring=='lift':
            scorer=metrics.make_scorer(self.custom_score_Lift,greater_is_better=True,needs_proba=True)
        else:
            raise IOError('scoring not understood,should be "ks","auc","lift")')
        
        r_gird=RandomizedSearchCV(self.Estimator(random_state=123),self.para_space,cv=self.cv,
                                scoring=scorer,error_score=0,n_iter=self.n_iter)
        
        self.r_gird_res=r_gird.fit(self.X,self.y)
        
        return self
    
    
    def custom_score_AUC(self,y_true, y_pred):        
        '''
        自定义验证评估指标AUC
        '''           
        return metrics.roc_auc_score(y_true,y_pred)
    
    def custom_score_KS(self,y_true, y_pred):
        '''
        自定义验证评估指标KS
        '''   
        fpr,tpr,thresholds= metrics.roc_curve(y_true,y_pred)
        ks = max(tpr-fpr)
        return ks             
        
        
    def custom_score_Lift(self,y_true,y_pred):
        '''
        自定义验证评估指标Lift
        '''   
        thrs = np.linspace(y_pred.min(), y_pred.max(),100)
        lift=[]
        for thr in thrs:
            tn, fp, fn, tp = metrics.confusion_matrix(y_true,y_pred>thr).ravel()
            #depth = (tp + fp)/(tn+fp+fn+tp)
            ppv = tp/(tp + fp)
            lift.append(ppv/((tp + fn)/(tn+fp+fn+tp)))
        return(np.nanmean(lift)) 
    
    def cvresult_to_df(self,cv_results_):
        '''
        输出交叉验证结果
        '''          
        return pd.DataFrame(cv_results_)    
    
    