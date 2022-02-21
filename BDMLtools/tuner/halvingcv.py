#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 09:32:09 2021

@author: zengke
"""

from sklearn.base import BaseEstimator
from BDMLtools.base import Base
from sklearn import metrics
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV,HalvingRandomSearchCV,RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import pandas as pd
from joblib import effective_n_jobs

class hgirdTuner(Base,BaseEstimator):
    
    '''
    Xgb与Lgbm的sucessive halving搜索与sucessive halving搜索
    Parameters:
    --
        Estimator:拟合器,XGBClassifier或LGBMClassifier
        method:str,可选"h_gird"或"h_random"
        para_space:dict,参数空间,注意随机搜索与网格搜索对应不同的dict结构,参数空间写法见后                       
        n_candidates:int or 'exhaust',halving random_search的抽样候选参数个数,当method="h_gird"时该参数会被忽略
        factor:int,halving search中，1/factor的候选参数将被用于下一次迭代
        scoring:str,寻优准则,可选'auc','ks','lift'
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        cv:int,交叉验证的折数
        n_jobs,int,运行交叉验证时的joblib的并行数,默认-1
        verbose,int,并行信息输出等级
        random_state:随机种子
        sample_weight:样本权重
        calibration:使用sklearn的CalibratedClassifierCV对refit=True下的模型进行概率校准
        cv_calibration:CalibratedClassifierCV的交叉验证数
        
        """参数空间写法
            当Estimator=XGBClassifier,method="h_gird":
                
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
            
            当Estimator=XGBClassifier,method="h_random":     
               
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
              
             当Estimator=LGBMClassifier,method="h_gird": 
                 
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
                
             当Estimator=LGBMClassifier,method="h_random": 
                 
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
        h_gird_res:method="h_gird"下的优化结果,需先使用fit
        h_random_res:method="h_random"下的优化结果,需先使用fit
        model_refit:最优参数下的模型,需先使用fit
        
    Examples
    --
    
    

    '''     
    
    
    def __init__(self,Estimator,para_space,method='h_random',scoring='auc',repeats=1,cv=5,
                 factor=3,n_candidates='exhaust',
                 n_jobs=-1,verbose=0,random_state=123,sample_weight=None,calibration=False,cv_calibration=5):
      
        self.Estimator=Estimator
        self.para_space=para_space
        self.method=method
        self.factor=factor
        self.n_candidates=n_candidates
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
        
        pred = self.model_refit.predict_proba(X)[:,1]    
        
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
        
        pred = self.model_refit.predict_proba(X)[:,1]  
        pred = self._p_to_score(pred,PDO,base,ratio)
        
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
        
        self._check_data(X, y)

        if self.method=='h_gird':
            
            self._h_gird_search(X,y)
            #输出最优参数组合
            self.params_best=self.h_gird_res.best_params_
            self.cv_result=self._cvresult_to_df(self.h_gird_res.cv_results_)
            self.model_refit = self.h_gird_res.best_estimator_   
            
            if self.calibration:
                
                self.model_refit=CalibratedClassifierCV(self.model_refit,cv=self.cv_calibration,
                                                      n_jobs=self.n_jobs).fit(X,y,sample_weight=self.sample_weight)
        
        elif self.method=='h_random':
            
            self._h_random_search(X,y)
            #输出最优参数组合
            self.params_best=self.h_random_res.best_params_
            self.cv_result=self._cvresult_to_df(self.h_random_res.cv_results_)
            self.model_refit = self.h_random_res.best_estimator_
            
            if self.calibration:
                
                self.model_refit=CalibratedClassifierCV(self.model_refit,cv=self.cv_calibration,
                                                      n_jobs=self.n_jobs).fit(X,y,sample_weight=self.sample_weight)
            
        else:
            raise ValueError('method should be "gird" or "random_gird".')
            
        #交叉验证结果保存             
        
        return self    
    
    def _h_gird_search(self,X,y):          
        '''
        网格搜索
        '''  
        if self.scoring=='ks':
            
            scorer=metrics.make_scorer(self._custom_score_KS,greater_is_better=True,needs_proba=True)
            
        elif self.scoring=='auc':
            
            scorer=metrics.make_scorer(self._custom_score_AUC,greater_is_better=True,needs_proba=True)
            
        elif self.scoring=='lift':
            
            scorer=metrics.make_scorer(self._custom_score_Lift,greater_is_better=True,needs_proba=True)
        else:
            raise ValueError('scoring not understood,should be "ks","auc","lift")')
            
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state) 
        
        n_jobs=effective_n_jobs(self.n_jobs) 
        
        hgird=HalvingGridSearchCV(self.Estimator(random_state=self.random_state),
                                  param_grid=self.para_space,
                                  cv=cv,
                                  refit=True,
                                  factor=self.factor,
                                  scoring=scorer,
                                  verbose=self.verbose,
                                  n_jobs=n_jobs)
        
        self.h_gird_res=hgird.fit(X,y,sample_weight=self.sample_weight)
        
        return self
        
        
    def _h_random_search(self,X,y):  
        '''
        随机网格搜索
        '''         
        
        if self.scoring=='ks':
            
            scorer=metrics.make_scorer(self._custom_score_KS,greater_is_better=True,needs_proba=True)
            
        elif self.scoring=='auc':
            
            scorer=metrics.make_scorer(self._custom_score_AUC,greater_is_better=True,needs_proba=True)
            
        elif self.scoring=='lift':
            
            scorer=metrics.make_scorer(self._custom_score_Lift,greater_is_better=True,needs_proba=True)
            
        else:
            
            raise ValueError('scoring not understood,should be "ks","auc","lift")')
        
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state) 
        
        n_jobs=effective_n_jobs(self.n_jobs) 
        
        h_r_gird=HalvingRandomSearchCV(self.Estimator(random_state=self.random_state),
                                     param_distributions=self.para_space,
                                     n_candidates=self.n_candidates,
                                     factor=self.factor,
                                     cv=cv,
                                     n_jobs=n_jobs,
                                     verbose=self.verbose,
                                     refit=True,
                                     random_state=self.random_state,
                                     scoring=scorer,
                                     error_score=0)
        
        self.h_random_res=h_r_gird.fit(X,y,sample_weight=self.sample_weight)
        
        return self
    
    
    def _custom_score_AUC(self,y_true, y_pred):        
        '''
        自定义验证评估指标AUC
        '''           
        return metrics.roc_auc_score(y_true,y_pred)
    
    def _custom_score_KS(self,y_true, y_pred):
        '''
        自定义验证评估指标KS
        '''   
        fpr,tpr,thresholds= metrics.roc_curve(y_true,y_pred)
        ks = max(tpr-fpr)
        return ks             
        
        
    def _custom_score_Lift(self,y_true,y_pred):
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
    
    def _cvresult_to_df(self,cv_results_):
        '''
        输出交叉验证结果
        '''          
        return pd.DataFrame(cv_results_)    
    
    
    def _p_to_score(self,pred,PDO=75,base=660,ratio=1/15):
        
        B=1*PDO/np.log(2)
        A=base + B*np.log(ratio)
        score=A-B*np.log(pred/(1-pred))
        
        return np.round(score,0)
    
    