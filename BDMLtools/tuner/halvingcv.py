#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 09:32:09 2021

@author: zengke
"""

from sklearn.base import BaseEstimator
from BDMLtools.base import Base
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from BDMLtools.tuner.base import BaseTunner
from sklearn.model_selection import HalvingGridSearchCV,HalvingRandomSearchCV,RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from joblib import effective_n_jobs
from lightgbm.sklearn import LGBMClassifier
from catboost.core import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import early_stopping as lgbm_early_stopping


class hgridTuner(Base,BaseTunner,BaseEstimator):
    
    '''
    Xgb与Lgbm的sucessive halving搜索与sucessive halving搜索
    Parameters:
    --
        Estimator:拟合器,XGBClassifier、LGBMClassifier或CatBoostClassifier
        method:str,可选"h_grid"或"h_random"
        cat_features:list or None,分类特征名,仅当Estimator为LGBMClassifier或CatBoostClassifier时启用
        para_space:dict,参数空间,注意随机搜索与网格搜索对应不同的dict结构,参数空间写法见后                       
        n_candidates:int or 'exhaust',halving random_search的抽样候选参数个数,当method="h_grid"时该参数会被忽略
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
        
            当Estimator=XGBClassifier,method="h_grid":
                
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
                     
              
             当Estimator=LGBMClassifier,method="h_grid": 
                 
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
                     
             
             当Estimator=CatBoostClassifier,method="h_grid": 
            
                 para_space={
                     'nan_mode':['Min'],
                     'n_estimators': [80, 100],
                     'learning_rate': [0.03,0.05, 0.1],
                     'max_depth': [2,3],
                     'scale_pos_weight': [1],
                     'subsample': [1],
                     'colsample_bylevel': [1],
                     'reg_lambda': [0]
                 }
            
            
             当Estimator=CatBoostClassifier,method="h_random": 


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
        h_grid_res:method="h_grid"下的优化结果,需先使用fit
        h_random_res:method="h_random"下的优化结果,需先使用fit
        model_refit:最优参数下的模型,需先使用fit
        
    Examples
    --

    '''     
    
    
    def __init__(self,Estimator,para_space,cat_features=None,method='h_random',scoring='auc',repeats=1,cv=5,
                 factor=3,n_candidates='exhaust',early_stopping_rounds=10,validation_fraction=0.1,
                 n_jobs=-1,verbose=0,random_state=123,sample_weight=None,calibration=False,cv_calibration=5):
      
        self.Estimator=Estimator
        self.para_space=para_space
        self.cat_features=cat_features
        self.method=method
        self.factor=factor
        self.n_candidates=n_candidates
        self.scoring=scoring
        self.cv=cv
        self.repeats=repeats
        self.early_stopping_rounds=early_stopping_rounds
        self.validation_fraction=validation_fraction
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
            
            
        if self.early_stopping_rounds:
            
            X, X_val, y, y_val = train_test_split(X,y,test_size=self.validation_fraction,random_state=self.random_state,stratify=y)

            
        if self.method=='h_grid':
            
            self._h_grid_search(X,y)
            #输出最优参数组合
            self.params_best=self.h_grid_res.best_params_
            self.cv_result=self._cvresult_to_df(self.h_grid_res.cv_results_)
            
            if self.early_stopping_rounds:
            
                if self.Estimator is CatBoostClassifier:
                    
                    self.model_refit = self.Estimator(random_state=self.random_state,**self.params_best).fit(X,y,sample_weight=self.sample_weight,eval_set=[(X_val,y_val)],early_stopping_rounds=self.early_stopping_rounds,cat_features=self.cat_features)     
                    
                elif self.Estimator is LGBMClassifier:
                    
                    self.model_refit = self.Estimator(random_state=self.random_state,**self.params_best).fit(X,y,sample_weight=self.sample_weight,eval_set=[(X_val,y_val)],callbacks=[lgbm_early_stopping(self.early_stopping_rounds)])                    
                    
                else:
            
                    self.model_refit = self.Estimator(random_state=self.random_state,**self.params_best).fit(X,y,sample_weight=self.sample_weight,eval_set=[(X_val,y_val)],early_stopping_rounds=self.early_stopping_rounds)
                               
                self.params_best['best_iteration']=self.model_refit.best_iteration if self.Estimator is XGBClassifier else self.model_refit.best_iteration_                              
            
            else:
                
                if self.Estimator is CatBoostClassifier:
                    
                    self.model_refit = self.Estimator(random_state=self.random_state,**self.params_best).fit(X,y,sample_weight=self.sample_weight,cat_features=self.cat_features)             
                
                else:
                
                    self.model_refit = self.Estimator(random_state=self.random_state,**self.params_best).fit(X,y,sample_weight=self.sample_weight)
                          
            if self.calibration:
                
                self.model_refit=CalibratedClassifierCV(self.model_refit,cv=self.cv_calibration,
                                                      n_jobs=self.n_jobs).fit(X,y,sample_weight=self.sample_weight)
            
        elif self.method=='h_random':
            
            self._h_random_search(X,y)
            #输出最优参数组合
            self.params_best=self.h_random_res.best_params_
            self.cv_result=self._cvresult_to_df(self.h_random_res.cv_results_)
            
            if self.early_stopping_rounds:
                
                if self.Estimator is CatBoostClassifier:
                    
                    self.model_refit = self.Estimator(random_state=self.random_state,**self.params_best).fit(X,y,sample_weight=self.sample_weight,eval_set=[(X_val,y_val)],early_stopping_rounds=self.early_stopping_rounds,cat_features=self.cat_features)      
                
                elif self.Estimator is LGBMClassifier:
                    
                    self.model_refit = self.Estimator(random_state=self.random_state,**self.params_best).fit(X,y,sample_weight=self.sample_weight,eval_set=[(X_val,y_val)],callbacks=[lgbm_early_stopping(self.early_stopping_rounds)])                    
        
                else:
            
                    self.model_refit = self.Estimator(random_state=self.random_state,**self.params_best).fit(X,y,sample_weight=self.sample_weight,eval_set=[(X_val,y_val)],early_stopping_rounds=self.early_stopping_rounds)
                               
                self.params_best['best_iteration']=self.model_refit.best_iteration if self.Estimator is XGBClassifier else self.model_refit.best_iteration_                              
             
            else:
                
                if self.Estimator is CatBoostClassifier:
                    
                    self.model_refit = self.Estimator(random_state=self.random_state,**self.params_best).fit(X,y,sample_weight=self.sample_weight,cat_features=self.cat_features)             
                
                else:
                
                    self.model_refit = self.Estimator(random_state=self.random_state,**self.params_best).fit(X,y,sample_weight=self.sample_weight)
              
            if self.calibration:
                
                self.model_refit=CalibratedClassifierCV(self.model_refit,cv=self.cv_calibration,
                                                      n_jobs=self.n_jobs).fit(X,y,sample_weight=self.sample_weight)
            
        else:
            
            raise ValueError('method should be "h_random" or "h_grid".')
                       
        self._is_fitted=True
        
        return self    
    
    def _h_grid_search(self,X,y):          
        '''
        网格搜索
        '''  
        if self.scoring in ['ks','auc','lift','neglogloss']:
            
            scorer=self._get_scorer[self.scoring]
            
        else:
            
            raise ValueError('scoring not understood,should be "ks","auc","lift","neglogloss")')
            
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state) 
        
        n_jobs=effective_n_jobs(self.n_jobs) 
        
        hgrid=HalvingGridSearchCV(self.Estimator(random_state=self.random_state,verbose=0),
                                  param_grid=self.para_space,
                                  cv=cv,
                                  refit=True,
                                  factor=self.factor,
                                  scoring=scorer,
                                  random_state=self.random_state,
                                  verbose=self.verbose,
                                  n_jobs=n_jobs)
        
        if self.Estimator is CatBoostClassifier:
        
            self.h_grid_res=hgrid.fit(X,y,sample_weight=self.sample_weight,cat_features=self.cat_features)
            
        else:
            
            self.h_grid_res=hgrid.fit(X,y,sample_weight=self.sample_weight)
           
        return self
        
        
    def _h_random_search(self,X,y):  
        '''
        随机网格搜索
        '''         
        
        if self.scoring in ['ks','auc','lift','neglogloss']:
            
            scorer=self._get_scorer[self.scoring]
            
        else:
            
            raise ValueError('scoring not understood,should be "ks","auc","lift","neglogloss")')
        
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state) 
        
        n_jobs=effective_n_jobs(self.n_jobs) 
        
        h_r_grid=HalvingRandomSearchCV(self.Estimator(random_state=self.random_state,verbose=0),
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
        
        if self.Estimator is CatBoostClassifier:
        
            self.h_random_res=h_r_grid.fit(X,y,sample_weight=self.sample_weight,cat_features=self.cat_features)
            
        else:
            
            self.h_random_res=h_r_grid.fit(X,y,sample_weight=self.sample_weight)       
        
        return self   