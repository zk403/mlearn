#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 09:32:09 2021

@author: zengke
"""

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold,train_test_split
from BDMLtools.base import Base
from BDMLtools.tuner.base import BaseTunner,FLLGBMSklearn,FocalLoss
from joblib import effective_n_jobs
from sklearn import metrics,__version__
import numpy as np
np.int=int

class gridTuner(Base,BaseTunner):
    
    '''
    Xgb与Lgbm的网格搜索与随机搜索
    Parameters:
    --
        Estimator:拟合器,XGBClassifier、LGBMClassifier或CatBoostClassifier
        method:str,可选"grid"或"random_grid"
        para_space:dict,参数空间,注意随机搜索与网格搜索对应不同的dict结构,参数空间写法见后                        
        n_iter:随机网格搜索迭代次数,当method="grid"时该参数会被忽略
        scoring:str,寻优准则,可选'auc','logloss',目前不支持sample_weight
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        cv:int,交叉验证的折数
        early_stopping_rounds=10,int,训练数据中validation_fraction比例的数据被作为验证数据进行early_stopping
        validation_fraction=0.1,float,进行early_stopping的验证集比例
        eval_metric=‘auc’,early_stopping的评价指标,为可被Estimator识别的格式,参考Estimator.fit中的eval_metric参数
        n_jobs,int,运行交叉验证时的joblib的并行数,默认-1
        verbose,int,并行信息输出等级
        random_state:int,随机种子
        
        """参数空间写法
            当Estimator=XGBClassifier,method="grid":
                
               para_space={
                     'n_estimators':[100],
                     'learning_rate':[0.1],
                    
                     'max_depth':[3],
                     'gamma': [0,10],
                     
                     'subsample':[0.6],
                     'colsample_bytree' :[0.6],
                     'reg_lambda':[0,10], 
                     'scale_pos_weight':[1],
                     'max_delta_step':[0],
                     'use_label_encoder':[False]
                     }
            
            当Estimator=XGBClassifier,method="random_grid":     
                  
               from scipy.stats import randint as sp_randint
               from scipy.stats import uniform as sp_uniform 
               
               para_space={
                     'n_estimators':sp_randint(low=60,high=120),#迭代次数
                     'learning_rate':sp_uniform(loc=0.05,scale=0.15), #学习率
                    
                     'max_depth':sp_randint(low=2,high=4),
                     'gamma': sp_uniform(loc=0,scale=21),
                     'min_child_weight': sp_uniform(loc=0,scale=21),
                     
                     'subsample':sp_uniform(loc=0.5,scale=0.5),
                     'colsample_bytree' :sp_uniform(loc=0.5,scale=0.5),
                     
                     'reg_lambda':sp_randint(low=0,high=1), 
                     'max_delta_step':sp_uniform(loc=0,scale=0),
                     'use_label_encoder':[False]                      
                     } 
              
             当Estimator=LGBMClassifier,method="grid": 
                 
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
                
             当Estimator=LGBMClassifier,method="random_grid": 
                 
                 from scipy.stats import randint as sp_randint
                 from scipy.stats import uniform as sp_uniform 
                 
                 para_space={
                     'boosting_type':['gbdt','goss'], #'goss','gbdt'
                     'n_estimators':sp_randint(low=100,high=110),
                     'learning_rate':sp_uniform(loc=0.1,scale=0), 
                    
                     'max_depth':sp_randint(low=2,high=4),#[0,∞],
                     'min_split_gain': sp_uniform(loc=0,scale=0),
                     'min_child_samples': sp_randint(low=100,high=300),#[0,∞],
                     
                     'scale_pos_weight':[1],
                     'subsample':sp_uniform(loc=0.5,scale=0.5),
                     'colsample_bytree' :sp_uniform(loc=0.5,scale=0.5),
                     'reg_lambda':sp_uniform(loc=0,scale=20),
                     }
                     
            当Estimator=CatBoostClassifier,method="grid": 
            
                    para_space={
                     'nan_mode':['Min'],
                     'n_estimators': [80, 100],
                     'learning_rate': [0.03,0.05, 0.1],
                     'max_depth': [2,3],
                     'scale_pos_weight': [1],
                     'subsample': [1],
                     'colsample_bylevel': [1],
                     'reg_lambda': [0]}
            
            
            当Estimator=CatBoostClassifier,method="random_grid": 


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
        grid_res:method="grid"下的网格优化结果,需先使用fit
        r_grid_res:method="random_grid"下的网格优化结果,需先使用fit
        model_refit:最优参数下的模型,需先使用fit
        
    Examples
    --        

    '''    
    
    
    def __init__(self,Estimator,para_space,method='random_grid',n_iter=10,scoring='auc',eval_metric='auc',repeats=1,cv=5,early_stopping_rounds=10,validation_fraction=0.1,
                 n_jobs=-1,verbose=0,random_state=123):
       
        self.Estimator=Estimator
        self.para_space=para_space
        self.method=method
        self.n_iter=n_iter
        self.scoring=scoring
        self.cv=cv
        self.repeats=repeats
        self.early_stopping_rounds=early_stopping_rounds
        self.eval_metric=eval_metric
        self.validation_fraction=validation_fraction
        self.n_jobs=n_jobs
        self.verbose=verbose     
        self.random_state=random_state
        
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
        
        self._check_is_fitted()
        self._check_X(X)        

        return X
          
    def fit(self,X,y,cat_features=None,sample_weight=None):
        '''
        进行参数优化
        Parameters:
        --
        X:pd.DataFrame对象
        y:目标变量,pd.Series对象
        cat_features:list,分类特征列名列表,None是数据中的object,category类列将被识别为cat_features，当Estimator为Xgboost时将忽略该参数        
        sample_weight:pd.Series,样本权重,index必须与X,y一致,注意目前不支持样本权重应用于交叉验证寻优指标(scorer)
        '''   
        
        self._check_data(X, y)     
        self._check_ws(y, sample_weight)
        
        self.cat_features=X.head().select_dtypes(['object','category']).columns.tolist() if cat_features is None else cat_features    
   
        
        #search
        self._grid_search(X,y,sample_weight)
        
        #输出最优参数组合
        self.params_best=self.grid_res.best_params_
        self.cv_result=self._cvresult_to_df(self.grid_res.cv_results_)
        
        #refit with early_stopping_rounds  
        if self.Estimator.__module__ == 'catboost.core':
            
            refit_Estimator=self.Estimator(random_state=self.random_state,
                                           thread_count=effective_n_jobs(self.n_jobs),
                                           **self.params_best).set_params(**{'cat_features':self.cat_features})
         
        else :
            
            refit_Estimator=self.Estimator(random_state=self.random_state,
                                           n_jobs=effective_n_jobs(self.n_jobs),
                                           **self.params_best)

        self.model_refit = refit_Estimator.fit(X,y,sample_weight=sample_weight)   
 
        self._is_fitted=True
            
        return self    
    
    def _grid_search(self,X,y,sample_weight):          
        '''
        网格搜索
        '''  
        
        scorer=self._get_scorer(self.scoring) #sample_weight not supported yet
            
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state) 
        
        n_jobs=effective_n_jobs(self.n_jobs)              
         
         
        if self.early_stopping_rounds:
            
            X_tr, X_val, y_tr, y_val = train_test_split(X,y,test_size=self.validation_fraction,random_state=self.random_state,stratify=y)
                
            sample_weight_tr=sample_weight[y_tr.index] if sample_weight is not None else np.ones(y_tr.size)
            sample_weight_val=sample_weight[y_val.index] if sample_weight is not None else np.ones(y_val.size)
            

            if self.Estimator.__module__ == 'lightgbm.sklearn': 
                
                from lightgbm import early_stopping,log_evaluation
                
                if self.method=='grid':
                
                    self.grid_res=GridSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1),self.para_space,cv=cv,
                                  n_jobs=n_jobs,
                                  refit=False,
                                  verbose=self.verbose,
                                  scoring=scorer,error_score=0).fit(X_tr,y_tr,
                                                                   eval_set=[(X_val, y_val)],
                                                                   eval_metric=self.eval_metric,
                                                                   callbacks=[early_stopping(self.early_stopping_rounds,first_metric_only=True,verbose=0),log_evaluation(0)],
                                                                   eval_sample_weight=[sample_weight_val],
                                                                   sample_weight=sample_weight_tr)  
                                                                    
                elif self.method=='random_grid':                                                   
                                                                
                    self.grid_res=RandomizedSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1),self.para_space,cv=cv,
                                  n_jobs=n_jobs,
                                  refit=False,
                                  random_state=self.random_state,
                                  verbose=self.verbose,
                                  scoring=scorer,error_score=0).fit(X_tr,y_tr,
                                                                   eval_set=[(X_val, y_val)],
                                                                   eval_metric=self.eval_metric,
                                                                   callbacks=[early_stopping(self.early_stopping_rounds,first_metric_only=True,verbose=0),log_evaluation(0)],
                                                                   eval_sample_weight=[sample_weight_val],
                                                                   sample_weight=sample_weight_tr)    
                else:

                    raise ValueError('method in ("grid","random_grid")')                                                                                                       
                                                                
                
            elif self.Estimator.__module__ == 'xgboost.sklearn': 
                
                if self.method=='grid':

                    self.grid_res=GridSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1).set_params(**{"eval_metric":self.eval_metric,'early_stopping_rounds':self.early_stopping_rounds}),
                                              self.para_space,cv=cv,
                                              n_jobs=n_jobs,
                                              refit=False,
                                              verbose=self.verbose,
                                              scoring=scorer,error_score=0).fit(X_tr,y_tr,
                                                                                eval_set=[(X_val, y_val)],
                                                                                sample_weight_eval_set=[sample_weight_val],
                                                                                sample_weight=sample_weight_tr,verbose=False)
                    
                elif self.method=='random_grid':     

                    self.grid_res=RandomizedSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1).set_params(**{"eval_metric":self.eval_metric,'early_stopping_rounds':self.early_stopping_rounds}),
                                              self.para_space,cv=cv,
                                              n_jobs=n_jobs,
                                              refit=False,
                                              random_state=self.random_state,
                                              verbose=self.verbose,
                                              scoring=scorer,error_score=0).fit(X_tr,y_tr,
                                                                                eval_set=[(X_val, y_val)],
                                                                                sample_weight_eval_set=[sample_weight_val],
                                                                                sample_weight=sample_weight_tr,verbose=False)    
                                                                                
                else:

                    raise ValueError('method in ("grid","random_grid")')                                                                                                                                                  

            elif self.Estimator.__module__ == 'catboost.core': 
                
                from catboost import Pool
                from joblib import parallel_backend

                if self.eval_metric=='auc':

                    self.eval_metric='AUC'
                
                if self.method=='grid':
                
                    with parallel_backend('threading'):
                        
                        self.grid_res=GridSearchCV(self.Estimator(random_state=self.random_state,thread_count=1,eval_metric=self.eval_metric),
                                          self.para_space,cv=cv,
                                          n_jobs=n_jobs,
                                          refit=False,
                                          error_score=0,
                                          scoring=scorer).fit(X=X_tr,y=y_tr,cat_features=self.cat_features,sample_weight=sample_weight_tr,                                                               
                                                                 eval_set=Pool(X_val,y_val,cat_features=self.cat_features).set_weight(sample_weight_val),                                                                 
                                                                 early_stopping_rounds=self.early_stopping_rounds,
                                                                 verbose=False) 
                                                              
                elif self.method=='random_grid': 

                    print(sample_weight_tr.size,sample_weight_val.size)
                    
                    with parallel_backend('threading'):
                        
                        self.grid_res=RandomizedSearchCV(self.Estimator(random_state=self.random_state,thread_count=1,eval_metric=self.eval_metric,),
                                          self.para_space,cv=cv,
                                          n_jobs=n_jobs,
                                          refit=False,
                                          random_state=self.random_state,
                                          error_score=0,
                                          scoring=scorer).fit(X=X_tr,y=y_tr,cat_features=self.cat_features,sample_weight=sample_weight_tr,                                                               
                                                                 eval_set=Pool(X_val,y_val,cat_features=self.cat_features).set_weight(sample_weight_val),                                                                 
                                                                 early_stopping_rounds=self.early_stopping_rounds,
                                                                 verbose=False) 
                else:

                    raise ValueError('method in ("grid","random_grid")')                                                                                                                                                  
                                                                                        
           
            else:

                raise ValueError('Estimator in ("XGBClassifier","LGBMClassifier","CatBoostClassifier")')  

        else:                                        
            
            if self.Estimator.__module__ == 'catboost.core':
                
                if self.method=='grid':
                
                    grid=GridSearchCV(self.Estimator(random_state=self.random_state,thread_count=1),self.para_space,cv=cv,
                                      n_jobs=n_jobs,
                                      refit=False,
                                      verbose=self.verbose,
                                      scoring=scorer,error_score=0)   
                    
                elif self.method=='random_grid':

                    grid=RandomizedSearchCV(self.Estimator(random_state=self.random_state,thread_count=1),self.para_space,cv=cv,
                                      n_jobs=n_jobs,
                                      random_state=self.random_state,
                                      refit=False,
                                      verbose=self.verbose,
                                      scoring=scorer,error_score=0)                              
                    
                else:
                    
                    raise ValueError('method in ("grid","random_grid")')  
                    
                self.grid_res=grid.fit(X,y,sample_weight=sample_weight,cat_features=self.cat_features)
               
                
            elif self.Estimator.__module__ in ('lightgbm.sklearn','xgboost.sklearn'):
                
                if self.method=='grid':
                
                    grid=GridSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1),self.para_space,cv=cv,
                                      n_jobs=n_jobs,
                                      refit=False,
                                      verbose=self.verbose,
                                      scoring=scorer,error_score=0)   
                    
                elif self.method=='random_grid':
                    
                    grid=RandomizedSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1),self.para_space,cv=cv,
                                      n_jobs=n_jobs,
                                      random_state=self.random_state,
                                      refit=False,
                                      verbose=self.verbose,
                                      scoring=scorer,error_score=0)                      
                else:
                    
                    raise ValueError('method in ("grid","random_grid")')               
            
                self.grid_res=grid.fit(X,y,sample_weight=sample_weight)
                
            else:
                    
                raise ValueError('Estimator in ("XGBClassifier","LGBMClassifier","CatBoostClassifier")') 
        
        return self 
    
    
    
class FLgridTuner(Base,BaseTunner):
    
    '''
    使用二分类Focal Loss、网格优化的梯度提升模型(目前仅支持Sklearn接口下的LightGBM)
    
        + 为解决样本不平衡问题而被提出的Focal Loss，通过引入类权重与对减少易分类样本的损失贡献同时增加难分类样本的损失贡献来减少样本不平衡问题
        + Focal Loss是一种变种的交叉熵损失函数,存在一阶、二阶导数，因此满足lgbm、xgb的要求
        + 对于样本不平衡问题,相较于传统样本处理方法(欠抽样、过抽样、smote、加权...)，Focal Loss提供了新的思路(由损失函数的构造方向解决样本不平衡问题)
        + Focal Loss中包含了类权重(alpha)，范围为(0,1),因此本模块不支持额外得再设定类权重、样本权重
        + Focal Loss中的gamma[0,+inf)值能减少易分类样本的损失贡献同时增加难分类样本的损失贡献。原始论文中gamma=2时效果最好 
                            
    参考文献:
    [FocalLoss论文:Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
    [Lightgbm中Focal Loss损失函数的应用-scipy求导](https://github.com/jrzaurin/LightGBM-with-Focal-Loss)
    [Lightgbm中Focal Loss损失函数的应用-手工求导](https://maxhalford.github.io/blog/lightgbm-focal-loss/)
    [Lightgbm中自定义损失函数下预测边际值无法复现的问题](https://github.com/microsoft/LightGBM/issues/3312)                        
                            
                            
    Parameters:
    --
        gamma:float [0,+inf),Focal loss的gamma值,设定gamma=0,alpha=None时传统交叉熵损失函数
        alpha:float (0,1),Focal loss的alpha值,设定gamma=0,alpha=None时传统交叉熵损失函数
        para_space:dict,参数空间,注意随机搜索与网格搜索对应不同的dict结构,参数空间写法见后      
        method:str,可选"grid"或"random_grid"                  
        n_iter:随机网格搜索迭代次数,当method="grid"时该参数会被忽略
        scoring:str,寻优准则,可选'auc','logloss','focalloss'
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        cv:int,交叉验证的折数
        early_stopping_rounds=10,int,训练数据中validation_fraction比例的数据被作为验证数据进行early_stopping
        validation_fraction=0.1,float,进行early_stopping的验证集比例
        eval_metric=‘auc’,early_stopping的评价指标,为可被Estimator识别的格式,参考Estimator.fit中的eval_metric参数
        n_jobs,int,运行交叉验证时的joblib的并行数,默认-1
        verbose,int,并行信息输出等级
        random_state:int,随机种子
        
        """参数空间写法              
             当Estimator=LGBMClassifier,method="grid": 
                 
                para_space={
                     'boosting_type':['gbdt','goss'], 
                     'n_estimators':[100],
                     'learning_rate':[0.1], 
                    
                     'max_depth':[3],#[0,∞],
                     'min_split_gain': [0,0.02] #0-0.02,
                     'min_child_weight':[0],
                     
                     'scale_pos_weight':[1],
                     'subsample':[0.6,0.8],
                     'colsample_bytree' :[0.6,0.8],
                     'reg_lambda':[0,0.01] #0-0.01, 
                     }
                
             当Estimator=LGBMClassifier,method="random_grid": 
                 
                 from scipy.stats import randint as sp_randint
                 from scipy.stats import uniform as sp_uniform 
                 
                 para_space={
                     'boosting_type':['gbdt','goss'], #'goss','gbdt'
                     'n_estimators':sp_randint(low=100,high=110),
                     'learning_rate':sp_uniform(loc=0.1,scale=0), 
                    
                     'max_depth':sp_randint(low=2,high=4),#[0,∞],
                     'min_split_gain': sp_uniform(loc=0,scale=0),#0-0.02
                     'min_child_samples': sp_randint(low=100,high=300),#[0,∞],
                     
                     'subsample':sp_uniform(loc=0.5,scale=0.5),
                     'colsample_bytree' :sp_uniform(loc=0.5,scale=0.5),
                     'reg_lambda':sp_uniform(loc=0,scale=20),#0-0.01
                     }
                
        """   
    
    Attribute:    
    --
        cv_result:交叉验证结果,需先使用fit
        params_best:最优参数组合,需先使用fit
        grid_res:method="grid"下的网格优化结果,需先使用fit
        r_grid_res:method="random_grid"下的网格优化结果,需先使用fit
        model_refit:最优参数下的模型,需先使用fit
        
    Examples
    --        

    '''    
    
    
    def __init__(self,para_space,gamma=2,alpha=0.25,method='random_grid',n_iter=10,scoring='focalloss',eval_metric='negfocalloss',repeats=1,cv=5,early_stopping_rounds=10,validation_fraction=0.1,
                 n_jobs=-1,verbose=0,random_state=123):
       
        self.gamma=gamma
        self.alpha=alpha
        self.para_space=para_space
        self.method=method
        self.n_iter=n_iter
        self.scoring=scoring
        self.cv=cv
        self.repeats=repeats
        self.early_stopping_rounds=early_stopping_rounds
        self.eval_metric=eval_metric
        self.validation_fraction=validation_fraction
        self.n_jobs=n_jobs
        self.verbose=verbose     
        self.random_state=random_state
        
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
        
        self._check_is_fitted()
        self._check_X(X)

        return X
          
    def fit(self,X,y):
        '''
        进行参数优化
        Parameters:
        --
        X:pd.DataFrame对象
        y:目标变量,pd.Series对象       
        '''   
        
        self._check_data(X, y)     
        
        self.Estimator=FLLGBMSklearn        
        
        self._grid_search(X,y)
        
        #输出最优参数组合
        self.params_best=self.grid_res.best_params_
        self.cv_result=self._cvresult_to_df(self.grid_res.cv_results_)

            
        refit_Estimator=self.Estimator(objective=FocalLoss(alpha=self.alpha,gamma=self.gamma).lgb_obj,
                                       boost_from_average=False,
                                       random_state=self.random_state,                                               
                                       n_jobs=effective_n_jobs(self.n_jobs),
                                       **self.params_best)

        self.model_refit = refit_Estimator.fit(X,y)   
            
                       
        self._is_fitted=True
            
        return self    
    
    def _grid_search(self,X,y,sample_weight=None):          
        '''
        网格搜索
        '''  
        if self.scoring in ['focalloss']:
            
            if __version__<'1.4.0':
            
                scorer=metrics.make_scorer(FocalLoss(alpha=self.alpha,gamma=self.gamma).lgb_scorer,greater_is_better=True,needs_proba=True)
                
            else:
                
                scorer=metrics.make_scorer(FocalLoss(alpha=self.alpha,gamma=self.gamma).lgb_scorer,greater_is_better=True,response_method="predict_proba")
            
        else:
            
            scorer=self._get_scorer(self.scoring) #sample_weight not supported yet
            
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state) 
        
        n_jobs=effective_n_jobs(self.n_jobs)                        
         
         
        if self.early_stopping_rounds:
            
            X_tr, X_val, y_tr, y_val = train_test_split(X,y,test_size=self.validation_fraction,random_state=self.random_state,stratify=y)
                
            sample_weight_tr=sample_weight[y_tr.index] if sample_weight is not None else np.ones(y_tr.size)
            sample_weight_val=sample_weight[y_val.index] if sample_weight is not None else np.ones(y_val.size)
            
            if self.eval_metric=="focalloss":
                
                self.eval_metric=FocalLoss(alpha=self.alpha,gamma=self.gamma).lgb_eval
                
            from lightgbm import early_stopping,log_evaluation
            
            if self.method=='grid':
            
                self.grid_res=GridSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1),self.para_space,cv=cv,
                              n_jobs=n_jobs,
                              refit=False,
                              verbose=self.verbose,
                              scoring=scorer,error_score=0).fit(X_tr,y_tr,
                                                               eval_set=[(X_val, y_val)],
                                                               eval_metric=self.eval_metric,
                                                               callbacks=[early_stopping(self.early_stopping_rounds,first_metric_only=True,verbose=0),log_evaluation(0)],
                                                               eval_sample_weight=[sample_weight_val],
                                                               sample_weight=sample_weight_tr)  
                                                                
            elif self.method=='random_grid':                                                   
                                                            
                self.grid_res=RandomizedSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1),self.para_space,cv=cv,
                              n_jobs=n_jobs,
                              refit=False,
                              random_state=self.random_state,
                              verbose=self.verbose,
                              scoring=scorer,error_score=0).fit(X_tr,y_tr,
                                                               eval_set=[(X_val, y_val)],
                                                               eval_metric=self.eval_metric,
                                                               callbacks=[early_stopping(self.early_stopping_rounds,first_metric_only=True,verbose=0),log_evaluation(0)],
                                                               eval_sample_weight=[sample_weight_val],
                                                               sample_weight=sample_weight_tr)    
            else:

                raise ValueError('method in ("grid","random_grid")')                                                                                                       

                                        
        else:    
            
            if self.method=='grid':
            
                grid=GridSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1),self.para_space,cv=cv,
                                  n_jobs=n_jobs,
                                  refit=False,
                                  verbose=self.verbose,
                                  scoring=scorer,error_score=0)   
                
            elif self.method=='random_grid':
                
                grid=RandomizedSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1),self.para_space,cv=cv,
                                  n_jobs=n_jobs,
                                  random_state=self.random_state,
                                  refit=False,
                                  verbose=self.verbose,
                                  scoring=scorer,error_score=0)  
                
            else:
                
                raise ValueError('method in ("grid","random_grid")')
            
            self.grid_res=grid.fit(X,y,sample_weight=sample_weight)
        
        return self 