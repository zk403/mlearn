#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 09:32:09 2021

@author: zengke
"""
from BDMLtools.base import Base
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from BDMLtools.tuner.base import BaseTunner
from sklearn.model_selection import HalvingGridSearchCV,HalvingRandomSearchCV,RepeatedStratifiedKFold
from joblib import effective_n_jobs
import numpy as np

class hgridTuner(Base,BaseTunner):
    
    '''
    Xgb与Lgbm的sucessive halving搜索与sucessive halving搜索
    Parameters:
    --
        Estimator:拟合器,XGBClassifier、LGBMClassifier或CatBoostClassifier
        method:str,可选"h_grid"或"h_random"
        para_space:dict,参数空间,注意随机搜索与网格搜索对应不同的dict结构,参数空间写法见后                       
        n_candidates:int or 'exhaust',halving random_search的抽样候选参数个数,当method="h_grid"时该参数会被忽略
        factor:int,halving search中，1/factor的候选参数将被用于下一次迭代
        scoring:str,寻优准则,可选'auc','logoss',目前不支持sample_weight
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        cv:int,交叉验证的折数
        early_stopping_rounds=10,int,训练数据中validation_fraction比例的数据被作为验证数据进行early_stopping
        validation_fraction=0.1,float,进行early_stopping的验证集比例
        eval_metric=‘auc’,early_stopping的评价指标,为可被Estimator识别的格式,参考Estimator.fit中的eval_metric参数
        n_jobs,int,运行交叉验证时的joblib的并行数,默认-1
        verbose,int,并行信息输出等级
        random_state:int,随机种子
        
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
    
    
    def __init__(self,Estimator,para_space,method='h_random',scoring='auc',eval_metric='auc',repeats=1,cv=5,
                 factor=3,n_candidates='exhaust',early_stopping_rounds=10,validation_fraction=0.1,
                 n_jobs=-1,verbose=0,random_state=123):
      
        self.Estimator=Estimator
        self.para_space=para_space
        self.method=method
        self.factor=factor
        self.n_candidates=n_candidates
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
        
        self.cat_features=X.select_dtypes(['object','category']).columns.tolist() if cat_features is None else cat_features    
        
        
        self._hgrid_search(X,y,sample_weight)
        
        #输出最优参数组合
        self.params_best=self.hgrid_res.best_params_
        self.cv_result=self._cvresult_to_df(self.hgrid_res.cv_results_)
        
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


    def _hgrid_search(self,X,y,sample_weight):          
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
                
                if self.method=='h_grid':          
                
                    self.hgrid_res=HalvingGridSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1),
                                              param_grid=self.para_space,
                                              cv=cv,
                                              refit=False,
                                              factor=self.factor,
                                              scoring=scorer,
                                              random_state=self.random_state,
                                              verbose=self.verbose,
                                              n_jobs=n_jobs).fit(X_tr,y_tr,
                                                                   eval_set=[(X_val, y_val)],
                                                                   eval_metric=self.eval_metric,
                                                                   callbacks=[early_stopping(self.early_stopping_rounds,first_metric_only=True,verbose=0),log_evaluation(0)],
                                                                   eval_sample_weight=[sample_weight_val],
                                                                   sample_weight=sample_weight_tr)  
                                                                    
                elif self.method=='h_random':                                                   
                                                                
                    self.hgrid_res=HalvingRandomSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1),
                                                 param_distributions=self.para_space,
                                                 n_candidates=self.n_candidates,
                                                 factor=self.factor,
                                                 cv=cv,
                                                 n_jobs=n_jobs,
                                                 verbose=self.verbose,
                                                 refit=True,
                                                 random_state=self.random_state,
                                                 scoring=scorer,
                                                 error_score=0).fit(X_tr,y_tr,
                                                                   eval_set=[(X_val, y_val)],
                                                                   eval_metric=self.eval_metric,
                                                                   callbacks=[early_stopping(self.early_stopping_rounds,first_metric_only=True,verbose=0),log_evaluation(0)],
                                                                   eval_sample_weight=[sample_weight_val],
                                                                   sample_weight=sample_weight_tr)    
                else:

                    raise ValueError('method in ("h_grid","h_random")')                                                                                                       
                                                                
                
            elif self.Estimator.__module__ == 'xgboost.sklearn': 
                
                if self.method=='h_grid':

                    self.hgrid_res=HalvingGridSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1),
                                              param_grid=self.para_space,
                                              cv=cv,
                                              refit=False,
                                              factor=self.factor,
                                              scoring=scorer,
                                              random_state=self.random_state,
                                              verbose=self.verbose,
                                              n_jobs=n_jobs).fit(X_tr,y_tr,eval_set=[(X_val, y_val)],
                                                                 sample_weight_eval_set=[sample_weight_val],
                                                                 sample_weight=sample_weight_tr,verbose=False)
                    
                elif self.method=='h_random':     

                    self.hgrid_res=HalvingRandomSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1),
                                                 param_distributions=self.para_space,
                                                 n_candidates=self.n_candidates,
                                                 factor=self.factor,
                                                 cv=cv,
                                                 n_jobs=n_jobs,
                                                 verbose=self.verbose,
                                                 refit=True,
                                                 random_state=self.random_state,
                                                 scoring=scorer,
                                                 error_score=0).fit(X_tr,y_tr,
                                                                    eval_set=[(X_val, y_val)],
                                                                    sample_weight_eval_set=[sample_weight_val],
                                                                    sample_weight=sample_weight_tr,verbose=False)    
                                                                    
                else:

                    raise ValueError('method in ("h_grid","h_random")')                                                                                                                                                  

            elif self.Estimator.__module__ == 'catboost.core': 
                
                from catboost import Pool
                from joblib import parallel_backend

                if self.eval_metric=='auc':

                    self.eval_metric='AUC'
                
                if self.method=='h_grid':
                
                    with parallel_backend('threading'):
                        
                        self.hgrid_res=HalvingGridSearchCV(self.Estimator(random_state=self.random_state,thread_count=1),
                                                 param_grid=self.para_space,
                                                 cv=cv,
                                                 factor=self.factor,
                                                 n_jobs=n_jobs,
                                                 refit=False,
                                                 verbose=self.verbose,
                                                 scoring=scorer,error_score=0).fit(X=X_tr,y=y_tr,cat_features=self.cat_features,sample_weight=sample_weight_tr,                                                               
                                                                 eval_set=Pool(X_val,y_val,cat_features=self.cat_features).set_weight(sample_weight_val),                                                                 
                                                                 early_stopping_rounds=self.early_stopping_rounds,
                                                                 verbose=False) 
                                                              
                elif self.method=='h_random': 
                    
                    with parallel_backend('threading'):
                        
                        self.hgrid_res=HalvingRandomSearchCV(self.Estimator(random_state=self.random_state,thread_count=1),
                                                    param_distributions=self.para_space,
                                                    n_candidates=self.n_candidates,
                                                    factor=self.factor,
                                                    cv=cv,
                                                    n_jobs=n_jobs,
                                                    verbose=self.verbose,
                                                    refit=True,
                                                    random_state=self.random_state,
                                                    scoring=scorer,
                                                    error_score=0).fit(X=X_tr,y=y_tr,cat_features=self.cat_features,sample_weight=sample_weight_tr,                                                               
                                                                 eval_set=Pool(X_val,y_val,cat_features=self.cat_features).set_weight(sample_weight_val),                                                                 
                                                                 early_stopping_rounds=self.early_stopping_rounds,
                                                                 verbose=False) 
                else:

                    raise ValueError('method in ("h_grid","h_random")')                                                                                                                                                  
                                                                                        
           
            else:

                raise ValueError('Estimator in ("XGBClassifier","LGBMClassifier","CatBoostClassifier")')  
                                               
        else:
            
            if self.Estimator.__module__ == 'catboost.core':
                
                if self.method=='h_grid':
                
                    hgrid=HalvingGridSearchCV(self.Estimator(random_state=self.random_state,thread_count=1),
                                             param_grid=self.para_space,
                                             cv=cv,
                                             factor=self.factor,
                                             n_jobs=n_jobs,
                                             refit=False,
                                             verbose=self.verbose,
                                             scoring=scorer,error_score=0)   
                    
                elif self.method=='h_random':
                    
                    hgrid=HalvingRandomSearchCV(self.Estimator(random_state=self.random_state,thread_count=1),
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
                    
                else:
                    
                    raise ValueError('method in ("h_grid","h_random")')  
                    
                self.hgrid_res=hgrid.fit(X,y,**{'sample_weight':sample_weight,'cat_features':self.cat_features})      
                
            elif self.Estimator.__module__ in ('lightgbm.sklearn','xgboost.sklearn'):
                
                if self.method=='h_grid':
                
                    hgrid=HalvingGridSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1),
                                              param_grid=self.para_space,
                                              cv=cv,
                                              refit=False,
                                              factor=self.factor,
                                              scoring=scorer,
                                              random_state=self.random_state,
                                              verbose=self.verbose,
                                              n_jobs=n_jobs)
                    
                    self.h_grid_res=hgrid.fit(X,y,sample_weight=sample_weight) 
                        
                elif self.method=='h_random':
                    
                    hgrid=HalvingRandomSearchCV(self.Estimator(random_state=self.random_state,n_jobs=1),
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
                    
                else:
                    
                    raise ValueError('method in ("grid","random_grid")')  
                    
                self.hgrid_res=hgrid.fit(X,y,sample_weight=sample_weight)
                
            else:               

                raise ValueError('Estimator in ("XGBClassifier","LGBMClassifier","CatBoostClassifier")') 
      
        return self 