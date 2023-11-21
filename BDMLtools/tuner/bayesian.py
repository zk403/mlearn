#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:26:03 2021

@author: zengke
"""
from BDMLtools.base import Base
from BDMLtools.tuner.base import BaseTunner,FLLGBMSklearn,FocalLoss
#from sklearn.model_selection import GridSearchCV
#from bayes_opt import BayesianOptimization
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold,train_test_split
from scipy import special
#from lightgbm import early_stopping as lgbm_early_stopping
#from time import time
#import pandas as pd
from joblib import effective_n_jobs
#import numpy as np
from skopt import BayesSearchCV
from sklearn import metrics


class BayesianCVTuner(Base,BaseTunner):
    
    '''
    使用scikit-optmize进行贝叶斯超参优化，可支持分类超参，能够提供对sklearn生态更好的兼容性且效率较高
    
    scikit-optmize文档:https://scikit-optimize.github.io/stable/
    scikit-optmize源码:https://github.com/scikit-optimize/scikit-optimize
    
    Parameters:
    --
        Estimator:拟合器,XGBClassifier、LGBMClassifier或CatBoostClassifier
        para_space:dict,lgb的参数空间
        n_iter:贝叶斯优化搜索迭代次数
        init_points:int,贝叶斯优化起始搜索点的个数
        scoring:str,寻优准则,可选'auc','ks','lift','neglogloss'
        early_stopping_rounds=10,int,训练数据中validation_fraction比例的数据被作为验证数据进行early_stopping,
        validation_fraction=0.1,float,进行early_stopping的验证集比例
        eval_metric=‘auc’,early_stopping的评价指标,为可被Estimator识别的格式,参考Estimator.fit中的eval_metric参数
        cv:int,RepeatedStratifiedKFold交叉验证的折数
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        n_jobs,int,运行交叉验证和提升算法并行数,默认-1
        verbose,int,并行信息输出等级
        random_state,随机种子
        calibration:使用sklearn的CalibratedClassifierCV对refit模型进行概率校准
        cv_calibration:CalibratedClassifierCV的交叉验证数
        
        """参数空间写法 
        
            #XGBClassifier
            from skopt.utils import Categorical,Real,Integer
            
            para_space = {
                'n_estimators': Integer(30, 120),
                'max_depth':Integer(2, 4),
                'learning_rate': Real(0.05, 0.2,prior='uniform'),
                
                'gamma': Real(0,10,prior='uniform'),
                'subsample':Real(0.5,1.0,prior='uniform'),
                'colsample_bytree':Real(0.5,1.0,prior='uniform'),
                'reg_lambda':Real(0,10,prior='uniform'),
                
                'use_label_encoder':Categorical([False])
            
            }         
        
            #LGBMClassifier
            from skopt.utils import Categorical,Real,Integer
    
            para_space = {
                'boosting_type':Categorical(['gbdt','goss']),
                'n_estimators': Integer(30, 120),
                'max_depth':Integer(2, 4),
                'learning_rate': Real(0.05, 0.2,prior='uniform'),
            
                #'min_split_gain': Real(0, 10,prior='uniform'),
                'min_child_samples':Integer(1, 100,prior='uniform'),
                
                'subsample':Real(0.5,1.0,prior='uniform'),
                'colsample_bytree':Real(0.5,1.0,prior='uniform'),
                'reg_lambda':Real(0,10,prior='uniform'),    
            } 
                        
            #Catboost
            from skopt.utils import Categorical,Real,Integer
    
            para_space = {
            
                'n_estimators': Integer(30, 120),
                'learning_rate': Real(0.05, 0.2,prior='uniform'),
            
                'max_depth':Integer(2, 4),
                'min_child_samples':Integer(1, 100,prior='uniform'),
                
                'subsample':Real(0.5,1.0,prior='uniform'),
                'colsample_bylevel':Real(0.5,1.0,prior='uniform'),
                'reg_lambda':Real(0,10,prior='uniform')               
            }           
          """   
          
    Attribute:    
    --
        params_best:最优参数组合,需先使用fit
        model_refit:最优参数下的lgbm模型,需先使用fit
    
    Examples
    --

    '''    
    
    def __init__(self,Estimator,para_space={},n_iter=10,init_points=1,scoring='auc',eval_metric='auc',
                 cv=5,repeats=1,n_jobs=-1,verbose=0,early_stopping_rounds=10,validation_fraction=0.1,random_state=123,calibration=False,cv_calibration=5):
        
        self.Estimator=Estimator
        self.para_space=para_space
        self.n_iter=n_iter
        self.init_points=init_points
        self.scoring=scoring
        self.cv=cv
        self.repeats=repeats
        self.early_stopping_rounds=early_stopping_rounds
        self.eval_metric=eval_metric
        self.validation_fraction=validation_fraction
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.random_state=random_state
        self.calibration=calibration
        self.cv_calibration=cv_calibration
        
        self._is_fitted=False
        
    def predict_proba(self,X,y=None):
        '''
        最优参数下的lgbm模型的预测
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
        
        if self.Estimator.__module__ == "catboost.core":
            
            out=X.apply(lambda col:col.astype('str') if col.name in self.cat_features else col) if self.cat_features else X
            
        elif self.Estimator.__module__ == 'lightgbm.sklearn':
        
            out=X.apply(lambda col:col.astype('category') if col.name in self.cat_features else col) if self.cat_features else X
            
        else:
            
            out=X
            
        return out
          
    def fit(self,X,y,cat_features=None,sample_weight=None):
        '''
        进行贝叶斯优化
        Parameters:
        --
        X: pd.DataFrame对象
        y:目标变量,pd.Series对象
        cat_features:list,分类特征列名列表,None是数据中的object,category类列将被识别为cat_features，当Estimator为Xgboost时将忽略该参数        
        sample_weight:pd.Series,样本权重,index必须与X,y一致,注意目前不支持样本权重应用于交叉验证寻优指标(scorer)
        
        '''   
    
        self._check_data(X, y)
        self._check_ws(y, sample_weight)
        
        self.cat_features=X.select_dtypes(['object','category']).columns.tolist() if cat_features is None else cat_features    
        
        para_space=self._hpsearch_default(self.Estimator) if not self.para_space else self.para_space
        
        if self.Estimator.__module__ == "catboost.core":
            
            X=X.apply(lambda col:col.astype('str') if col.name in self.cat_features else col) if self.cat_features else X
            
        elif self.Estimator.__module__ == 'lightgbm.sklearn':
        
            X=X.apply(lambda col:col.astype('category') if col.name in self.cat_features else col) if self.cat_features else X
            
        else:
            
            X=X 
            
        
        if self.early_stopping_rounds:
 
            X_tr, X_val, y_tr, y_val = train_test_split(X,y,test_size=self.validation_fraction,random_state=self.random_state,stratify=y)
        
            sample_weight_tr=sample_weight[y_tr.index] if sample_weight is not None else None
        
            #BayesSearch_CV        
            self._BayesSearch_CV(para_space,X_tr,y_tr,sample_weight_tr)
            
            #params_best
            self.params_best=self.bs_res.best_params_
     
            #cv_result
            self.cv_result=self._cvresult_to_df(self.bs_res.cv_results_)
            
            #refit with early_stopping_rounds                  
            if self.Estimator.__module__ == "catboost.core":
                
                refit_Estimator=self.Estimator(random_state=self.random_state,
                                               thread_count=effective_n_jobs(self.n_jobs),
                                               **self.params_best)
                
                if self.eval_metric=='auc':
                    
                    self.eval_metric='AUC'
                
                refit_Estimator.set_params(**{"eval_metric":self.eval_metric})  
                
            #xgboost:eval_metric and early_stopping_rounds in `fit` method is deprecated for better compatibility with scikit-learn,version>=1.6.0
            elif self.Estimator.__module__ == 'xgboost.sklearn':
                    
                    refit_Estimator=self.Estimator(random_state=self.random_state,
                                                   thread_count=effective_n_jobs(self.n_jobs),
                                                   **self.params_best)
                    
                    refit_Estimator.set_params(**{"eval_metric":self.eval_metric,'early_stopping_rounds':self.early_stopping_rounds})                                                                               
                           
                
            else:
                
                refit_Estimator=self.Estimator(random_state=self.random_state,
                                               n_jobs=effective_n_jobs(self.n_jobs),
                                               **self.params_best)

            self.model_refit = refit_Estimator.fit(**self._get_fit_params(self.Estimator,X_tr,y_tr,X_val,y_val,sample_weight,y_tr.index,y_val.index))   
                                                                                                 
            self.params_best['best_iteration']=self.model_refit.best_iteration if self.Estimator.__module__ == 'xgboost.sklearn' else self.model_refit.best_iteration_ 
            
            
        else:
            
            #BayesSearch_CV        
            self._BayesSearch_CV(para_space,X,y,sample_weight)
            
            #params_best
            self.params_best=self.bs_res.best_params_
     
            #cv_result
            self.cv_result=self._cvresult_to_df(self.bs_res.cv_results_)
            
            #refit model
            if self.Estimator.__module__ == "catboost.core":
                
                refit_Estimator=self.Estimator(random_state=self.random_state,
                                               thread_count=effective_n_jobs(self.n_jobs),
                                               **self.params_best)
                
                refit_Estimator.set_params(**{'cat_features':self.cat_features})
                
            else:
                
                refit_Estimator=self.Estimator(random_state=self.random_state,
                                               n_jobs=effective_n_jobs(self.n_jobs),
                                               **self.params_best)

            self.model_refit = refit_Estimator.fit(X,y,sample_weight=sample_weight)             

        
        #get calibration done
        if self.calibration:

            self.model_refit=CalibratedClassifierCV(self.model_refit,cv=self.cv_calibration,
                                                  n_jobs=effective_n_jobs(self.n_jobs)).fit(X,y,sample_weight=sample_weight)

        self._is_fitted=True
        
        return self
    
    
    def _BayesSearch_CV(self,para_space,X,y,sample_weight):        
               
        if self.scoring in ['ks','auc','lift','neglogloss']:
            
            scorer=self._get_scorer[self.scoring]
            
        else:
            
            scorer=self.scoring
             
                    
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state)
        
        n_jobs=effective_n_jobs(self.n_jobs)
        
        if self.Estimator.__module__ == "catboost.core":
            
            bs=BayesSearchCV(
                self.Estimator(random_state=self.random_state,thread_count=n_jobs),para_space,cv=cv,
                n_iter=self.n_iter,n_points=self.init_points,
                n_jobs=-1 if self.n_jobs==-1 else 1,
                verbose=self.verbose,refit=False,random_state=self.random_state,
                scoring=scorer,error_score=0)    
            
            self.bs_res=bs.fit(X,y,**{'sample_weight':sample_weight,
                                      'cat_features':self.cat_features})
                    
        else:
                        
            bs=BayesSearchCV(
                self.Estimator(random_state=self.random_state,n_jobs=n_jobs),para_space,cv=cv,
                n_iter=self.n_iter,n_points=self.init_points,
                n_jobs=-1 if self.n_jobs==-1 else 1,
                verbose=self.verbose,refit=False,random_state=self.random_state,
                scoring=scorer,error_score=0)       
            
            self.bs_res=bs.fit(X,y,**{'sample_weight':sample_weight})
  
            
        return(self)     
    
    
    def _get_fit_params(self,Estimator,X_train,y_train,X_val,y_val,sample_weight=None,train_index=None,val_index=None):
        
        
        if Estimator.__module__ == "xgboost.sklearn":
            
            fit_params = {
                "X":X_train,
                "y":y_train,
                "eval_set": [(X_val, y_val)],
                #"eval_metric": self.eval_metric,
                #"early_stopping_rounds": self.early_stopping_rounds,
            }
            
            if sample_weight is not None:
                
                fit_params["sample_weight"] = sample_weight.loc[train_index]
                fit_params["sample_weight_eval_set"] = [sample_weight.loc[val_index]]
        
            
        elif Estimator.__module__ == 'lightgbm.sklearn':
            
            from lightgbm import early_stopping, log_evaluation

            fit_params = {
                "X":X_train,
                "y":y_train,
                "eval_set": [(X_val, y_val)],
                "eval_metric": self.eval_metric,
                "callbacks": [early_stopping(self.early_stopping_rounds, first_metric_only=True)],
            }
            
            if self.verbose >= 100:
                
                fit_params["callbacks"].append(log_evaluation(1))
                
            else:
                
                fit_params["callbacks"].append(log_evaluation(0))
                
            if sample_weight is not None:
                
                fit_params["sample_weight"] = sample_weight.loc[train_index]
                fit_params["eval_sample_weight"] = [sample_weight.loc[val_index]]
            
        
        elif Estimator.__module__ == "catboost.core":
            
            from catboost import Pool
            
            fit_params = {
                "X": Pool(X_train, y_train, cat_features=self.cat_features),
                "eval_set": Pool(X_val, y_val, cat_features=self.cat_features),
                "early_stopping_rounds": self.early_stopping_rounds,
                # Evaluation metric should be passed during initialization
            }
            
            if sample_weight is not None:
                fit_params["X"].set_weight(sample_weight.loc[train_index])
                fit_params["eval_set"].set_weight(sample_weight.loc[val_index])
            
        else:
            
            raise ValueError('Estimator in (XGBClassifier,LGBMClassifier,CatBoostClassifier)')
        
            
        return fit_params
    
    
    @staticmethod
    def _hpsearch_default(Estimator):
        
        from skopt.utils import Categorical,Real,Integer
        
        if Estimator.__module__ == "xgboost.sklearn":
            
            para_space = {
                'n_estimators': Integer(30, 120),
                'max_depth':Integer(2, 4),
                'learning_rate': Real(0.05, 0.2,prior='uniform'),
                
                'gamma': Real(0,10,prior='uniform'),
                'subsample':Real(0.5,1.0,prior='uniform'),
                'colsample_bytree':Real(0.5,1.0,prior='uniform'),
                'reg_lambda':Real(0,10,prior='uniform'),
                
                'use_label_encoder':Categorical([False])            
            } 
            
        elif Estimator.__module__ == 'lightgbm.sklearn':
            
            para_space = {
                
                'verbose':Categorical([-1]),
                'boosting_type':Categorical(['gbdt','goss']),
                'n_estimators': Integer(30, 120),
                'max_depth':Integer(2, 4),
                'learning_rate': Real(0.05, 0.2,prior='uniform'),
            
                'min_split_gain': Real(0, 10,prior='uniform'),
                'min_child_samples':Integer(1, 100,prior='uniform'),
                
                'subsample':Real(0.5,1.0,prior='uniform'),
                'colsample_bytree':Real(0.5,1.0,prior='uniform'),
                'reg_lambda':Real(0,10,prior='uniform'),    
            } 
            
        elif Estimator.__module__ == "catboost.core":
            
            para_space = {
            
                'n_estimators': Integer(30, 120),
                'learning_rate': Real(0.05, 0.2,prior='uniform'),
            
                'max_depth':Integer(2, 4),
                'min_child_samples':Integer(1, 100,prior='uniform'),
                
                'subsample':Real(0.5,1.0,prior='uniform'),
                'colsample_bylevel':Real(0.5,1.0,prior='uniform'),
                'reg_lambda':Real(0,10,prior='uniform')               
            }  
            
        else:
            
            raise ValueError('Estimator in (XGBClassifier,LGBMClassifier,CatBoostClassifier)')
        
        return para_space 
    
    
    
class FLBSTuner(Base,BaseTunner):
    
        '''
        使用二分类Focal Loss、贝叶斯超参优化(scikit-optmize)的梯度提升模型(目前仅支持Sklearn接口下的LightGBM)
        
            + 为解决样本不平衡问题而被提出的Focal Loss，通过引入类权重与对减少易分类样本的损失贡献同时增加难分类样本的损失贡献来减少样本不平衡问题
            + Focal Loss是一种变种的交叉熵损失函数,存在一阶、二阶导数，因此满足lgbm、xgb的要求
            + 对于样本不平衡问题,相较于传统样本处理方法(欠抽样、过抽样、smote、加权...)，Focal Loss提供了新的思路(由损失函数的构造方向解决样本不平衡问题)
            + Focal Loss中包含了类权重(alpha)，范围为(0,1),因此本模块不支持额外得再设定类权重、样本权重
            + Focal Loss中的gamma[0,+inf)值能减少易分类样本的损失贡献同时增加难分类样本的损失贡献。原始论文中gamma=2时效果最好            
                         
        参考资料:
        [FocalLoss论文:Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
        [Lightgbm中Focal Loss损失函数的应用-scipy求导](https://github.com/jrzaurin/LightGBM-with-Focal-Loss)
        [Lightgbm中Focal Loss损失函数的应用-手工求导](https://maxhalford.github.io/blog/lightgbm-focal-loss/)
        [Lightgbm中自定义损失函数下预测边际值无法复现的问题](https://github.com/microsoft/LightGBM/issues/3312)
        
        Parameters:
        --
            para_space:dict,lgb的参数空间
            gamma:float [0,+inf),Focal loss的gamma值,设定gamma=0,alpha=None时传统交叉熵损失函数
            alpha:float (0,1),Focal loss的alpha值,设定gamma=0,alpha=None时传统交叉熵损失函数
            n_iter:贝叶斯优化搜索迭代次数
            init_points:int,贝叶斯优化起始搜索点的个数
            scoring=‘negfocalloss’:str,寻优准则,可选'auc','ks','lift','neglogloss','negfocalloss'或其他可被sklearn识别的寻优准则
            early_stopping_rounds=10,int,训练数据中validation_fraction比例的数据被作为验证数据进行early_stopping,
            validation_fraction=0.1,float,进行early_stopping的验证集比例
            eval_metric=‘negfocalloss’,early_stopping的评价指标,为可被Estimator识别的格式,默认"negfocalloss"
            cv:int,RepeatedStratifiedKFold交叉验证的折数
            repeats:int,RepeatedStratifiedKFold交叉验证重复次数
            n_jobs,int,运行交叉验证和提升算法并行数,默认-1
            verbose,int,并行信息输出等级
            random_state,随机种子
            calibration:使用sklearn的CalibratedClassifierCV对refit模型进行概率校准
            cv_calibration:CalibratedClassifierCV的交叉验证数
            
            """参数空间写法     
                        
                #LGBMClassifier
                from skopt.utils import Categorical,Real,Integer
        
                para_space = {
                    'boosting_type':Categorical(['gbdt','goss']),
                    'n_estimators': Integer(30, 120),
                    'max_depth':Integer(2, 4),
                    'learning_rate': Real(0.05, 0.2,prior='uniform'),
                
                    'min_split_gain': Real(0, 0.02,prior='uniform'),
                    'min_child_samples':Integer(1, 100,prior='uniform'),
                    
                    'subsample':Real(0.5,1.0,prior='uniform'),
                    'colsample_bytree':Real(0.5,1.0,prior='uniform'),
                    'reg_lambda':Real(0,0.01,prior='uniform'),    
                }                                  
              """   
              
        Attribute:    
        --
            params_best:最优参数组合,需先使用fit
            model_refit:最优参数下的lgbm模型,需先使用fit
        
        Examples
        --        
        
        '''
    
        def __init__(self,para_space={},gamma=2,alpha=0.25,
                     n_iter=10,init_points=1,scoring='negfocalloss',eval_metric='negfocalloss',
                     cv=5,repeats=1,n_jobs=-1,verbose=0,
                     early_stopping_rounds=10,validation_fraction=0.1,random_state=123,calibration=False,cv_calibration=5):
        
            self.para_space=para_space
            self.gamma=gamma
            self.alpha=alpha
            self.n_iter=n_iter
            self.init_points=init_points
            self.scoring=scoring
            self.cv=cv
            self.repeats=repeats
            self.early_stopping_rounds=early_stopping_rounds
            self.eval_metric=eval_metric
            self.validation_fraction=validation_fraction
            self.n_jobs=n_jobs
            self.verbose=verbose
            self.random_state=random_state
            self.calibration=calibration
            self.cv_calibration=cv_calibration

            self._is_fitted=False
            
        def predict_proba(self,X,y=None):
            '''
            最优参数下的lgbm模型的预测
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

            out=X.apply(lambda col:col.astype('category') if col.name in self.cat_features else col) if self.cat_features else X

            return out  
        
        def fit(self,X,y,cat_features=None):
            '''
            进行贝叶斯优化
            Parameters:
            --
            X: pd.DataFrame对象
            y:目标变量,pd.Series对象
            cat_features:list,分类特征列名列表,None是数据中的object,category类列将被识别为cat_features，当Estimator为Xgboost时将忽略该参数        

            '''   

            self._check_data(X, y)

            self.cat_features=X.select_dtypes(['object','category']).columns.tolist() if cat_features is None else cat_features    
            
            self.Estimator=FLLGBMSklearn

            para_space=self._hpsearch_default(self.Estimator) if not self.para_space else self.para_space

            #if self.Estimator.__module__ == "catboost.core":

            #    X=X.apply(lambda col:col.astype('str') if col.name in self.cat_features else col) if self.cat_features else X

            #elif self.Estimator.__module__ == 'lightgbm.sklearn':

            X=X.apply(lambda col:col.astype('category') if col.name in self.cat_features else col) if self.cat_features else X

            #else:

            #    X=X 
            



            if self.early_stopping_rounds:

                X_tr, X_val, y_tr, y_val = train_test_split(X,y,test_size=self.validation_fraction,random_state=self.random_state,stratify=y)

                #BayesSearch_CV        
                self._BayesSearch_CV(para_space,X_tr,y_tr)

                #params_best
                self.params_best=self.bs_res.best_params_

                #cv_result
                self.cv_result=self._cvresult_to_df(self.bs_res.cv_results_)

                refit_Estimator=self.Estimator(objective=FocalLoss(alpha=self.alpha,gamma=self.gamma).lgb_obj,
                                               boost_from_average=False,
                                               random_state=self.random_state,
                                               n_jobs=effective_n_jobs(self.n_jobs),
                                               **self.params_best)

                self.model_refit = refit_Estimator.fit(**self._get_fit_params(self.Estimator,X_tr,y_tr,X_val,y_val))   

                self.params_best['best_iteration']=self.model_refit.best_iteration_ 


            else:

                #BayesSearch_CV        
                self._BayesSearch_CV(para_space,X,y)

                #params_best
                self.params_best=self.bs_res.best_params_

                #cv_result
                self.cv_result=self._cvresult_to_df(self.bs_res.cv_results_)

                #refit model
                # if self.Estimator.__module__ == "catboost.core":

                #     refit_Estimator=self.Estimator(random_state=self.random_state,
                #                                    thread_count=effective_n_jobs(self.n_jobs),
                #                                    **self.params_best)

                #     refit_Estimator.set_params(**{'cat_features':self.cat_features})

                # else:

                refit_Estimator=self.Estimator(objective=FocalLoss(alpha=self.alpha,gamma=self.gamma).lgb_obj,
                                               random_state=self.random_state,
                                               boost_from_average=False,
                                               n_jobs=effective_n_jobs(self.n_jobs),
                                               **self.params_best)

                self.model_refit = refit_Estimator.fit(X,y)             


            #get calibration done
            if self.calibration:

                self.model_refit=CalibratedClassifierCV(self.model_refit,cv=self.cv_calibration,
                                                      n_jobs=effective_n_jobs(self.n_jobs)).fit(X,y)

            self._is_fitted=True

            return self   
        
        def _BayesSearch_CV(self,para_space,X,y):        

            if self.scoring in ['ks','auc','lift','neglogloss']:

                scorer=self._get_scorer[self.scoring]

            elif self.scoring in ['negfocalloss']:

                scorer=metrics.make_scorer(FocalLoss(alpha=self.alpha,gamma=self.gamma).lgb_scorer,greater_is_better=True,needs_proba=True)
                
            else:
                
                scorer=self.scoring


            cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state)

            n_jobs=effective_n_jobs(self.n_jobs)

            # if self.Estimator.__module__ == "catboost.core":

            #     bs=BayesSearchCV(
            #         self.Estimator(random_state=self.random_state,thread_count=n_jobs),para_space,cv=cv,
            #         n_iter=self.n_iter,n_points=self.init_points,
            #         n_jobs=-1 if self.n_jobs==-1 else 1,
            #         verbose=self.verbose,refit=False,random_state=self.random_state,
            #         scoring=scorer,error_score=0)    

            #     self.bs_res=bs.fit(X,y,**{'cat_features':self.cat_features})

            #else:

            bs=BayesSearchCV(
                self.Estimator(objective=FocalLoss(alpha=self.alpha,gamma=self.gamma).lgb_obj,
                               boost_from_average=False,random_state=self.random_state,n_jobs=n_jobs),para_space,cv=cv,
                n_iter=self.n_iter,n_points=self.init_points,
                n_jobs=-1 if self.n_jobs==-1 else 1,
                verbose=self.verbose,refit=False,random_state=self.random_state,
                scoring=scorer,error_score=0)       

            self.bs_res=bs.fit(X,y)


            return(self)     
    
    
        def _get_fit_params(self,Estimator,X_train,y_train,X_val,y_val):


            # if Estimator.__module__ == "xgboost.sklearn":

            #     fit_params = {
            #         "X":X_train,
            #         "y":y_train,
            #         "eval_set": [(X_val, y_val)],
            #         "eval_metric": self.eval_metric,
            #         "early_stopping_rounds": self.early_stopping_rounds,
            #     }

            #     if sample_weight is not None:

            #         fit_params["sample_weight"] = sample_weight.loc[train_index]
            #         fit_params["sample_weight_eval_set"] = [sample_weight.loc[val_index]]


            #if Estimator.__module__ == 'lightgbm.sklearn':
                
            if self.eval_metric=="negfocalloss":
                
                self.eval_metric=FocalLoss(alpha=self.alpha,gamma=self.gamma).lgb_eval
                

            from lightgbm import early_stopping, log_evaluation

            fit_params = {
                "X":X_train,
                "y":y_train,
                "eval_set": [(X_val, y_val)],
                "eval_metric": self.eval_metric,
                "callbacks": [early_stopping(self.early_stopping_rounds)],
            }

            if self.verbose >= 100:

                fit_params["callbacks"].append(log_evaluation(1))

            else:

                fit_params["callbacks"].append(log_evaluation(0))

            # if sample_weight is not None:

            #     fit_params["sample_weight"] = sample_weight.loc[train_index]
            #     fit_params["eval_sample_weight"] = [sample_weight.loc[val_index]]


            # elif Estimator.__module__ == "catboost.core":

            #     from catboost import Pool

            #     fit_params = {
            #         "X": Pool(X_train, y_train, cat_features=self.cat_features),
            #         "eval_set": Pool(X_val, y_val, cat_features=self.cat_features),
            #         "early_stopping_rounds": self.early_stopping_rounds,
            #         # Evaluation metric should be passed during initialization
            #     }

            #     if sample_weight is not None:
            #         fit_params["X"].set_weight(sample_weight.loc[train_index])
            #         fit_params["eval_set"].set_weight(sample_weight.loc[val_index])

            # else:

            #     raise ValueError('Estimator in (LGBMClassifier)')


            return fit_params
    
    
        @staticmethod
        def _hpsearch_default(Estimator):

            from skopt.utils import Categorical,Real,Integer

            # if Estimator.__module__ == "xgboost.sklearn":

            #     para_space = {
            #         'n_estimators': Integer(30, 120),
            #         'max_depth':Integer(2, 4),
            #         'learning_rate': Real(0.05, 0.2,prior='uniform'),

            #         'gamma': Real(0,10,prior='uniform'),
            #         'subsample':Real(0.5,1.0,prior='uniform'),
            #         'colsample_bytree':Real(0.5,1.0,prior='uniform'),
            #         'reg_lambda':Real(0,10,prior='uniform'),

            #         'use_label_encoder':Categorical([False])            
            #     } 

            # elif Estimator.__module__ == 'lightgbm.sklearn':

            para_space = {

                'verbose':Categorical([-1]),
                'boosting_type':Categorical(['gbdt','goss']),
                'n_estimators': Integer(30, 120),
                'max_depth':Integer(2, 4),
                'learning_rate': Real(0.05, 0.2,prior='uniform'),

                'min_split_gain': Real(0, 0.02,prior='uniform'),
                'min_child_samples':Integer(1, 100,prior='uniform'),

                'subsample':Real(0.5,1.0,prior='uniform'),
                'colsample_bytree':Real(0.5,1.0,prior='uniform'),
                'reg_lambda':Real(0,0.01,prior='uniform'),    
            } 

            # elif Estimator.__module__ == "catboost.core":

            #     para_space = {

            #         'n_estimators': Integer(30, 120),
            #         'learning_rate': Real(0.05, 0.2,prior='uniform'),

            #         'max_depth':Integer(2, 4),
            #         'min_child_samples':Integer(1, 100,prior='uniform'),

            #         'subsample':Real(0.5,1.0,prior='uniform'),
            #         'colsample_bylevel':Real(0.5,1.0,prior='uniform'),
            #         'reg_lambda':Real(0,10,prior='uniform')               
            #     }  

            # else:

            #     raise ValueError('Estimator in (XGBClassifier,LGBMClassifier,CatBoostClassifier)')

            return para_space     
    
    
    
    