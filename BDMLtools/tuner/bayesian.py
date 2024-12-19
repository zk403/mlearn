#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:14:47 2024

@author: zengke
"""

from BDMLtools.base import Base
from BDMLtools.tuner.base import BaseTunner,FLLGBMSklearn,FocalLoss
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import early_stopping,log_evaluation
import pandas as pd
from joblib import effective_n_jobs,Parallel,delayed        
from sklearn.metrics import roc_auc_score,log_loss
import numpy as np
import optuna


class BayesianCVTuner(Base,BaseTunner):
    
    '''
    使用optuna进行贝叶斯超参优化
    
    optuna文档:https://optuna.readthedocs.io/en/stable/
    optuna源码:https://github.com/optuna/optuna
    
    Parameters:
    --
        Estimator:拟合器,XGBClassifier、LGBMClassifier或CatBoostClassifier
        para_space:dict,lgb的参数空间
        n_trials:贝叶斯优化搜索迭代次数
        init_points:int,贝叶斯优化起始搜索点的个数
        scoring:str,寻优准则,可选'auc','logloss',同sklearn.metrics中的roc_auc_score与log_loss
        early_stopping_rounds=10,int,交叉验证时早停轮数
        eval_metric=‘auc’,early_stopping的评价指标,为可被Estimator识别的格式,参考Estimator.fit中的eval_metric参数
        cv:int,RepeatedStratifiedKFold交叉验证的折数
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        n_jobs,int,运行交叉验证和提升算法并行数,默认-1
        verbose,int,并行信息输出等级
        random_state,随机种子
        
        """参数空间写法 
        #XGBClassifier
        param="""{"n_estimators":trial.suggest_int('n_estimators', 30, 120),
            "verbosity": trial.suggest_int("verbosity",0,0),
            "objective": "binary:logistic",
            "max_depth":trial.suggest_int("max_depth", 1, 3, step=1),
            "min_child_weight":trial.suggest_int("min_child_weight", 2, 10),
            "eta":trial.suggest_float("eta", 0.01, 0.3, log=False),
            "gamma":trial.suggest_float("gamma", 0, 1.0, log=False),
            "grow_policy":trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            }"""    
    
        #LGBMClassifier
        param= """{'n_estimators':trial.suggest_int('n_estimators', 30, 120),
                'boosting_type':trial.suggest_categorical("boosting_type", ["gbdt", "goss"]),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=False),
                'max_depth': trial.suggest_int('max_depth', 1, 3),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 100, log=False),                 
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0, log=False),
                'verbosity': trial.suggest_int("verbosity",-1,-1)
                 }"""   
                    
        #Catboost
        param= """{"iterations": trial.suggest_int("iterations",80,120),            
           "depth": trial.suggest_int("depth", 1, 3),
           "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=False),            
           "subsample": trial.suggest_float("subsample", 0.05, 1.0),
           "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
           "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
           "reg_lambda":trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
           "silent":trial.suggest_categorical("silent",[True]),
           }""" 
      """   
        
            
        """   
          
    Attribute:    
    --
        params_best:最优参数组合,需先使用fit
        model_refit:最优参数下的lgbm模型,需先使用fit
    
    Examples
    --

    '''    
    
    def __init__(self,Estimator,para_space=None,n_trials=10,scoring='auc',eval_metric='auc',
                 cv=5,repeats=1,n_jobs=-1,verbose=0,early_stopping_rounds=10,random_state=123):
        
        self.Estimator=Estimator
        self.para_space=para_space
        self.n_trials=n_trials
        self.scoring=scoring
        self.cv=cv
        self.repeats=repeats
        self.early_stopping_rounds=early_stopping_rounds
        self.eval_metric=eval_metric
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.random_state=random_state
        
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
        
        if self.Estimator.__module__ == "catboost.core":
        
            self.cat_features=X.select_dtypes(['object','category']).columns.tolist() if cat_features is None else cat_features    
            
            self.cat_features_idx=np.argwhere(np.isin(X.columns,self.cat_features)).reshape(len(self.cat_features),) if len(self.cat_features) else None

                
            
        n_jobs=effective_n_jobs(self.n_jobs)       
            
        func = lambda trial: self.objective(trial,self.Estimator,X=np.array(X),y=np.array(y),param=self.para_space,
                                            sample_weight=sample_weight,metrics=self.scoring,n_jobs=n_jobs)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.random_state) #reproduce result using seed
        study = optuna.create_study(direction='maximize',sampler=sampler)
        study.optimize(func, n_trials=self.n_trials) 
        
        self.cv_result=pd.DataFrame([i.params for i in study.trials]).join(
            pd.DataFrame(
                [i.value for i in study.trials],columns=['score_mean_cv']
            )
        )
        
        self.params_best=study.best_trial.params
        
        if self.Estimator.__module__ == "catboost.core":
            
            self.model_refit = self.Estimator(**study.best_trial.params,random_state=self.random_state).fit(X,y,sample_weight=sample_weight,cat_features=self.cat_features)   
            
        else:    
            
            self.model_refit = self.Estimator(**study.best_trial.params,random_state=self.random_state).fit(X,y,sample_weight=sample_weight)   
        
        self._is_fitted=True
        
        return self
    
    
    def objective(self,trial,Estimator,X,y,param=None,sample_weight=None,metrics='auc',n_jobs=1):
        
    
        if param is None:
    
            param=eval(self._hpsearch_default(Estimator))
    
        else:
            
            param=eval(param)
        
        def _cv_parallel(X,y,tr_idx,val_idx,ws):    
    
            if ws is None:
        
                ws=np.ones(len(X))
    
            else:
                
                ws=np.array(ws)
    
            if self.early_stopping_rounds:
    
                if Estimator.__module__ == "catboost.core":
                    
                    model=Estimator(**param,random_state=self.random_state,thread_count=1 if self.n_jobs>=0 else -1)
                    
                    if self.eval_metric=='auc':                       
                        self.eval_metric='AUC'
                    elif self.eval_metric=='logloss':
                        self.eval_metric='Logloss'
                    
                    model.set_params(**{"eval_metric":self.eval_metric,"early_stopping_rounds":self.early_stopping_rounds})       
                    
                #xgboost:eval_metric and early_stopping_rounds in `fit` method is deprecated for better compatibility with scikit-learn,version>=1.6.0
                elif Estimator.__module__ == 'xgboost.sklearn':
                        
                    model=Estimator(**param,random_state=self.random_state,n_jobs=1 if self.n_jobs>=0 else -1)
                        
                    model.set_params(**{"eval_metric":self.eval_metric,'early_stopping_rounds':self.early_stopping_rounds})                                                                               
                                               
                else:
                    
                    model=Estimator(**param,random_state=self.random_state,n_jobs=1 if self.n_jobs>=0 else -1)
    
                model.fit(**self._get_fit_params(model,X,y,tr_idx,val_idx,ws)) 
    
            else:
                
                if Estimator.__module__ == "catboost.core":
                    
                    model=Estimator(**param,random_state=self.random_state,thread_count=1 if self.n_jobs>=0 else -1)
                    
                    model.set_params(**{'cat_features':self.cat_features_idx}).fit(X[tr_idx],y[tr_idx],sample_weight=ws[tr_idx])
                    
                else:
                    
                    model=Estimator(**param,random_state=self.random_state,n_jobs=1 if self.n_jobs>=0 else -1).fit(X[tr_idx],y[tr_idx],sample_weight=ws[tr_idx])            
            
            pred = model.predict_proba(X[val_idx])[:,1]    
    
            #sample_weight=ws[val_idx]
            if metrics=='logloss':
                
                score = - log_loss(y[val_idx],pred,sample_weight=ws[val_idx])
    
            elif metrics=='auc':
    
                score = roc_auc_score(y[val_idx],pred,sample_weight=ws[val_idx])
    
            else:
    
                raise ValueError("metrics in ('logloss','auc')")
            
            return score  
        
        # StratifiedKFold cross validation
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state)
        n_jobs=effective_n_jobs(n_jobs)
        parallel=Parallel(n_jobs=n_jobs,verbose=self.verbose,backend='threading')
        outlist=np.array(parallel(delayed(_cv_parallel)(X,y,tr_idx,val_idx,sample_weight)
                        for tr_idx,val_idx in cv.split(X, y)),dtype=object)
        
        return sum(outlist) / len(outlist)
    
       
    
    def _get_fit_params(self,Estimator,X,y,train_index,val_index,sample_weight):
            
        if Estimator.__module__ == "xgboost.sklearn":
            
            fit_params = {
                "X":X[train_index],
                "y":y[train_index],
                "eval_set": [(X[val_index], y[val_index])],
                "sample_weight":sample_weight[train_index],
                "sample_weight_eval_set":[sample_weight[val_index]],
                "verbose":0
            }
            
        elif Estimator.__module__ == 'lightgbm.sklearn':            
    
            fit_params = {
                "X":X[train_index],
                "y":y[train_index],
                "eval_set": [(X[val_index], y[val_index])],
                "eval_metric":self.eval_metric,#eval_metric,
                "callbacks": [early_stopping(self.early_stopping_rounds,first_metric_only=True,verbose=0),log_evaluation(0)],
                "sample_weight":sample_weight[train_index],
                "eval_sample_weight":[sample_weight[val_index]]
            }        
        
        elif Estimator.__module__ == "catboost.core":
            
            from catboost import Pool
            
            fit_params = {
                "X": Pool(X[train_index], y[train_index],cat_features=self.cat_features_idx).set_weight(sample_weight[train_index]), #cat_features=self.cat_features
                "eval_set": Pool(X[val_index], y[val_index],cat_features=self.cat_features_idx).set_weight(sample_weight[val_index]), #cat_features=self.cat_features    
                "verbose":False
            }
            
        else:
            
            raise ValueError('Estimator in (XGBClassifier,LGBMClassifier,CatBoostClassifier)')
        
            
        return fit_params
    
    
    @staticmethod
    def _hpsearch_default(Estimator):
        
        if Estimator.__module__ == "xgboost.sklearn":
            
            param="""{
                "n_estimators":trial.suggest_int('n_estimators', 30, 120),
                "verbosity": trial.suggest_int("verbosity",0,0),
                "objective": "binary:logistic",
                "max_depth":trial.suggest_int("max_depth", 1, 3, step=1),
                "min_child_weight":trial.suggest_int("min_child_weight", 2, 10),
                "eta":trial.suggest_float("eta", 0.01, 0.3, log=False),
                "gamma":trial.suggest_float("gamma", 0, 1.0, log=False),
                "grow_policy":trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            }"""
        
            
        elif Estimator.__module__ == 'lightgbm.sklearn':
    
            param= """{'n_estimators':trial.suggest_int('n_estimators', 30, 120),
                    'boosting_type':trial.suggest_categorical("boosting_type", ["gbdt", "goss"]),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=False),
                    'max_depth': trial.suggest_int('max_depth', 1, 3),
                    'min_child_samples': trial.suggest_int('min_child_samples', 1, 100, log=False),                 
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0, log=False),
                    'verbosity': trial.suggest_int("verbosity",-1,-1)
                     }"""        
            
        elif Estimator.__module__ == "catboost.core":
    
             param= """{"iterations": trial.suggest_int("iterations",80,120),            
                "depth": trial.suggest_int("depth", 1, 3),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=False),            
                "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                "reg_lambda":trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                "silent":trial.suggest_categorical("silent",[True]),
                }"""
      
        else:
            
            raise ValueError('Estimator in (XGBClassifier,LGBMClassifier,CatBoostClassifier)')
    
        return param
    
    


class FLBSTuner(Base,BaseTunner):
    
        '''
        使用二分类Focal Loss、贝叶斯超参优化(optuna)的梯度提升模型(目前仅支持Sklearn接口下的LightGBM)
        
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
            n_trials:optuna贝叶斯优化搜索迭代次数
            scoring=‘negfocalloss’:str,寻优准则,可选'auc','logloss','focalloss'
            early_stopping_rounds=10,int,训练数据中validation_fraction比例的数据被作为验证数据进行early_stopping,
            eval_metric=‘focalloss’,early_stopping的评价指标，可选auc','logloss','focalloss'
            cv:int,RepeatedStratifiedKFold交叉验证的折数
            repeats:int,RepeatedStratifiedKFold交叉验证重复次数
            n_jobs,int,运行交叉验证和提升算法并行数,默认-1
            verbose,int,并行信息输出等级
            random_state,随机种子
            
            """参数空间写法     
                        
              #LGBMClassifier
              param= """{'n_estimators':trial.suggest_int('n_estimators', 30, 120),
                    'boosting_type':trial.suggest_categorical("boosting_type", ["gbdt", "goss"]),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=False),
                    'max_depth': trial.suggest_int('max_depth', 1, 4),
                    'min_child_samples': trial.suggest_int('min_child_samples', 1, 100, log=False),     
                    'min_split_gain': trial.suggest_float('min_split_gain', 0, 0.02, log=False),                         
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0, log=False),
                    'verbosity': trial.suggest_int("verbosity",-1,-1)
                     }""" 
              
        Attribute:    
        --
            params_best:最优参数组合,需先使用fit
            model_refit:最优参数下的lgbm模型,需先使用fit
        
        Examples
        --        
        
        '''
    
        def __init__(self,para_space=None,gamma=2,alpha=0.25,
                     n_trials=10,scoring='focalloss',eval_metric='focalloss',
                     cv=5,repeats=1,n_jobs=-1,verbose=0,
                     early_stopping_rounds=10,random_state=123):
        
            self.para_space=para_space
            self.gamma=gamma
            self.alpha=alpha
            self.n_trials=n_trials
            self.scoring=scoring
            self.cv=cv
            self.repeats=repeats
            self.early_stopping_rounds=early_stopping_rounds
            self.eval_metric=eval_metric
            self.n_jobs=n_jobs
            self.verbose=verbose
            self.random_state=random_state

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
        
        def fit(self,X,y):
            '''
            进行贝叶斯优化
            Parameters:
            --
            X: pd.DataFrame对象
            y:目标变量,pd.Series对象    
            '''   

            self._check_data(X, y)
            
            self.Estimator=FLLGBMSklearn

            para_space=self._hpsearch_default() if not self.para_space else self.para_space
            
            n_jobs=effective_n_jobs(self.n_jobs)
            
                
            func = lambda trial: self.objective(trial,self.Estimator,X=np.array(X),y=np.array(y),param=para_space,
                                                sample_weight=None,metrics=self.scoring,n_jobs=n_jobs)
            
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler = optuna.samplers.TPESampler(seed=self.random_state) #reproduce result using seed
            study = optuna.create_study(direction='maximize',sampler=sampler)
            study.optimize(func, n_trials=self.n_trials) 
            
            self.cv_result=pd.DataFrame([i.params for i in study.trials]).join(
                pd.DataFrame(
                    [i.value for i in study.trials],columns=['score_mean_cv']
                )
            )
            
            self.params_best=study.best_trial.params

            self.model_refit = self.Estimator(**study.best_trial.params,random_state=self.random_state).fit(X,y,sample_weight=None)   
            
            self._is_fitted=True

            return self   

    
        def objective(self,trial,Estimator,X,y,param=None,sample_weight=None,metrics='auc',n_jobs=1):
            
        
            if param is None:
        
                param=eval(self._hpsearch_default(Estimator))
        
            else:
                
                param=eval(param)
            
            def _cv_parallel(X,y,tr_idx,val_idx,ws):    
        
                if ws is None:
            
                    ws=np.ones(len(X))
        
                else:
                    
                    ws=np.array(ws)
        
                if self.early_stopping_rounds:
     
                    model=Estimator(objective=FocalLoss(alpha=self.alpha,gamma=self.gamma).lgb_obj,
                                    boost_from_average=False,
                                    random_state=self.random_state,
                                    **param,
                                    n_jobs=1 if self.n_jobs>=0 else -1)
        
                    model.fit(**self._get_fit_params(X,y,tr_idx,val_idx,ws)) 
        
                else:
                                            
                    model=Estimator(**param,random_state=self.random_state,n_jobs=1 if self.n_jobs>=0 else -1).fit(X[tr_idx],y[tr_idx],sample_weight=ws[tr_idx])            
                
                pred = model.predict_proba(X[val_idx])[:,1]    
        
                #sample_weight=ws[val_idx]
                if metrics=='logloss':
                    
                    score = - log_loss(y[val_idx],pred,sample_weight=None)
        
                elif metrics=='auc':
        
                    score = roc_auc_score(y[val_idx],pred,sample_weight=None)
                    
                elif metrics=='focalloss':
                
                    score = FocalLoss(alpha=self.alpha,gamma=self.gamma).lgb_scorer(y[val_idx],pred)
        
                else:
        
                    raise ValueError("metrics in ('logloss','auc','focalloss')")
                
                return score  
            
            # StratifiedKFold cross validation
            cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=1)
            n_jobs=effective_n_jobs(n_jobs)
            parallel=Parallel(n_jobs=n_jobs,verbose=self.verbose,backend='threading')
            outlist=np.array(parallel(delayed(_cv_parallel)(X,y,tr_idx,val_idx,sample_weight)
                            for tr_idx,val_idx in cv.split(X, y)),dtype=object)
            
            return sum(outlist) / len(outlist)
        
        
        def _get_fit_params(self,X,y,train_index,val_index,sample_weight):

            if self.eval_metric=="focalloss":
                
                self.eval_metric=FocalLoss(alpha=self.alpha,gamma=self.gamma).lgb_eval
                
            fit_params = {
                "X":X[train_index],
                "y":y[train_index],
                "eval_set": [(X[val_index], y[val_index])],
                "eval_metric":self.eval_metric,#eval_metric,
                "callbacks": [early_stopping(self.early_stopping_rounds,first_metric_only=True,verbose=0),log_evaluation(0)],
                "sample_weight":sample_weight[train_index],
                "eval_sample_weight":[sample_weight[val_index]]
            }       
   
            return fit_params       
    
    
        @staticmethod
        def _hpsearch_default():
            
            param= """{'n_estimators':trial.suggest_int('n_estimators', 30, 120),
                    'boosting_type':trial.suggest_categorical("boosting_type", ["gbdt", "goss"]),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=False),
                    'max_depth': trial.suggest_int('max_depth', 1, 4),
                    'min_child_samples': trial.suggest_int('min_child_samples', 1, 100, log=False),     
                    'min_split_gain': trial.suggest_float('min_split_gain', 0, 0.02, log=False),                         
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0, log=False),
                    'verbosity': trial.suggest_int("verbosity",-1,-1)
                     }"""        
                        
            return param      