#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:03:34 2022

@author: zengke
"""

import pandas as pd
from joblib import effective_n_jobs
from BDMLtools.base import Base
from BDMLtools.tuner.base import BaseTunner
from sklearn.model_selection import RepeatedStratifiedKFold
from probatus.feature_elimination import EarlyStoppingShapRFECV,ShapRFECV
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
from BDMLtools.tuner.base import sLGBMClassifier
from BDMLtools.selector.bin_fun import R_pretty
import warnings
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.base import TransformerMixin
import numpy as np


class LgbmPISelector(TransformerMixin,Base,BaseTunner):
    
    '''
    使用基于LightGBM的Permutation importance进行特征筛选
    
    Permutation importance介绍:https://scikit-learn.org/stable/modules/permutation_importance.html
    
    Permutation importance源码:https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/inspection/_permutation_importance.py

    Parameters:
    --
        threshold:float,默认0,保留PI高于此值的特征
        method:str,基模型是否进行超参优化，可选"raw","bs",
            - “raw”,基模型不进行超参优化，使用预定超参拟合
            - “bs”，基模型进行贝叶斯超参优化(sklearn-optimize),这将会增加计算量但会得到最优超参条件下较小偏差的筛选结果            
        clf_params:dict,默认{}(代表默认参数)
            当method="raw"时为LGBMClassifier的超参数设置,{}代表默认参数            
            当method="bs"时为LGBMClassifier的超参数设置,LgbmPISelector._lgbm_hpsearch_default可查看默认超参设置  
        pi_repeats:int,默认10,PI中特征的随机排序次数,次数越多结果越有统计意义但会增加计算量
        n_iters:method='bs'时贝叶斯优化迭代次数
        init_points:method='bs'时贝叶斯优化起始搜索点个数
        scoring:str,method='bs'时超参优化目标和所有情况下的PI的评价指标,参考sklearn.metrics.SCORERS.keys(),
        early_stopping_rounds=10,int,训练数据中validation_fraction比例的数据被作为验证数据进行LightGBM的early_stopping,
                                注意使用early_stopping后boosting次数将比设定值小,可能出现迭代数量不足导致某些特征的PI值为0
        validation_fraction=0.1,float,进行early_stopping的验证集比例
        eval_metric=‘auc’,early_stopping的评价指标,为可被LightGBM识别的格式,参考LGBMClassifier.fit中的eval_metric参数
        cv:int,RepeatedStratifiedKFold交叉验证的折数
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        random_state:int,随机种子
        n_jobs,int,运行交叉验证时的joblib的并行数,默认-1
        verbose,int,并行信息输出等级
    
    Attribute:    
    --
        
        
    Method:    
    --        
        fit(X,y,categorical_feature=None,sample_weight=None):拟合模型，
        transform(X):对X进行特征筛选，返回筛选后的数据
        
    '''     

    def __init__(self,threshold=0,method='raw',clf_params={},pi_repeats=10,n_iter=5,init_points=5,scoring='roc_auc',
                 eval_metric='auc',validation_fraction=0.1,early_stopping_rounds=10,
                 cv=5,repeats=1,
                 random_state=123,n_jobs=-1,verbose=0):

        self.threshold=threshold
        self.method=method
        self.clf_params=clf_params
        self.pi_repeats=pi_repeats
        self.scoring=scoring
        self.n_iter=n_iter
        self.init_points=init_points
        self.scoring=scoring
        self.early_stopping_rounds=early_stopping_rounds
        self.validation_fraction=validation_fraction
        self.eval_metric=eval_metric
        self.cv=cv
        self.repeats=repeats
        self.random_state=random_state
        self.n_jobs=n_jobs
        self.verbose=verbose
        
        self._is_fitted=False
        
    def transform(self,X,y=None):
        
        """
        
        Parameters:
        --
            X:pd.DataFrame,训练数据X
            threshold:float or None,保留PI高于此值的特征,注意None时将self.threshold
        """            
        self._check_X(X)
        self._check_is_fitted()
               
        return X[self.keep_col]
            
    
    def fit(self,X,y,cat_features=None,sample_weight=None):
        
        """
        
        Parameters:
        --
            X:pd.DataFrame,训练数据X
            y:pd.Series,训练数据y
            categorical_feature:list,分类列列名,默认None即使用默认“object”或”category“列作为分类特征列名
            sample_weight:list,样本权重,默认None

        """                

        self._check_data(X, y)
        
        cat_features=X.select_dtypes(['object','category']).columns.tolist() if cat_features is None else cat_features
        
        X=X.apply(lambda col:col.astype('category') if col.name in cat_features else col) if cat_features else X  

        
        if self.method=='raw':
                        
            if self.early_stopping_rounds:
                
                X_tr, X_val, y_tr, y_val = train_test_split(X,y,test_size=self.validation_fraction,random_state=self.random_state,stratify=y)
                
                estimator=sLGBMClassifier(random_state=self.random_state,
                                          n_jobs=effective_n_jobs(self.n_jobs),
                                          **self.clf_params).fit(
                    X_tr,y_tr,**self._get_fit_params(X_val,y_val,sample_weight,y.index,y_val.index)
                )
                
            else:
                
                estimator = sLGBMClassifier(random_state=self.random_state,
                                            n_jobs=effective_n_jobs(self.n_jobs),
                                            **self.clf_params).fit(
                    X,y,**{'sample_weight':sample_weight})  
                                                         
        else:
            
            para_space=self.clf_params if self.clf_params else self._lgbm_hpsearch_default()
            
            if self.early_stopping_rounds:
                
                X_tr, X_val, y_tr, y_val = train_test_split(X,y,test_size=self.validation_fraction,random_state=self.random_state,stratify=y)

                self._BayesSearch_CV(para_space,X_tr,y_tr,None if sample_weight is None else sample_weight[y_tr.index])
    
                estimator=sLGBMClassifier(random_state=self.random_state,
                                          n_jobs=effective_n_jobs(self.n_jobs),
                                          **self.bs_res.best_params_).fit(
                    X_tr,y_tr,**self._get_fit_params(X_val,y_val,sample_weight,y.index,y_val.index)
                )  
            
            else:
                
                self._BayesSearch_CV(para_space,X,y,sample_weight)
                
                estimator = sLGBMClassifier(random_state=self.random_state,
                                            n_jobs=effective_n_jobs(self.n_jobs),
                                            **self.bs_res.best_params_).fit(
                    X,y,**{'sample_weight':sample_weight})  
                
         
        self.pi = permutation_importance(estimator, X, y, n_repeats=self.pi_repeats,random_state=self.random_state,
                                         scoring=self.scoring,n_jobs=effective_n_jobs(self.n_jobs))
        
        self.pi_df = pd.concat([pd.Series(self.pi[key],name=key,index=X.columns) for key in self.pi if key!='importances'],axis=1)

        self.keep_col=self.pi_df[self.pi_df['importances_mean']>self.threshold].index.to_list()
 
        self._is_fitted=True
            
        return self
    
    
    def _get_fit_params(self,X_val,y_val,sample_weight=None,train_index=None,val_index=None):
        
        from lightgbm import early_stopping, log_evaluation

        fit_params = {
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
            
        return fit_params
    
    
    def _BayesSearch_CV(self,para_space,X,y,sample_weight):        
               
        if self.scoring in ['ks','auc','lift','neglogloss']:
            
            scorer=self._get_scorer[self.scoring]
            
        else:
            
            scorer=self.scoring             
                    
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state)
        
        n_jobs=effective_n_jobs(self.n_jobs)
                        
        bs=BayesSearchCV(
            sLGBMClassifier(random_state=self.random_state,n_jobs=n_jobs),para_space,cv=cv,
            n_iter=self.n_iter,n_points=self.init_points,
            n_jobs=-1 if self.n_jobs==-1 else 1,verbose=self.verbose,refit=False,random_state=self.random_state,
            scoring=scorer,error_score=0)       
        
        self.bs_res=bs.fit(X,y,**{'sample_weight':sample_weight})
            
        return(self)      
    
    @staticmethod
    def _lgbm_hpsearch_default():

        from skopt.utils import Categorical,Real,Integer

        para_space={
            'boosting_type':Categorical(['goss','gbdt']),
            'verbose':Categorical([-1]), 
            'n_estimators': Integer(low=200, high=300, prior='uniform', transform='identity'),
            'learning_rate': Real(low=0.05, high=0.2, prior='uniform', transform='identity'),
            'max_depth': Integer(low=2, high=4, prior='uniform', transform='identity')            
        }
        
        return para_space
    
    
 
class LgbmShapRFECVSelector(TransformerMixin,Base,BaseTunner):
    
    '''
    使用LightGBM进行基于交叉验证的SHAP重要性的递归式特征消除(Recursive feature elimination with CV and Shap-value)
    
    Sklearn的RFECV算法介绍:
    https://scikit-learn.org/stable/modules/feature_selection.html#rfe  
    
    probatus的EarlyStoppingShapRFECV算法介绍:
    https://medium.com/ing-blog/open-sourcing-shaprfecv-improved-feature-selection-powered-by-shap-994fe7861560
    
    probatus的EarlyStoppingShapRFECV源码:
    https://github.com/ing-bank/probatus/blob/main/probatus/feature_elimination/feature_elimination.py
    
    可支持设定sample_weight,categorical_feature,early_stopping等
    
    Parameters:
    --
        step:int,float(0-1),RFE中每次消除特征的个数(int)/百分比(float)
        min_features_to_select:int,最少选择的特征个数
        method:str,可选"raw","bs",
            - “raw”时,每一步消除时基模型超参设定不变,这样能够加快消除过程但得到的筛选结果并非在最优超参条件下,因此结果会产生偏差
            - “bs”时，每一步消除均进行贝叶斯超参优化(sklearn-optimize),这将会增加计算量但会得到最优超参条件下较小偏差的筛选结果            
        clf_params:dict,默认{}(代表默认参数)
            当method="raw"时为LGBMClassifier的超参数设置,{}代表默认参数            
            当method="bs"时为LGBMClassifier的超参数设置,使用LgbmShapRFECVSelector._para_space_default可查看默认超参设置            
        n_iters:method='bs'时贝叶斯优化迭代次数
        n_points:method='bs'时贝叶斯优化起始搜索点个数
        scoring:str,method='bs'时超参优化目标或plot时的y轴指标,参考sklearn.metrics.SCORERS.keys(),
        cv:int,RepeatedStratifiedKFold交叉验证的折数
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        random_state:int,随机种子
        n_jobs,int,运行交叉验证时的joblib的并行数,默认-1
        verbose,int,并行信息输出等级
    
    Attribute:    
    --
        clf:RFECV对象
        
    Method:    
    --        
        fit(X,y,categorical_feature=None,sample_weight=None,check_additivity):拟合模型，
        transform(X):对X进行特征筛选，返回筛选后的数据
        
    '''     

    def __init__(self,step=1,min_features_to_select=1,method='raw',clf_params={},
                 n_iter=5,n_points=5,
                 scoring='roc_auc',
                 cv=5,repeats=1,
                 early_stopping_rounds=None,
                 random_state=123,n_jobs=-1,verbose=0):
        
        self.step=step
        self.min_features_to_select=min_features_to_select
        self.method=method
        self.clf_params=clf_params
        self.scoring=scoring
        self.n_iter=n_iter
        self.n_points=n_points
        self.cv=cv
        self.repeats=repeats
        self.early_stopping_rounds=early_stopping_rounds
        self.random_state=random_state
        self.n_jobs=n_jobs
        self.verbose=verbose
        
        self._is_fitted=False
        
    def transform(self,X,y=None,features_to_select=None):
        
        """
        
        Parameters:
        --
            X:pd.DataFrame,训练数据X
            features_to_select:int,选择的特征个数
        """            
        self._check_X(X)
        self._check_is_fitted()
        
        if features_to_select:
            
            keep_cols=self.clf.get_reduced_features_set(features_to_select)
            
        else:
            
            keep_cols=self.clf.get_reduced_features_set(self.min_features_to_select)
            
        return X[keep_cols]
    
        
    def fit(self,X,y,cat_features=None,sample_weight=None,check_additivity=True):
    
        if not self.verbose:
    
            with warnings.catch_warnings():   

                warnings.filterwarnings("ignore")

                return self._fit(X,y,cat_features=None,sample_weight=None,check_additivity=check_additivity)
            
        else:
            
            return self._fit(X,y,cat_features=None,sample_weight=None,check_additivity=check_additivity)

    
    
    def _fit(self,X,y,cat_features=None,sample_weight=None,check_additivity=True):
        
        """
        Parameters:
        --
            X:pd.DataFrame,训练数据X
            y:pd.Series,训练数据y
            categorical_feature:list,分类列列名,默认None即使用默认“object”或”category“列作为分类特征列名
            sample_weight:list,样本权重,默认None
            check_additivity:bool,进行shap-value的可加性校验,默认True

        """
        self._check_data(X, y)
        
        X=X.copy()
        y=y.copy()
        
        X=X.apply(lambda col:col.astype('category') if col.name in cat_features else col) if cat_features else X       
    
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state)       
        
        n_jobs=effective_n_jobs(self.n_jobs)

        if self.method=='raw':
            
            estimator=sLGBMClassifier(random_state=self.random_state,
                                      n_jobs=effective_n_jobs(self.n_jobs),
                                      **self.clf_params)
            
        else:
            
            para_space=self.clf_params if self.clf_params else self._lgbm_hpsearch_default()
    
            
            estimator=BayesSearchCV(sLGBMClassifier(random_state=self.random_state,n_jobs=n_jobs),para_space,
                                    n_iter=self.n_iter,n_points=self.n_points,
                                    random_state=self.random_state,
                                    n_jobs=-1 if self.n_jobs==-1 else 1,
                                    verbose=self.verbose,cv=cv)
        
        if self.early_stopping_rounds:
        
        
            self.clf=EarlyStoppingShapRFECV(estimator,
                                        min_features_to_select=self.min_features_to_select,
                                        step=self.step,
                                        random_state=self.random_state,
                                        scoring=self.scoring,
                                        n_jobs=-1 if self.n_jobs==-1 else 1,
                                        verbose=-1,
                                        early_stopping_rounds=self.early_stopping_rounds,
                                        cv=cv)
        
            self.clf.fit_compute(X,y,check_additivity=check_additivity,sample_weight=sample_weight)
            
        else:

            self.clf=ShapRFECV(estimator,
                                min_features_to_select=self.min_features_to_select,
                                step=self.step,
                                random_state=self.random_state,
                                scoring=self.scoring,
                                n_jobs=-1 if self.n_jobs==-1 else 1,
                                verbose=-1,
                                cv=cv)
        
            self.clf.fit_compute(X,y,check_additivity=check_additivity,sample_weight=sample_weight)
            
        
        self._is_fitted=True
    
        return self
    
    @staticmethod
    def _lgbm_hpsearch_default():

        from skopt.utils import Categorical,Real,Integer

        para_space={
            'verbose':Categorical([-1]), 
            'boosting_type':Categorical(['goss','gbdt']),
            'n_estimators': Integer(low=30, high=300, prior='uniform', transform='identity'),
            'learning_rate': Real(low=0.05, high=0.2, prior='uniform', transform='identity'),
            'max_depth': Integer(low=1, high=4, prior='uniform', transform='identity')
        }
        
        return para_space
    
    def plot(self,figure_size=(10,5)):
    
        self._check_is_fitted()
        
        from plotnine import ggplot,theme,theme_bw,ggtitle,labs,geom_errorbar,aes,geom_line,geom_point,scale_x_reverse
        
        dt=self.clf.report_df.copy()
        
        dt['val_metric_std_error']=dt['val_metric_std'].div(np.sqrt(self.cv*self.repeats-1))
    
        title="SHAP RFE using CV and Lightgbm"
        
        if dt['num_features'].size>50 and self.step==1:
            
            n=50
        
        else:
            
            n=dt['num_features'].size
            
        breaks=R_pretty(dt['num_features'].min(),dt['num_features'].max(),n)
        
        p=(ggplot(dt,aes(x='num_features', y='val_metric_mean'))+
            geom_point(color='red',size=3)+
            geom_line()+
            geom_errorbar(aes(ymin=dt['val_metric_mean']-dt['val_metric_std_error'],ymax=dt['val_metric_mean']+dt['val_metric_std_error']),width=.2)+
            ggtitle(title)+
            labs(x = "Num of features", y = "Val-Score") +
            theme_bw() +
            theme(figure_size=figure_size) +
            scale_x_reverse(breaks=breaks)
        )
        
        return p  
    

class LgbmSeqSelector(TransformerMixin,Base,BaseTunner):
    
    '''
    使用LightGBM进行基于交叉验证的序列特征消除(Sequential Feature Selection with CV)
    
    Sequential Feature Selection算法介绍:
    http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector


    缺陷:
    + 目前不支持early_stopping
    + 分类特征将不参与序列法筛选
    + 目前不支持在特征筛选过程中通过交叉验证中进行基模型的超参优化

    Parameters:
    --
        forward:bool,默认False,当floating=False时
            + True:前向序列特征选择(Sequential Forward Selection)
            + False:后向序列特征选择(Sequential Backward Selection)
        floating:bool,默认False,代表使用基础的SFS或SBS
            + 当为True时,若forward=True则执行SFFS(Sequential Forward Floating Selection)
            + 当为True时,若forward=False则执行SBFS(Sequential Backward Floating Selection)           
        k_features:int,选择的特征个数
        clf_params:dict,LGBMClassifier的超参数设置,{}代表默认参数
        scoring:str,寻优准则,可选'auc','ks','lift','neglogloss'
        fixed_features:list or None
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

    def __init__(self,forward=False,floating=False,k_features=5,clf_params={'verbose':-1},scoring='roc_auc',
                 cv=5,repeats=1,fixed_features=None,
                 random_state=123,n_jobs=-1,verbose=0):
        
        self.forward=forward
        self.floating=floating
        self.k_features=k_features
        self.clf_params=clf_params
        self.scoring=scoring
        self.fixed_features=fixed_features
        self.cv=cv
        self.repeats=repeats
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

        return X[self.keep_cols+self.cat_features]
            
    
    def fit(self,X,y,sample_weight=None):
        
        """
        
        Parameters:
        --
            X:pd.DataFrame,训练数据X
            y:pd.Series,训练数据y

        """
               

        self._check_data(X, y)
        
        self.cat_features=X.select_dtypes(include=['category','object']).columns.tolist()
        
        X=X.select_dtypes('number')
     
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state)       
        
        n_jobs=effective_n_jobs(self.n_jobs)
        
        lgbm=sLGBMClassifier(random_state=self.random_state,n_jobs=n_jobs,**self.clf_params)
        
        seq_clf = SequentialFeatureSelector(lgbm,                                            
                        k_features=self.k_features,
                        forward=self.forward,
                        floating=self.floating,
                        verbose=self.verbose,  
                        scoring=self.scoring,
                        cv=cv,  
                        fixed_features=self.fixed_features,                    
                        n_jobs=n_jobs)
        
        seq_clf.fit(X, y, **{'sample_weight':sample_weight})
        
        self.keep_cols=list(seq_clf.k_feature_names_)
        
        self.clf=seq_clf
             
        self._is_fitted=True
    
        return self
    
    def plot(self,figure_size=(10,5)):
        
        self._check_is_fitted()
        
        from plotnine import ggplot,theme,theme_bw,ggtitle,labs,geom_errorbar,aes,geom_line,geom_point,scale_x_continuous,scale_x_reverse
        
        dt=pd.DataFrame(self.clf.get_metric_dict()).T.apply(lambda x:pd.to_numeric(x,errors='ignore'))
        dt['num_features']=dt.index
        
        
        title="Sequential Feature Selection using CV and Lightgbm"
        
        breaks=R_pretty(dt['num_features'].min(),dt['num_features'].max(),50 if dt['num_features'].size>50 else dt['num_features'].size)                
        
        p=(ggplot(dt,aes(x='num_features', y='avg_score'))+
            geom_point(color='red',size=3)+
            geom_line()+
            geom_errorbar(aes(ymin=dt['avg_score']-dt['std_err'],ymax=dt['avg_score']+dt['std_err']),width=.2)+
            ggtitle(title)+
            labs(x = "Num of features", y = "Val-Score") +
            theme_bw() +
            theme(figure_size=figure_size)     
        )
        
        if self.forward:
            
            p = p + scale_x_continuous(breaks=breaks)
            
        else:
            
            p = p + scale_x_reverse(breaks=breaks) 
        
        return p        