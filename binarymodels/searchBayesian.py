#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:26:03 2021

@author: zengke
"""
from sklearn.base import BaseEstimator
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
#from time import time
import numpy as np
import pandas as pd

class searchBayesianXGB(BaseEstimator):
    
    def __init__(self,para_space,n_iter=10,init_points=5,scoring='auc',cv=5):
        '''
        使用贝叶斯优化参数的Xgboost
        Parameters:
        --
            para_space:dict,xgboost的参数空间
            n_iter:贝叶斯优化搜索迭代次数
            init_points:int,贝叶斯优化起始搜索点的个数
            scoring:str,寻优准则,可选'auc','ks','lift'
            cv:int,交叉验证的折数
        
        Attribute:    
        --
            Optimize:贝叶斯优化迭代器,需先使用fit
            params_best:最优参数组合,需先使用fit
            xgb_refit:最优参数下的xgboost模型,需先使用fit
        
        Examples
        --
        
        

        '''        
        self.para_space=para_space
        self.n_iter=n_iter
        self.init_points=init_points
        self.scoring=scoring
        self.cv=cv

        
    def predict_proba(self,X,y=None):
        '''
        最优参数下的xgboost模型的预测
        Parameters:
        --
        X:pd.DataFrame对象
        '''      
        pred = self.xgb_refit.predict_proba(X)[:,1]        
        return pred
    
    def transform(self,X,y=None):     
        '''
        使用逐步回归进行特征筛选,返回逐步法筛选后的训练数据
        Parameters:
        --
        X:woe编码数据,pd.DataFrame对象,需与训练数据woe编码具有相同的特征
        '''        
        return self
          
    def fit(self,X,y):
        '''
        拟合逐步回归
        Parameters:
        --
        X:woe编码训练数据,pd.DataFrame对象
        y:目标变量,pd.Series对象
        '''   
        
        self.X=X
        self.y=y
        
        self.Optimize = BayesianOptimization(self.XGB_CV,self.para_space)
        self.Optimize.maximize(n_iter=self.n_iter,init_points=self.init_points)
        
        #输出最优参数组合
        self.params_best=self.Optimize.max['params']
        self.params_best['max_depth']=int(self.params_best['max_depth'])
        self.params_best['n_estimators']=int(self.params_best['n_estimators'])   
        
        #交叉验证结果保存
        self.cv_result=self.cvresult_to_df()
        
        self.xgb_refit = XGBClassifier(
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
        
        return self
    
    
    def XGB_CV(self,n_estimators,max_depth,gamma,learning_rate,min_child_weight,
               max_delta_step,subsample,colsample_bytree,reg_lambda,scale_pos_weight):
            
        para_space = {
                      'booster' : ['gbtree'],
                      'max_depth' : [int(max_depth)],
                      'gamma' : [gamma],
                      'scale_pos_weight':[scale_pos_weight],
                      'n_estimators':[int(n_estimators)],
                      'learning_rate' : [learning_rate],
                      #'n_jobs' : [4],
                      'subsample' : [max(min(subsample, 1), 0)],
                      'colsample_bytree' : [max(min(colsample_bytree, 1), 0)],
                      'min_child_weight' : [min_child_weight],
                      'max_delta_step' : [int(max_delta_step)],
                      'reg_lambda':[reg_lambda],
                      }       
        
        
        if self.scoring=='ks':
            scorer=metrics.make_scorer(self.custom_score_KS,greater_is_better=True,needs_proba=True)
        elif self.scoring=='auc':
            scorer=metrics.make_scorer(self.custom_score_AUC,greater_is_better=True,needs_proba=True)
        elif self.scoring=='lift':
            scorer=metrics.make_scorer(self.custom_score_Lift,greater_is_better=True,needs_proba=True)
        else:
            raise IOError('scoring not understood,should be "ks","auc","lift")')
                        
        cv_res=GridSearchCV(
            XGBClassifier(seed=123,use_label_encoder=False,verbosity=0),para_space,cv=self.cv,
            scoring=scorer,error_score=0).fit(self.X,self.y)
        
        print(cv_res.cv_results_['mean_test_score'])
        val_score = cv_res.cv_results_['mean_test_score'][0]
        print(' Stopped after %d iterations with val-%s = %f' % (n_estimators,self.scoring,val_score))
        return(val_score)    
    
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
    
    def cvresult_to_df(self):

        ParaDf_all=pd.DataFrame()
    
        for i in range(len(self.Optimize.res)):
            ParaDf=pd.DataFrame([self.Optimize.res[i]['params']])
            ParaDf['val_'+self.scoring]=self.Optimize.res[i]['target']    
            ParaDf_all=pd.concat([ParaDf,ParaDf_all],ignore_index=True)
        
        return ParaDf_all
    

class searchBayesianLGBM(BaseEstimator):
    
    def __init__(self,para_space,n_iter=10,init_points=5,scoring='auc',cv=5):
        '''
        使用贝叶斯优化参数的LightGBM
        Parameters:
        --
            para_space:dict,lgb的参数空间
            n_iter:贝叶斯优化搜索迭代次数
            init_points:int,贝叶斯优化起始搜索点的个数
            scoring:str,寻优准则,可选'auc','ks','lift'
            cv:int,交叉验证的折数
        
        Attribute:    
        --
            Optimize:贝叶斯优化迭代器,需先使用fit
            params_best:最优参数组合,需先使用fit
            xgb_refit:最优参数下的xgboost模型,需先使用fit
        
        Examples
        --
        
        

        '''        
        self.para_space=para_space
        self.n_iter=n_iter
        self.init_points=init_points
        self.scoring=scoring
        self.cv=cv

        
    def predict_proba(self,X,y=None):
        '''
        最优参数下的xgboost模型的预测
        Parameters:
        --
        X:pd.DataFrame对象
        '''      
        pred = self.lgbm_refit.predict_proba(X)[:,1]        
        return pred
    
    def transform(self,X,y=None):     
        '''
        使用逐步回归进行特征筛选,返回逐步法筛选后的训练数据
        Parameters:
        --
        X:woe编码数据,pd.DataFrame对象,需与训练数据woe编码具有相同的特征
        '''        
        return self
          
    def fit(self,X,y):
        '''
        拟合逐步回归
        Parameters:
        --
        X:woe编码训练数据,pd.DataFrame对象
        y:目标变量,pd.Series对象
        '''   
        
        self.X=X
        self.y=y
        
        para_space_num={key:self.para_space[key] for key in self.para_space if key not in ('boosting_type','class_weight')}  
        
        self.Optimize = BayesianOptimization(self.LGBM_CV,para_space_num)
        self.Optimize.maximize(n_iter=self.n_iter,init_points=self.init_points)
        
        #输出最优参数组合
        self.params_best=self.Optimize.max['params']
        self.params_best['max_depth']=int(self.params_best['max_depth'])
        self.params_best['n_estimators']=int(self.params_best['n_estimators'])   
        
        #交叉验证结果保存
        self.cv_result=self.cvresult_to_df()
        
        print (self.para_space)
        self.lgbm_refit = LGBMClassifier(
            boosting_type=self.para_space['boosting_type'],
            n_estimators=self.params_best['n_estimators'],
            learning_rate=self.params_best['learning_rate'],
            max_depth=self.params_best['max_depth'],          
            min_split_gain=self.params_best['min_split_gain'],
            min_sum_hessian_in_leaf=self.params_best['min_sum_hessian_in_leaf'],
            subsample=self.params_best['subsample'],
            colsample_bytree=self.params_best['colsample_bytree'],
            class_weight=self.para_space['class_weight'],
            reg_lambda=self.params_best['reg_lambda']
            ).fit(X,y)      
        
        return self
    
    
    def LGBM_CV(self,n_estimators,learning_rate,max_depth,
               min_split_gain,min_sum_hessian_in_leaf,subsample,colsample_bytree,
               reg_lambda
              ):
               
        para_space = {                   
                      'n_estimators' : [int(n_estimators)],
                      'learning_rate' : [learning_rate],   
                      'max_depth':  [int(max_depth)], 
                      'min_split_gain':[min_split_gain],  
                      'min_sum_hessian_in_leaf':[min_sum_hessian_in_leaf],        
                      'subsample' : [subsample],
                      'colsample_bytree' : [colsample_bytree],
                      'reg_lambda':[reg_lambda]
                      }       
        
        
        if self.scoring=='ks':
            scorer=metrics.make_scorer(self.custom_score_KS,greater_is_better=True,needs_proba=True)
        elif self.scoring=='auc':
            scorer=metrics.make_scorer(self.custom_score_AUC,greater_is_better=True,needs_proba=True)
        elif self.scoring=='lift':
            scorer=metrics.make_scorer(self.custom_score_Lift,greater_is_better=True,needs_proba=True)
        else:
            raise IOError('scoring not understood,should be "ks","auc","lift")')
                        
        cv_res=GridSearchCV(
            LGBMClassifier(seed=123,min_child_weight=None),para_space,cv=self.cv,
            scoring=scorer,error_score=0).fit(self.X,self.y)
        
        print(cv_res.cv_results_['mean_test_score'])
        val_score = cv_res.cv_results_['mean_test_score'][0]
        print(' Stopped after %d iterations with val-%s = %f' % (n_estimators,self.scoring,val_score))
        return(val_score)    
    
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
    
    def cvresult_to_df(self):

        ParaDf_all=pd.DataFrame()
    
        for i in range(len(self.Optimize.res)):
            ParaDf=pd.DataFrame([self.Optimize.res[i]['params']])
            ParaDf['val_'+self.scoring]=self.Optimize.res[i]['target']    
            ParaDf_all=pd.concat([ParaDf,ParaDf_all],ignore_index=True)
        
        ParaDf_all['boosting_type']=self.para_space['boosting_type']    
        ParaDf_all['class_weight']=self.para_space['class_weight']
        
        return ParaDf_all    
    
    
    
    
    
    