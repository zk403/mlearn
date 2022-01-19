#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:26:03 2021

@author: zengke
"""
from sklearn.base import BaseEstimator
from BDMLtools.base import Base
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from BDMLtools.tuner.fun import sLGBMClassifier
from bayes_opt import BayesianOptimization
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold
#from time import time
import numpy as np
import pandas as pd
from joblib import effective_n_jobs


class BayesianXGBTuner(Base,BaseEstimator):
    
    '''
    使用贝叶斯优化参数的Xgboost
    Parameters:
    --
        para_space:dict,xgboost的参数空间
        n_iter:贝叶斯优化搜索迭代次数
        init_points:int,贝叶斯优化起始搜索点的个数
        scoring:str,寻优准则,可选'auc','ks','lift'
        cv:int,RepeatedStratifiedKFold交叉验证的折数
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        refit:bool,最优参数下是否重新在全量数据上拟合模型，默认True
        n_jobs,int,运行交叉验证时的joblib的并行数,默认-1
        verbose,int,并行信息输出等级
        random_state,随机种子
        sample_weight:样本权重
        calibration:使用sklearn的CalibratedClassifierCV对refit=True下的模型进行概率校准
        cv_calibration:CalibratedClassifierCV的交叉验证数,注意因无验证数据，不推荐设定为'prefit'
        
        """参数空间写法
    
        para_space={
             'n_estimators': (80, 150),
             'learning_rate': (0.05, 0.2),
             'max_depth': (3, 10),
             'gamma': (0, 20),
             'min_child_weight': (0, 10),
             'max_delta_step': (0, 0),
             'scale_pos_weight': (11,11),
             'subsample': (0.5, 1),
             'colsample_bytree': (0.5, 1),
             'reg_lambda': (0, 10)
                       }
        
        """        
    
    Attribute:    
    --
        Optimize:贝叶斯优化迭代器,需先使用fit
        params_best:最优参数组合,需先使用fit
        model_refit:最优参数下的xgboost模型,需先使用fit且参数refit=True 

    '''     
    
    
    def __init__(self,para_space,n_iter=10,init_points=5,scoring='auc',cv=5,repeats=1,refit=True,
                 n_jobs=-1,verbose=0,random_state=123,sample_weight=None,calibration=False,cv_calibration=5):
       
        self.para_space=para_space
        self.n_iter=n_iter
        self.init_points=init_points
        self.scoring=scoring
        self.cv=cv
        self.repeats=repeats
        self.refit=refit
        self.n_jobs=n_jobs
        self.verbose=verbose 
        self.random_state=random_state
        self.sample_weight=sample_weight
        self.calibration=calibration
        self.cv_calibration=cv_calibration
        
        self._is_fitted=False
        
    def predict_proba(self,X,y=None):
        '''
        最优参数下的xgboost模型的预测
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
        
        return X
          
    def fit(self,X,y):
        '''
        进行贝叶斯优化
        Parameters:
        --
        X:pd.DataFrame对象
        y:目标变量,pd.Series对象
        '''   
        self._check_data(X, y)        
        
        self.X=X.copy()
        self.y=y.copy()
        
        self.Optimize = BayesianOptimization(self._XGB_CV,self.para_space)
        self.Optimize.maximize(n_iter=self.n_iter,init_points=self.init_points)
        
        #输出最优参数组合
        self.params_best=self.Optimize.max['params']
        self.params_best['max_depth']=int(self.params_best['max_depth'])
        self.params_best['n_estimators']=int(self.params_best['n_estimators'])   
        
        #交叉验证结果保存
        self.cv_result=self._cvresult_to_df()
        
        #refit
        if self.refit:
            
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
                ).fit(X,y,sample_weight=self.sample_weight)      
            
            if self.calibration:
                
                self.model_refit=CalibratedClassifierCV(self.model_refit,cv=self.cv_calibration,
                                                      n_jobs=self.n_jobs).fit(X,y,sample_weight=self.sample_weight)
                
        del self.X,self.y
        
        self._is_fitted=True
        
        return self
    
    
    def _XGB_CV(self,n_estimators,max_depth,gamma,learning_rate,min_child_weight,
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
            
            scorer=metrics.make_scorer(self._custom_score_KS,greater_is_better=True,needs_proba=True)
            
        elif self.scoring=='auc':
            
            scorer=metrics.make_scorer(self._custom_score_AUC,greater_is_better=True,needs_proba=True)
            
        elif self.scoring=='lift':
            
            scorer=metrics.make_scorer(self._custom_score_Lift,greater_is_better=True,needs_proba=True)
            
        else:
            
            raise ValueError('scoring not understood,should be "ks","auc","lift")')
            
        cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state)        
        
        n_jobs=effective_n_jobs(self.n_jobs)
                        
        cv_res=GridSearchCV(
            XGBClassifier(seed=self.random_state,use_label_encoder=False,verbosity=0),para_space,cv=cv,
            n_jobs=n_jobs,verbose=self.verbose,
            scoring=scorer,error_score=0).fit(self.X,self.y,sample_weight=self.sample_weight)
        
        #print(cv_res.cv_results_['mean_test_score'])
        val_score = cv_res.cv_results_['mean_test_score'][0]
        
        print(' Stopped after %d iterations with val-%s = %f' % (n_estimators,self.scoring,val_score))
        
        return(val_score)    
    
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
    
    def _cvresult_to_df(self):
        '''
        输出交叉验证结果
        '''   

        ParaDf_all=pd.DataFrame()
    
        for i in range(len(self.Optimize.res)):
            ParaDf=pd.DataFrame([self.Optimize.res[i]['params']])
            ParaDf['val_'+self.scoring]=self.Optimize.res[i]['target']    
            ParaDf_all=pd.concat([ParaDf,ParaDf_all],ignore_index=True)
        
        ParaDf_all['booster']='gbtree'
        
        return ParaDf_all
    
    def _p_to_score(self,pred,PDO=75,base=660,ratio=1/15):
        
        B=1*PDO/np.log(2)
        A=base + B*np.log(ratio)
        score=A-B*np.log(pred/(1-pred))
        
        return np.round(score,0)
    

class BayesianLgbmTuner(Base,BaseEstimator):
    
    '''
    使用贝叶斯优化参数的LightGBM
    Parameters:
    --
        para_space:dict,lgb的参数空间
        n_iter:贝叶斯优化搜索迭代次数
        init_points:int,贝叶斯优化起始搜索点的个数
        scoring:str,寻优准则,可选'auc','ks','lift'
        cv:int,RepeatedStratifiedKFold交叉验证的折数
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        refit:bool,最优参数下是否重新在全量数据上拟合模型，默认True  
        n_jobs,int,运行交叉验证时的joblib的并行数,默认-1
        verbose,int,并行信息输出等级
        random_state,随机种子
        sample_weight:样本权重
        calibration:使用sklearn的CalibratedClassifierCV对refit=True下的模型进行概率校准
        cv_calibration:CalibratedClassifierCV的交叉验证数,注意因无验证数据，不推荐设定为'prefit'
        
        """参数空间写法        
    
        para_space={
                 'boosting_type':'gbdt', 
                 'n_estimators':(30,120),
                 'learning_rate':(0.05,0.2), 
                
                 'max_depth':(2,4),
                 'min_split_gain': (0,20),
                 'min_sum_hessian_in_leaf': (0,20),
                 
                 'scale_pos_weight':(1,1),
                 'subsample':(0.5,1),
                 'colsample_bytree' :(0.5,1),
                 'reg_lambda':(0,10), 
                 }
        
          """
    
    
    Attribute:    
    --
        Optimize:贝叶斯优化迭代器,需先使用fit
        params_best:最优参数组合,需先使用fit
        model_refit:最优参数下的lgbm模型,需先使用fit
    
    Examples
    --

    '''    
    
    def __init__(self,para_space,n_iter=10,init_points=5,scoring='auc',cv=5,repeats=1,refit=True,
                 n_jobs=-1,verbose=0,random_state=123,sample_weight=None,calibration=False,cv_calibration=5):
        
        self.para_space=para_space
        self.n_iter=n_iter
        self.init_points=init_points
        self.scoring=scoring
        self.cv=cv
        self.repeats=repeats
        self.refit=refit
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.random_state=random_state
        self.sample_weight=sample_weight
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
        进行贝叶斯优化
        Parameters:
        --
        X: pd.DataFrame对象
        y:目标变量,pd.Series对象
        '''   
        self._check_data(X, y)
        
        self.X=X.copy()
        self.y=y.copy()
        
        para_space_num={key:self.para_space[key] for key in self.para_space if key not in ('boosting_type','class_weight')}  
        
        self.Optimize = BayesianOptimization(self._LGBM_CV,para_space_num)
        self.Optimize.maximize(n_iter=self.n_iter,init_points=self.init_points)
        
        #输出最优参数组合
        self.params_best=self.Optimize.max['params']
        self.params_best['max_depth']=int(self.params_best['max_depth'])
        self.params_best['n_estimators']=int(self.params_best['n_estimators'])   
        
        #交叉验证结果保存
        self.cv_result=self._cvresult_to_df()
        
        if self.refit:
            #print (self.para_space)
            self.model_refit = sLGBMClassifier(
                boosting_type=self.para_space['boosting_type'],
                n_estimators=self.params_best['n_estimators'],
                learning_rate=self.params_best['learning_rate'],
                max_depth=self.params_best['max_depth'],          
                min_split_gain=self.params_best['min_split_gain'],
                min_sum_hessian_in_leaf=self.params_best['min_sum_hessian_in_leaf'],
                subsample=self.params_best['subsample'],
                colsample_bytree=self.params_best['colsample_bytree'],
                #class_weight=self.para_space['class_weight'],
                scale_pos_weight=self.params_best['scale_pos_weight'],
                reg_lambda=self.params_best['reg_lambda']
                ).fit(X,y,self.sample_weight)      
            
            if self.calibration:
                
                self.model_refit=CalibratedClassifierCV(self.model_refit,cv=self.cv_calibration,
                                                      n_jobs=self.n_jobs).fit(X,y,sample_weight=self.sample_weight)
                
        self._is_fitted=True
        
        return self
    
    
    def _LGBM_CV(self,n_estimators,learning_rate,max_depth,
               min_split_gain,min_sum_hessian_in_leaf,subsample,colsample_bytree,scale_pos_weight,
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
                      'scale_pos_weight' : [scale_pos_weight],
                      'reg_lambda':[reg_lambda]
                      }               
        
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
                        
        cv_res=GridSearchCV(
            sLGBMClassifier(seed=self.random_state,min_child_weight=None),para_space,cv=cv,
            n_jobs=n_jobs,verbose=self.verbose,
            scoring=scorer,error_score=0).fit(self.X,self.y,sample_weight=self.sample_weight)
        
        #print(cv_res.cv_results_['mean_test_score'])
        val_score = cv_res.cv_results_['mean_test_score'][0]
        
        print(' Stopped after %d iterations with val-%s = %f' % (n_estimators,self.scoring,val_score))
        
        return(val_score)    
    
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
    
    def _cvresult_to_df(self):
        '''
        输出交叉验证结果
        '''   

        ParaDf_all=pd.DataFrame()
    
        for i in range(len(self.Optimize.res)):
            ParaDf=pd.DataFrame([self.Optimize.res[i]['params']])
            ParaDf['val_'+self.scoring]=self.Optimize.res[i]['target']    
            ParaDf_all=pd.concat([ParaDf,ParaDf_all],ignore_index=True)
        
        ParaDf_all['boosting_type']=self.para_space['boosting_type']    
        
        return ParaDf_all    
    
    def _p_to_score(self,pred,PDO=75,base=660,ratio=1/15):
        
        B=1*PDO/np.log(2)
        A=base + B*np.log(ratio)
        score=A-B*np.log(pred/(1-pred))
        
        return np.round(score,0)
    
    
    
    