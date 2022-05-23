#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:46:42 2022

@author: zengke
"""

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c
from joblib import Parallel,delayed,effective_n_jobs
from BDMLtools.base import Base
import numpy as np
import pandas as pd


class lassoSelector(Base,BaseEstimator,TransformerMixin):    
    
    """
    Lasso进行特征筛选
    Parameters:
    --
        
        c_num=50:l1正则项的粒度，越大的数值代表越多的l1正则项被遍历
        method='1se':str,Lasso进行变量筛选的方法，可选‘1se’,‘aic’,'bic'  
            + '1se':一倍标准差原则，同R的glmnet。交叉验证中最优指标(logloss或其他优化指标)下的c值对应一倍标准差范围内的最小c值作为筛选标准
            + ‘aic’或'bic',使用aic或bic作为lasso变量筛选标准，筛选最小aic或最小bic下的模型特征，注意使用aic或bic时将不进行交叉验证
                + aic:平衡特征个数与模型错误率
                + bic:平衡样本量、特征个数与模型错误率                                           
        max_iter:sklearn.linear_model.LogisticRegression的最大迭代次数
        class_weight:sklearn.linear_model.LogisticRegression的类权重
        scoring:method='1se'时优化指标,可以为logloss、auc、ks等
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        cv:int,交叉验证的折数
        keep:需要保留的列的列名list
        n_jobs,int,joblib的并行数,默认-1
        verbose,int,并行信息输出等级
        random_state:随机种子
        sample_weight:样本权重
     
    Attribute:    
    --
        columns:筛选后的特征名,不包括keep列
        
    Examples
    --

    """  
    def __init__(self):
        pass
        
    def transform(self,X,y):
        """ 
        变量筛选
        """

        return self
          
    def fit(self,X,y):

        return self  
    
    # def __init__(self,c_num=50,method='1se',max_iter=500,scoring='logloss',repeats=1,cv=5,class_weight=None,keep=None,
    #              n_jobs=-1,verbose=0,random_state=123,sample_weight=None):
      
    #     self.c_num=c_num
    #     self.method=method
    #     self.max_iter=max_iter
    #     self.class_weight=class_weight
    #     self.scoring=scoring
    #     self.cv=cv
    #     self.repeats=repeats
    #     self.keep=keep
    #     self.n_jobs=n_jobs
    #     self.verbose=verbose     
    #     self.random_state=random_state
    #     self.sample_weight=sample_weight
        
    #     self._is_fitted=False

        
    # def predict_proba(self,X,y=None):
    #     '''
    #     最优参数下的模型的预测
    #     Parameters:
    #     --
    #     X:pd.DataFrame对象
    #     '''      
    #     self._check_is_fitted()
    #     pred = self.model_refit.predict_proba(X)[:,1]        
    #     return pred

    
    # def transform(self,X,y=None): 
        
    #     self._check_is_fitted()
    #     self._check_X(X)
        
    #     if self.keep and isinstance(self.keep,list):
            
    #         columns=list(set(self.columns.tolist()+self.keep))
            
    #     else:
            
    #         columns=self.columns.tolist()

    #     return X[columns]
          
    # def fit(self,X,y):
    #     '''
    #     进行参数优化
    #     Parameters:
    #     --
    #     X:pd.DataFrame对象
    #     y:目标变量,pd.Series对象
    #     '''   
    #     self._check_data(X,y)
        
    #     cs = l1_min_c(X, y, loss="log") * np.logspace(0, 7, self.c_num)        
        
    #     X_s=StandardScaler().fit_transform(X)
        
    #     if self.method=='1se':
            
    #         self._lasso_1se(X_s, y, cs)   
    #         self.columns=X.columns[self.model_refit.coef_[0]!=0]
       
    #     elif self.method in ('aic','bic'):            
            
    #         self._lasso_criteria(X,y,cs)
    #         self.columns=X.columns[self.model_refit.coef_[0]!=0]

            
    #     else:
    #         raise ValueError('method should be "1se","aic","bic".')
            
        
    #     self._is_fitted=True
        
    #     return self    
    
    # def _lasso_1se(self,X,y,cs):          
    #     '''
    #     网格搜索
    #     '''  
    #     if self.scoring=='ks':
            
    #         scorer=metrics.make_scorer(self._custom_score_KS,greater_is_better=True,needs_proba=True)
            
    #     elif self.scoring=='auc':
            
    #         scorer=metrics.make_scorer(self._custom_score_AUC,greater_is_better=True,needs_proba=True)
            
    #     elif self.scoring=='lift':
            
    #         scorer=metrics.make_scorer(self._custom_score_Lift,greater_is_better=True,needs_proba=True)
            
    #     elif self.scoring=='logloss':
            
    #         scorer=metrics.make_scorer(self._custom_score_logloss,greater_is_better=True,needs_proba=True)            
    #     else:
    #         scorer=self.scoring    
            
    #     cv = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=self.repeats, random_state=self.random_state) 
        
    #     para_space={'C':cs}
        
    #     n_jobs=effective_n_jobs(self.n_jobs) 
                
    #     gird=GridSearchCV(LogisticRegression(random_state=self.random_state,class_weight=self.class_weight,
    #                                          penalty='l1',solver='saga',max_iter=self.max_iter),
    #                       para_space,
    #                       cv=cv,
    #                       n_jobs=n_jobs,
    #                       refit=True,
    #                       verbose=self.verbose,
    #                       scoring=scorer,error_score=0)    
        
    #     clf=gird.fit(X,y,sample_weight=self.sample_weight)
        
    #     cv_r=clf.cv_results_
    #     self.c_best=clf.best_params_['C']
    #     best_1se=clf.best_score_-cv_r['std_test_score'][cv_r['mean_test_score']==clf.best_score_][0]
    #     self.c_1se=cv_r['param_C'][
    #         cv_r['mean_test_score']==cv_r['mean_test_score'][cv_r['mean_test_score']>best_1se].min()
    #     ][0]
        
    #     self.model_refit = LogisticRegression(random_state=self.random_state,class_weight=self.class_weight,C=self.c_1se,
    #                                    penalty='l1',solver='saga',max_iter=self.max_iter).fit(X,y,sample_weight=self.sample_weight)                
        
    #     return self
    
    
    # def _lasso_criteria(self,X,y,cs):
        
    #     clf = LogisticRegression(random_state=self.random_state,class_weight=self.class_weight,
    #                                    penalty='l1',solver='saga',max_iter=self.max_iter).fit(X,y,sample_weight=self.sample_weight)
        
    #     n_jobs=effective_n_jobs(self.n_jobs) 
        
    #     parallel=Parallel(n_jobs=n_jobs,verbose=self.verbose)
        
    #     def fit_reg(X,y,clf,c):
            
    #         k=len(X)
            
    #         clf.set_params(C=c)
    #         clf.fit(X, y)        
            
    #         pred=clf.predict_proba(X)[:,1]     
            
    #         if self.method=='aic':
                
    #             criteria=self._calculate_aic(self._custom_score_logloss(y,pred),np.sum(clf.coef_!=0))
                
    #         elif self.method=='bic':
                
    #             criteria=self._calculate_bic(k,self._custom_score_logloss(y,pred),np.sum(clf.coef_!=0))   
                
    #         coefs_=clf.coef_.ravel().copy()
            
    #         return {'c':c,'criteria':criteria,'coefs_':coefs_}
        
    #     lasso_out=parallel(delayed(fit_reg)(X,y,clf,c) for c in cs)  
    #     lasso_out_df=pd.DataFrame(lasso_out)       
    #     c_best=lasso_out_df.sort_values('criteria').head(1)['c'].ravel()[0]
    #     self.criteria=lasso_out_df.sort_values('criteria').head(1)['criteria'].ravel()[0]        
    #     self.model_refit = LogisticRegression(random_state=self.random_state,class_weight=self.class_weight,C=c_best,
    #                                    penalty='l1',solver='saga',max_iter=self.max_iter).fit(X,y,sample_weight=self.sample_weight)  

    
    # def _custom_score_AUC(self,y_true, y_pred):        
    #     '''
    #     自定义验证评估指标AUC
    #     '''           
    #     return metrics.roc_auc_score(y_true,y_pred)
    
    # def _custom_score_KS(self,y_true, y_pred):
    #     '''
    #     自定义验证评估指标KS
    #     '''   
    #     fpr,tpr,thresholds= metrics.roc_curve(y_true,y_pred)
    #     ks = max(tpr-fpr)
    #     return ks             
        
        
    # def _custom_score_Lift(self,y_true,y_pred):
    #     '''
    #     自定义验证评估指标Lift
    #     '''   
    #     thrs = np.linspace(y_pred.min(), y_pred.max(),100)
    #     lift=[]
    #     for thr in thrs:
    #         tn, fp, fn, tp = metrics.confusion_matrix(y_true,y_pred>thr).ravel()
    #         #depth = (tp + fp)/(tn+fp+fn+tp)
    #         ppv = tp/(tp + fp)
    #         lift.append(ppv/((tp + fn)/(tn+fp+fn+tp)))
    #     return(np.nanmean(lift)) 
    
    # def _custom_score_logloss(self,y_true,y_pred):     
    #     '''
    #     自定义验证评估指标logoss
    #     '''           
    #     logloss=y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred)    
        
    #     return logloss.sum()
    
    # def _calculate_bic(self,n, log_loss, k):
        
    #     bic = -2*log_loss + np.log(n) * k
        
    #     return bic
    
    # def _calculate_aic(self,log_loss, k):
        
    #     aic = -2*log_loss+2*k
        
    #     return aic        
        

    
class sequentialSelector(BaseEstimator):
    
    def __init__(self):
        pass
        
    def transform(self,X,y):
        """ 
        变量筛选
        """

        return self
          
    def fit(self,X,y):

        return self  


    