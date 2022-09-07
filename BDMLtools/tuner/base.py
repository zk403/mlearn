# -*- coding: utf-8 -*-

from sklearn import metrics
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import warnings
#from catboost.core import CatBoostClassifier
#from xgboost import XGBClassifier


class BaseTunner:
    
    @property
    def _get_scorer(self):
        
        return {'auc':self._custom_scorer_AUC,
                'ks':self._custom_scorer_KS,
                'lift':self._custom_scorer_Lift,
                'neglogloss':self._custom_scorer_neglogloss
                }
    
    @property    
    def _custom_scorer_AUC(self):
    
        def custom_score_AUC(y_true, y_pred):        
            '''
            自定义验证评估指标AUC
            '''           
            return metrics.roc_auc_score(y_true,y_pred)  

        scorer=metrics.make_scorer(custom_score_AUC,greater_is_better=True,needs_proba=True)
        
        return scorer

    @property        
    def _custom_scorer_KS(self):
        
        def custom_score_KS(y_true, y_pred):
            '''
            自定义验证评估指标KS
            '''   
            fpr,tpr,thresholds= metrics.roc_curve(y_true,y_pred)
            ks = max(tpr-fpr)
            return ks  
    
        scorer=metrics.make_scorer(custom_score_KS,greater_is_better=True,needs_proba=True)
        
        return scorer
    
    @property    
    def _custom_scorer_Lift(self):
        
        def custom_score_Lift(y_true,y_pred):
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
        
        scorer=metrics.make_scorer(custom_score_Lift,greater_is_better=True,needs_proba=True)
        
        return scorer        
    
    @property    
    def _custom_scorer_neglogloss(self):
        
        def custom_score_neglogloss(y_true,y_pred):     
            '''
            自定义验证评估指标logoss
            '''           
            logloss=y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred)    
            
            return logloss.sum()
    
        scorer=metrics.make_scorer(custom_score_neglogloss,greater_is_better=True,needs_proba=True)  
        
        return scorer
        
    def _p_to_score(self,pred,PDO=75,base=660,ratio=1/15):
        
        B=1*PDO/np.log(2)
        A=base + B*np.log(ratio)
        score=A-B*np.log(pred/(1-pred))
        
        return np.round(score,0)
    
    
    def _cvresult_to_df(self,cv_results_):
    
        return pd.DataFrame(cv_results_)   


class sLGBMClassifier(LGBMClassifier):
    
    """ 
    过滤掉LGBM的warning信息
    """    
    
    def fit(self, *args, **kwargs):        
        
        with warnings.catch_warnings():   
            
            warnings.filterwarnings("ignore", category=UserWarning)
            
            return super().fit(*args, verbose=False, **kwargs)
        