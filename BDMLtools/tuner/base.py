# -*- coding: utf-8 -*-

from sklearn import metrics
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import warnings
from BDMLtools.base import DataTypeError,XyIndexError
from catboost.core import CatBoostClassifier
from xgboost import XGBClassifier


class BaseTunner:
    
    def _check_params_dup(self,para_space,fixed_params):
        
        if len(set(para_space.keys()) & set(fixed_params.keys())) > 0:
            
            raise ValueError('duplicated params found in fixed_params and para_space')
            
    def _check_params(self,y,sample_weight):
        
        if sample_weight is not None:
            
            if not isinstance(sample_weight,pd.Series):
                
                raise DataTypeError("sample_weight is not pandas.Series.") 
                
            if not y.index.equals(sample_weight.index):
                
                raise XyIndexError("index of sample_weight not equal to y's index") 
    
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
    
    
    
    def _get_fit_params(self,Estimator,X_train,y_train,X_val,y_val,sample_weight=None,train_index=None,val_index=None):
        
        
        if Estimator is XGBClassifier:
            
            fit_params = {
                "X":X_train,
                "y":y_train,
                "eval_set": [(X_val, y_val)],
                "eval_metric": self.eval_metric,
                "early_stopping_rounds": self.early_stopping_rounds,
            }
            
            if sample_weight is not None:
                
                fit_params["sample_weight"] = sample_weight.loc[train_index]
                fit_params["sample_weight_eval_set"] = [sample_weight.loc[val_index]]
        
            
        elif Estimator is LGBMClassifier:
            
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
            
        
        elif Estimator is CatBoostClassifier:
            
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



class sLGBMClassifier(LGBMClassifier):
    
    """ 
    过滤掉LGBM的warning信息
    """    
    
    def fit(self, *args, **kwargs):        
        
        with warnings.catch_warnings():   
            
            warnings.filterwarnings("ignore", category=UserWarning)
            
            return super().fit(*args, verbose=False, **kwargs)
        