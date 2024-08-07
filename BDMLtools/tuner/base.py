# -*- coding: utf-8 -*-

from sklearn import metrics,__version__
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import special


class BaseTunner:
    
    def _get_scorer(self,name):
        
        scorer={'auc':self._custom_scorer_AUC,
                'logloss':self._custom_scorer_neglogloss
                }
        
        if name not in scorer:
            
            raise ValueError("scorer in ('auc','logloss'),if using focaloss model then scorer in ('auc','logloss','negfocalloss')")
        
        return scorer[name]
    
    @property    
    def _custom_scorer_AUC(self):   
        
        if __version__<'1.4.0':

            scorer=metrics.make_scorer(metrics.roc_auc_score,greater_is_better=True,needs_proba=True)
            
        else:
            
            scorer=metrics.make_scorer(metrics.roc_auc_score,greater_is_better=True,response_method="predict_proba")
        
        return scorer
    
    
    @property    
    def _custom_scorer_neglogloss(self):
            
        if __version__<'1.4.0':
        
            scorer=metrics.make_scorer(metrics.log_loss,greater_is_better=False,needs_proba=True)  
            
        else:
            
            scorer=metrics.make_scorer(metrics.log_loss,greater_is_better=False,response_method="predict_proba")  
        
        return scorer
    
        
    def _p_to_score(self,pred,PDO=75,base=660,ratio=1/15):
        
        B=1*PDO/np.log(2)
        A=base + B*np.log(ratio)
        score=A-B*np.log(pred/(1-pred))
        
        return np.round(score,0)
    
    
    def _cvresult_to_df(self,cv_results_):
    
        return pd.DataFrame(cv_results_)   
    
    
        
class FLLGBMSklearn(LGBMClassifier):
    """ 
    适合Focal Loss的lightgbm
    改造了predict_proba，注意前提为init_score被设定为定值
    """  

    
    def predict(self, *args, **kwargs):        
        
        pred_pos=special.expit(super().predict_proba(*args, **kwargs,raw_score=True)) 
        return np.int64(pred_pos>0.5)
    
    
    def predict_proba(self, *args, **kwargs):        
        
        pred_pos=special.expit(super().predict_proba(*args, **kwargs,raw_score=True))
        pred_neg=1-pred_pos  
        return np.array([pred_neg,pred_pos]).T
    
    
class FocalLoss:
    """ 
    自定义的Focal Loss损失函数，用于定制lightgbm的objective
    参考:[Focal loss implementation for LightGBM](https://maxhalford.github.io/blog/lightgbm-focal-loss/)
    """  

    def __init__(self, gamma, alpha=None):
        
        self.alpha = alpha #权重
        self.gamma = gamma #惩罚

    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        """ 
        主函数
        """ 
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        """ 
        Focal Loss的一阶梯度
        """ 
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        """ 
        Focal Loss的二阶梯度
        """ 
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def lgb_obj(self,y_true,y_pred):
        """ 
        Focal Loss目标函数，用于传入LGBMClassifier
        """ 
        y = y_true
        p = special.expit(y_pred)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, y_true,y_pred):
        """ 
        Focal Loss评估函数，用于存在验证集和earlystopping时传入LGBMClassifier.fit,这里取了负数
        """ 

        p = y_pred
        is_higher_better = True
        return 'neg_focal_loss', -self(y_true, p).mean(), is_higher_better
        
    def lgb_scorer(self, y_true,y_pred):
        """ 
        Focal Loss打分函数，用于参数寻优交叉验证时评估验证集效果,这里取了负数
        """ 

        p = y_pred
        return -self(y_true, p).mean() 