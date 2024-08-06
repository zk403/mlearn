#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:00:29 2022

@author: zengke
"""

from BDMLtools.tuner import gridTuner,hgridTuner,FLgridTuner
from BDMLtools.tuner import BayesianCVTuner,FLBSTuner
from BDMLtools.tuner.fun import shapCheck
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform 
import pandas as pd
import mock


def test_gridTuner_xgb():    
    
    X=pd.DataFrame(
        {
         'a':[1,2,2,4,5,1,2,2,4,5,1,2,2,4,5,1,2,2,4,5],
         'b':[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5],
         'c':[1,1,1,2,1,1,1,1,2,1,1,1,1,2,1,1,1,1,2,1]}
        )
    y=pd.Series([0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1],name='y')
          
    para_space_gird={    'n_estimators':[10],
                         'learning_rate':[0.1],
                        
                         'max_depth':[3],
                         'gamma': [0],
                         'min_child_weight':[0],
                         
                         'subsample':[0.6],
                         'colsample_bytree' :[0.6],
                         'reg_lambda':[0],
                         
                         #'use_label_encoder':[False]
                         }
    
    res=gridTuner(XGBClassifier,para_space_gird,method='grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=None).fit(X,y)  
    res=gridTuner(XGBClassifier,para_space_gird,method='grid',n_jobs=1,cv=2,
              validation_fraction=0.1,early_stopping_rounds=None).fit(X,y)  
    res=gridTuner(XGBClassifier,para_space_gird,method='grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=10).fit(X,y)   
    res.predict_proba(X);res.predict_score(X)
    

    res=hgridTuner(XGBClassifier,para_space_gird,method='h_grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=None).fit(X,y)  
    res=hgridTuner(XGBClassifier,para_space_gird,method='h_grid',n_jobs=1,cv=2,
              validation_fraction=0.1,early_stopping_rounds=None).fit(X,y)  
    res=hgridTuner(XGBClassifier,para_space_gird,method='h_grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=10).fit(X,y)
    res.predict_proba(X);res.predict_score(X)
    
    
    gridTuner(XGBClassifier,para_space_gird,method='grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=None,scoring='auc').fit(X,y)     
    gridTuner(XGBClassifier,para_space_gird,method='grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=None,scoring='logloss').fit(X,y)  
    

    para_space_random={   'n_estimators':sp_randint(low=60,high=120),#迭代次数
                          'learning_rate':sp_uniform(loc=0.05,scale=0.15), #学习率                         
                          'max_depth':sp_randint(low=2,high=4),
                          'gamma': sp_uniform(loc=0,scale=21),
                          'min_child_weight': sp_uniform(loc=0,scale=21),                          
                          'subsample':sp_uniform(loc=0.5,scale=0.5),
                          'colsample_bytree' :sp_uniform(loc=0.5,scale=0.5),                          
                          'reg_lambda':sp_randint(low=0,high=1), 
                          #'use_label_encoder':[False]
                         }   
    
    res=gridTuner(XGBClassifier,para_space_random,method='random_grid',n_jobs=1,cv=2,n_iter=1,
              validation_fraction=None,early_stopping_rounds=None).fit(X,y)
    res=gridTuner(XGBClassifier,para_space_random,method='random_grid',n_jobs=1,cv=2,n_iter=1,
              validation_fraction=0.1,early_stopping_rounds=None).fit(X,y)
    res=gridTuner(XGBClassifier,para_space_random,method='random_grid',n_jobs=1,cv=2,n_iter=1,
              validation_fraction=None,early_stopping_rounds=10).fit(X,y)
    res.predict_proba(X);res.predict_score(X)
    
    res=hgridTuner(XGBClassifier,para_space_random,method='h_random',n_jobs=1,cv=2,n_candidates=1,
              validation_fraction=None,early_stopping_rounds=None).fit(X,y)
    res=hgridTuner(XGBClassifier,para_space_random,method='h_random',n_jobs=1,cv=2,n_candidates=1,
              validation_fraction=0.1,early_stopping_rounds=None).fit(X,y)
    res=hgridTuner(XGBClassifier,para_space_random,method='h_random',n_jobs=1,cv=2,n_candidates=1,
              validation_fraction=None,early_stopping_rounds=10).fit(X,y)
    res.predict_proba(X);res.predict_score(X)
    
    
def test_gridTuner_lgbm():
    
    X=pd.DataFrame(
        {
         'a':[1,2,2,4,5,1,2,2,4,5,1,2,2,4,5,1,2,2,4,5],
         'b':[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5],
         'c':[1,1,1,2,1,1,1,1,2,1,1,1,1,2,1,1,1,1,2,1]}
        )
    y=pd.Series([0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1],name='y')
    
    para_space_gird={ 'boosting_type':['goss'], 
                        'n_estimators':[10],
                        'learning_rate':[0.1],                        
                        'max_depth':[3],
                        'min_split_gain': [0],                       
                        'subsample':[1],
                        'colsample_bytree' :[1],
                        'reg_lambda':[0], 
                        'verbosity':[-1]
                         }
    res=gridTuner(LGBMClassifier,para_space_gird,method='grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=None,scoring='auc').fit(X,y)
    res=gridTuner(LGBMClassifier,para_space_gird,method='grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=None,scoring='logloss').fit(X,y)
    
    res=gridTuner(LGBMClassifier,para_space_gird,method='grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=None).fit(X,y)
    res=gridTuner(LGBMClassifier,para_space_gird,method='grid',n_jobs=1,cv=2,
              validation_fraction=0.1,early_stopping_rounds=None).fit(X,y)
    res=gridTuner(LGBMClassifier,para_space_gird,method='grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=10).fit(X,y)
    res.predict_proba(X);res.predict_score(X)
    
    res=hgridTuner(LGBMClassifier,para_space_gird,method='h_grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=None).fit(X,y)
    res=hgridTuner(LGBMClassifier,para_space_gird,method='h_grid',n_jobs=1,cv=2,
              validation_fraction=0.1,early_stopping_rounds=None).fit(X,y)
    res=hgridTuner(LGBMClassifier,para_space_gird,method='h_grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=10).fit(X,y)
    res.predict_proba(X);res.predict_score(X)
    

    para_space_random={  
                     'boosting_type':['gbdt','goss'], #'goss','gbdt'
                     'n_estimators':sp_randint(low=100,high=110),
                     'learning_rate':sp_uniform(loc=0.1,scale=0),                    
                     'max_depth':sp_randint(low=2,high=4),#[0,∞],
                     'min_split_gain': sp_uniform(loc=0,scale=0),
                     'min_child_weight': sp_uniform(loc=0,scale=0),           
                     'subsample':sp_uniform(loc=0.5,scale=0.5),
                     'colsample_bytree' :sp_uniform(loc=0.5,scale=0.5),
                     'reg_lambda':sp_uniform(loc=0,scale=20),
                         }   
    
    res=gridTuner(LGBMClassifier,para_space_random,method='random_grid',n_jobs=1,cv=2,n_iter=1,
              validation_fraction=None,early_stopping_rounds=None).fit(X,y)
    res=gridTuner(LGBMClassifier,para_space_random,method='random_grid',n_jobs=1,cv=2,n_iter=1,
              validation_fraction=0.1,early_stopping_rounds=None).fit(X,y)
    res=gridTuner(LGBMClassifier,para_space_random,method='random_grid',n_jobs=1,cv=2,n_iter=1,
              validation_fraction=None,early_stopping_rounds=10).fit(X,y)
    res.predict_proba(X);res.predict_score(X)
    
    res=hgridTuner(LGBMClassifier,para_space_random,method='h_random',n_jobs=1,cv=2,n_candidates=1,
              validation_fraction=None,early_stopping_rounds=None).fit(X,y)    
    res=hgridTuner(LGBMClassifier,para_space_random,method='h_random',n_jobs=1,cv=2,n_candidates=1,
              validation_fraction=0.1,early_stopping_rounds=None).fit(X,y)    
    res=hgridTuner(LGBMClassifier,para_space_random,method='h_random',n_jobs=1,cv=2,n_candidates=1,
              validation_fraction=None,early_stopping_rounds=10).fit(X,y)    
    res.predict_proba(X);res.predict_score(X)
    

def test_gridTuner_cb():
    
    X=pd.DataFrame(
            {
             'a':[1,2,2,4,5,1,2,2,4,5,1,2,2,4,5,1,2,2,4,5],
             'b':[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5],
             'c':['1','1','1','2','1','1','1','1','2','1','1','1','1','2','1','1','1','1','2','1']},
            )
    y=pd.Series([0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1],name='y')

    para_space_gird={
        'nan_mode':['Min'],
        'n_estimators': [80],
        'learning_rate': [0.03],
        'max_depth': [2],
        'scale_pos_weight': [1],
        'subsample': [1],
        'colsample_bylevel': [1],
        'reg_lambda': [0]
    }
    
    res=gridTuner(CatBoostClassifier,para_space_gird,method='grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=None).fit(X,y)  
    res=gridTuner(CatBoostClassifier,para_space_gird,method='grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=10).fit(X,y)  
    res=gridTuner(CatBoostClassifier,para_space_gird,method='grid',n_jobs=1,cv=2,
              validation_fraction=0.1,early_stopping_rounds=None).fit(X,y)  
    res.predict_proba(X);res.predict_score(X)
    
    res=hgridTuner(CatBoostClassifier,para_space_gird,method='h_grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=None).fit(X,y)
    res=hgridTuner(CatBoostClassifier,para_space_gird,method='h_grid',n_jobs=1,cv=2,
              validation_fraction=0.1,early_stopping_rounds=None).fit(X,y)
    res=hgridTuner(CatBoostClassifier,para_space_gird,method='h_grid',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=10).fit(X,y)
    res.predict_proba(X);res.predict_score(X)
    
    para_space_random={  
                        'nan_mode':['Min'],
                        'n_estimators':sp_randint(low=100,high=110),
                        'learning_rate':sp_uniform(loc=0.1,scale=0),                     
                        'max_depth':sp_randint(low=2,high=4),#[0,∞],                     
                        'scale_pos_weight':[1],
                        'subsample':sp_uniform(loc=0.5,scale=0.5),
                        'colsample_bylevel' :sp_uniform(loc=0.5,scale=0.5),
                        'reg_lambda':sp_uniform(loc=0,scale=20),
                         }      
    res=gridTuner(CatBoostClassifier,para_space_random,method='random_grid',n_jobs=1,cv=2,n_iter=1,
              validation_fraction=None,early_stopping_rounds=None).fit(X,y)  
    res=gridTuner(CatBoostClassifier,para_space_random,method='random_grid',n_jobs=1,cv=2,n_iter=1,
              validation_fraction=0.1,early_stopping_rounds=None).fit(X,y)
    res=gridTuner(CatBoostClassifier,para_space_random,method='random_grid',n_jobs=1,cv=2,n_iter=1,
              validation_fraction=None,early_stopping_rounds=10).fit(X,y)
    res.predict_proba(X);res.predict_score(X)
   
    res=hgridTuner(CatBoostClassifier,para_space_random,method='h_random',n_jobs=1,cv=2,n_candidates=1,
              validation_fraction=None,early_stopping_rounds=None).fit(X,y)  
    res=hgridTuner(CatBoostClassifier,para_space_random,method='h_random',n_jobs=1,cv=2,n_candidates=1,
              validation_fraction=0.1,early_stopping_rounds=None).fit(X,y)  
    res=hgridTuner(CatBoostClassifier,para_space_random,method='h_random',n_jobs=1,cv=2,n_candidates=1,
              validation_fraction=None,early_stopping_rounds=10).fit(X,y)   
    res.predict_proba(X);res.predict_score(X)      


def test_FLgridTuner():    
    
    import numpy as np

    X=pd.DataFrame(np.random.rand(1000,4))
    y=pd.Series(np.random.randint(0,2,1000),name='y')

    para_space_gird={ 'boosting_type':['goss'], 
                        'n_estimators':[10],
                        'learning_rate':[0.1],                        
                        'max_depth':[3],
                        'min_split_gain': [0],                       
                        'subsample':[1],
                        'colsample_bytree' :[1],
                        'reg_lambda':[0], 
                         }
    
    
    res=FLgridTuner(para_space=para_space_gird,gamma=2,alpha=0.2,method='grid',
              scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=None).fit(X,y)
    
    res=FLgridTuner(para_space=para_space_gird,gamma=0,alpha=None,method='grid',scoring='auc',
                eval_metric='auc',n_jobs=1,cv=2,        
                validation_fraction=0.1,early_stopping_rounds=20).fit(X,y)
    
    res=FLgridTuner(para_space_gird,gamma=2,alpha=0.2,method='grid',scoring='focalloss',eval_metric='focalloss',
              n_jobs=1,cv=2,validation_fraction=None,early_stopping_rounds=None).fit(X,y)
    
    res=FLgridTuner(para_space_gird,gamma=2,alpha=0.2,method='grid',scoring='focalloss',eval_metric='focalloss',n_jobs=1,cv=2,             
              validation_fraction=0.1,early_stopping_rounds=10).fit(X,y)
    
    res.predict_proba(X);res.predict_score(X)    
    
    para_space_random={  
                     'boosting_type':['gbdt','goss'], #'goss','gbdt'
                     'n_estimators':sp_randint(low=100,high=110),
                     'learning_rate':sp_uniform(loc=0.1,scale=0),                    
                     'max_depth':sp_randint(low=2,high=4),#[0,∞],
                     'min_split_gain': sp_uniform(loc=0,scale=0),
                     'min_child_weight': sp_uniform(loc=0,scale=0),           
                     'subsample':sp_uniform(loc=0.5,scale=0.5),
                     'colsample_bytree' :sp_uniform(loc=0.5,scale=0.5),
                     'reg_lambda':sp_uniform(loc=0,scale=20),
                         }   
    
    res=FLgridTuner(para_space_random,gamma=2,alpha=0.2,method='random_grid',
              scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
              validation_fraction=None,early_stopping_rounds=None).fit(X,y)
    
    res=FLgridTuner(para_space_random,gamma=0,alpha=None,method='random_grid',scoring='auc',
                eval_metric='auc',n_jobs=1,cv=2,        
                validation_fraction=0.1,early_stopping_rounds=20).fit(X,y)
    
    res=FLgridTuner(para_space_random,gamma=2,alpha=0.2,method='random_grid',scoring='focalloss',eval_metric='focalloss',
              n_jobs=1,cv=2,validation_fraction=None,early_stopping_rounds=None).fit(X,y)
    
    res=FLgridTuner(para_space_random,gamma=2,alpha=0.2,method='random_grid',scoring='focalloss',eval_metric='focalloss',n_jobs=1,cv=2,             
              validation_fraction=0.1,early_stopping_rounds=10).fit(X,y)
    

    res.predict_proba(X);res.predict_score(X)       
    
    

def test_BayesianCVTuner():    
    
    X=pd.DataFrame(
        {
         'a':[1,2,2,4,5,1,2,2,4,5,1,2,2,4,5,1,2,2,4,5],
         'b':[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5],
         'c':[1,1,1,2,1,1,1,1,2,1,1,1,1,2,1,1,1,1,2,1]}
        )
    y=pd.Series([0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1],name='y')

    res=BayesianCVTuner(XGBClassifier,scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
                    n_trials=1,early_stopping_rounds=None).fit(X,y)
    res=BayesianCVTuner(XGBClassifier,scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
                    n_trials=1,early_stopping_rounds=10).fit(X,y)
    res=BayesianCVTuner(XGBClassifier,scoring='logloss',eval_metric='logloss',n_jobs=1,cv=2,
                    n_trials=10).fit(X,y)
    res.predict_proba(X);res.predict_score(X)
    
    res=BayesianCVTuner(LGBMClassifier,scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
                    n_trials=1,early_stopping_rounds=None).fit(X,y)
    res=BayesianCVTuner(LGBMClassifier,scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
                    n_trials=1,early_stopping_rounds=10).fit(X,y)
    res=BayesianCVTuner(LGBMClassifier,scoring='logloss',eval_metric='logloss',n_jobs=1,cv=2,
                    n_trials=1).fit(X,y)
    res.predict_proba(X);res.predict_score(X)
    
    
    X=pd.DataFrame(
            {
             'a':[1,2,2,4,5,1,2,2,4,5,1,2,2,4,5,1,2,2,4,5],
             'b':[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5],
             'c':['1','1','1','2','1','1','1','1','2','1','1','1','1','2','1','1','1','1','2','1']},
            )
    y=pd.Series([0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1],name='y')    

    res=BayesianCVTuner(CatBoostClassifier,scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
                    n_trials=1,early_stopping_rounds=None).fit(X,y,cat_features=['c'])
    res=BayesianCVTuner(CatBoostClassifier,scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
                    n_trials=1,early_stopping_rounds=10).fit(X,y,cat_features=['c'])
    res=BayesianCVTuner(CatBoostClassifier,scoring='logloss',eval_metric='logloss',n_jobs=1,cv=2,
                    n_trials=1,early_stopping_rounds=None).fit(X,y,cat_features=['c'])
    res=BayesianCVTuner(CatBoostClassifier,scoring='logloss',eval_metric='logloss',n_jobs=1,cv=2,
                    n_trials=1).fit(X[['a','b']],y) 
    res=BayesianCVTuner(CatBoostClassifier,scoring='auc',eval_metric='logloss',n_jobs=1,cv=2,
                    n_trials=1).fit(X,y) 
    res.predict_proba(X);res.predict_score(X)
    
    
    
def test_FLBSTuner():    
    
    import numpy as np
    X=pd.DataFrame(np.random.rand(1000,4))
    y=pd.Series(np.random.randint(0,2,1000),name='y')
    
    FLBSTuner(gamma=2,alpha=0.2,scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
              n_trials=1,early_stopping_rounds=None).fit(X,y)
    
    FLBSTuner(gamma=2,alpha=0.2,scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
              n_trials=1,early_stopping_rounds=10).fit(X,y)
    
    FLBSTuner(gamma=0,alpha=None,scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
              n_trials=1,early_stopping_rounds=None).fit(X,y)
    
    FLBSTuner(gamma=2,alpha=0.2,scoring='focalloss',eval_metric='focalloss',n_jobs=1,cv=2,
              n_trials=1,early_stopping_rounds=None).fit(X,y)
    
    FLBSTuner(gamma=2,alpha=0.2,scoring='focalloss',eval_metric='focalloss',n_jobs=1,cv=2,
              n_trials=1,early_stopping_rounds=10).fit(X,y)
    
    FLBSTuner(gamma=2,alpha=None,scoring='focalloss',eval_metric='focalloss',n_jobs=1,cv=2,
              n_trials=1,early_stopping_rounds=None).fit(X,y)       
    
    FLBSTuner(gamma=2,alpha=0.2,scoring='logloss',eval_metric='logloss',n_jobs=1,cv=2,
              n_trials=1,early_stopping_rounds=None).fit(X,y)
    
    FLBSTuner(gamma=2,alpha=0.2,scoring='logloss',eval_metric='logloss',n_jobs=1,cv=2,
              n_trials=1,early_stopping_rounds=10).fit(X,y)
    
    FLBSTuner(gamma=2,alpha=None,scoring='logloss',eval_metric='logloss',n_jobs=1,cv=2,
              n_trials=1,early_stopping_rounds=None).fit(X,y)
    
    res=FLBSTuner(gamma=2,alpha=0.2,scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
              n_trials=1,early_stopping_rounds=None).fit(X,y)
    
    res.predict_proba(X);res.predict_score(X)   
    

@mock.patch('matplotlib.pyplot.show')
def test_shapcheck(mock_show):
      
    X=pd.DataFrame(
            {
             'a':[1,2,2,4,5,1,2,2,4,5,1,2,2,4,5,1,2,2,4,5],
             'b':[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5],
             },
            )
    y=pd.Series([0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1],name='y')
    
    res=BayesianCVTuner(XGBClassifier,scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
                        n_trials=1,early_stopping_rounds=None).fit(X,y)
    
    xgb_m=res.model_refit.get_booster()
    
    xgb_m_1=xgb_m.save_raw('deprecated')
    
    def save_raw(self=None):
 
        return xgb_m_1
    
    xgb_m.save_raw=save_raw
        
    p=shapCheck(xgb_m).fit_plot(X,y)
    assert p.figs
    p=shapCheck(res.model_refit,woe_raw=True).fit_plot(X,y)
    assert p.figs
    
    X=pd.DataFrame(
        {
         'a':[1,2,2,4,5,1,2,2,4,5,1,2,2,4,5,1,2,2,4,5],
         'b':[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5],
         'c':[1,1,1,2,1,1,1,1,2,1,1,1,1,2,1,1,1,1,2,1]}
        )
    y=pd.Series([0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1],name='y')

    
    res=BayesianCVTuner(LGBMClassifier,scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
                        n_trials=1,early_stopping_rounds=None).fit(X,y)
        
    p=shapCheck(res.model_refit).fit_plot(X,y)
    assert p.figs
    
    X=pd.DataFrame(
            {
             'a':[1,2,2,4,5,1,2,2,4,5,1,2,2,4,5,1,2,2,4,5],
             'b':[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5],
             'c':['1','1','1','2','1','1','1','1','2','1','1','1','1','2','1','1','1','1','2','1']},
            )
    y=pd.Series([0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1],name='y')
    
    res=BayesianCVTuner(CatBoostClassifier,scoring='auc',eval_metric='auc',n_jobs=1,cv=2,
                        n_trials=1,early_stopping_rounds=None).fit(X,y)
        
    p=shapCheck(res.model_refit).fit_plot(X,y) 
    
    assert p.figs
    