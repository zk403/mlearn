#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:34:22 2022

@author: zengke
"""

from BDMLtools.selector import binSelector,faSelector,preSelector,prefitModel,LassoLogit,binAdjuster
from BDMLtools.selector import stepLogit,cardScorer,LgbmSeqSelector,LgbmShapRFECVSelector,LgbmPISelector
from BDMLtools.encoder import woeTransformer,binTransformer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import mock
from io import StringIO


def test_binSelector():

    X=pd.DataFrame(
        {
         'a':np.arange(100,dtype='float'),
         'b':np.concatenate([[-999],np.ones(98),[999]]),
         'c':np.concatenate([np.repeat('a',30),np.repeat('b',30),np.repeat('c',30),np.repeat('d',10)],dtype=object),
         'd':np.repeat(999,100),
         'e':np.repeat(np.nan,100)
         }
        )
    
    ws=pd.Series(np.ones(100),name='ws')
    
    y=pd.Series(np.append(np.zeros(50),np.ones(50)),name='y')
    
    binSelector(method='freq').fit_transform(X,y)        
    binSelector(method='freq',coerce_monotonic=True).fit_transform(X,y)  
    binSelector(method='freq',sample_weight=ws,keep=['a','c']).fit_transform(X,y)
    binSelector(method='freq',sample_weight=ws,keep=['a','c'],coerce_monotonic=True).fit_transform(X,y)   
    binSelector(method='freq',special_values=[1,2,3,4,'a']).fit_transform(X,y)   

    binSelector(method='freq-kmeans',n_jobs=1,iv_limit=0,bin_num_limit=2).fit_transform(X,y)       
    binSelector(method='freq-kmeans',n_jobs=1,iv_limit=0,coerce_monotonic=True).fit_transform(X,y)   
    binSelector(method='freq-kmeans',n_jobs=1,iv_limit=0,sample_weight=ws,keep=['a','c']).fit_transform(X,y)   
    binSelector(method='freq-kmeans',n_jobs=1,iv_limit=0,special_values=[1,2,3,4,'a']).fit_transform(X,y)        
    
    binSelector(method='tree',n_jobs=1,iv_limit=0).fit_transform(X,y)              
    binSelector(method='tree',n_jobs=1,iv_limit=0,coerce_monotonic=True).fit_transform(X,y)  
    binSelector(method='tree',n_jobs=1,iv_limit=0,sample_weight=ws).fit_transform(X,y)
    binSelector(method='tree',n_jobs=1,iv_limit=0,special_values=[1,2,3,4,'a']).fit_transform(X,y)                
    
    binSelector(method='chi2',n_jobs=1,iv_limit=0).fit_transform(X,y)               
    binSelector(method='chi2',n_jobs=1,iv_limit=0,coerce_monotonic=True).fit_transform(X,y)        
    binSelector(method='chi2',n_jobs=1,iv_limit=0,sample_weight=ws,keep=['a','c']).fit_transform(X,y)     
    binSelector(method='chi2',n_jobs=1,iv_limit=0,special_values=[1,2,3,4,'a']).fit_transform(X,y)        
    
    binSelector(method='pretty',n_jobs=1,iv_limit=0).fit_transform(X,y)               
    binSelector(method='pretty',n_jobs=1,iv_limit=0,coerce_monotonic=True).fit_transform(X,y)         
    binSelector(method='pretty',n_jobs=1,iv_limit=0,sample_weight=ws,keep=['a','c']).fit_transform(X,y)      
    binSelector(method='pretty',n_jobs=1,iv_limit=0,special_values=[1,2,3,4,'a']).fit_transform(X,y)   


@mock.patch('matplotlib.pyplot.show')
def test_binAdjuster(mock_show,monkeypatch):

    X=pd.DataFrame(
        {
         'a':np.arange(100),
         'c':np.concatenate([np.repeat('a',50),np.repeat('b',50)],dtype=object)}
        )

    y=pd.Series(np.append(np.zeros(50),np.ones(50)),name='y')

    br_raw={'a':[40,80],'c':['a','b']}  
    
    number_inputs = StringIO('1\n1')
    monkeypatch.setattr('sys.stdin', number_inputs)
    adj=binAdjuster(br_raw).fit(X,y)
    res=adj.transform(X)
    assert hasattr(adj,'breaks_list_adj')
    assert hasattr(adj,'vtabs_dict_adj')
    assert all(np.equal(res.columns,['a','c']))
    
    
    number_inputs = StringIO('2\n40,50\n1\n2\na%,%b\n1')
    monkeypatch.setattr('sys.stdin', number_inputs)
    adj=binAdjuster(br_raw).fit(X,y)
    assert all(np.equal(adj.breaks_list_adj['a'],[40,50]))
    assert adj.breaks_list_adj['c'][0]=='a%,%b'
    
    number_inputs = StringIO('1\n3\n2\n20,70\n1\n1')
    monkeypatch.setattr('sys.stdin', number_inputs)
    adj=binAdjuster(br_raw).fit(X,y)
    assert all(np.equal(adj.breaks_list_adj['a'],[20,70]))
    
    number_inputs = StringIO('1\n4')
    monkeypatch.setattr('sys.stdin', number_inputs)
    adj=binAdjuster(br_raw).fit(X,y)
    assert 'a' in adj.breaks_list_adj.keys() and 'c' not in adj.breaks_list_adj.keys()   

    number_inputs = StringIO('0\ny')
    monkeypatch.setattr('sys.stdin', number_inputs)
    adj=binAdjuster(br_raw).fit(X,y)
        
    X=pd.DataFrame(
        {
         'a':np.arange(100),
         'c':np.concatenate([np.repeat('a',50),np.repeat('b',50)],dtype=object),
         'g':np.concatenate([np.repeat('a',30),np.repeat('b',30),np.repeat('c',40)],dtype=object)}    
        )

    y=pd.Series(np.append(np.zeros(50),np.ones(50)),name='y')

    br_raw={'a':[40,80],'c':['a','b']}   
    
    number_inputs = StringIO('1\n1')
    monkeypatch.setattr('sys.stdin', number_inputs)
    adj=binAdjuster(br_raw,column='g',sort_column=['a','b','c']).fit(X,y)
    res=adj.transform(X)  
    assert hasattr(adj,'breaks_list_adj')
    assert hasattr(adj,'vtabs_dict_adj')
    assert all(np.equal(res.columns,['a','c']))
    
    number_inputs = StringIO('2\n40,50\n1\n2\na%,%b\n1')
    monkeypatch.setattr('sys.stdin', number_inputs)
    adj=binAdjuster(br_raw,column='g',sort_column=['a','b','c']).fit(X,y)
    assert all(np.equal(adj.breaks_list_adj['a'],[40,50]))
    assert adj.breaks_list_adj['c'][0]=='a%,%b'
    
    number_inputs = StringIO('1\n3\n2\n20,70\n1\n1')
    monkeypatch.setattr('sys.stdin', number_inputs)
    adj=binAdjuster(br_raw,column='g',sort_column=['a','b','c']).fit(X,y)
    assert all(np.equal(adj.breaks_list_adj['a'],[20,70]))
    
    number_inputs = StringIO('1\n4')
    monkeypatch.setattr('sys.stdin', number_inputs)
    adj=binAdjuster(br_raw,column='g',sort_column=['a','b','c']).fit(X,y)
    assert 'a' in adj.breaks_list_adj.keys() and 'c' not in adj.breaks_list_adj.keys()   

    number_inputs = StringIO('0\ny')
    monkeypatch.setattr('sys.stdin', number_inputs)
    adj=binAdjuster(br_raw,column='g',sort_column=['a','b','c']).fit(X,y)
        
    
@mock.patch('matplotlib.pyplot.show')   
def test_faSelector(mock_show):

    X=pd.DataFrame(
        {'a':[1,2,2,4,5],'b':[1,2,3,4,5],'c':[1,1,1,2,1]}
        )
    y=pd.Series([0,0,1,1,1],name='y')
    
    by=pd.Series([3,2,1],index=['a','b','c'],name='ws')
    
    fa=faSelector(n_clusters=2,distance_metrics='r2').fit(X,y)
    fa=faSelector(n_clusters=2,distance_metrics='spearman').fit(X,y)
    fa=faSelector(n_clusters='auto').fit(X,y)    
    fa=faSelector(n_clusters='auto').fit(X,y)
    fa=faSelector(distance_threshold=0.1).fit(X,y)
    fa=faSelector(n_clusters=2).fit(X,y)
    fa.plot_dendrogram(X)
    res=fa.transform(X)
    
    assert hasattr(fa,'components_infos')
    assert hasattr(fa,'rsquare_infos')
    assert all(np.equal(res.columns,['a','c']))
    
    res=faSelector(n_clusters=2,by=by).fit_transform(X,y)
    assert all(np.equal(res.columns,['a','c']))
    
    res=faSelector(n_clusters=2,by=by,is_greater_better=False).fit_transform(X,y)
    assert all(np.equal(res.columns,['b','c']))


def test_prefitModel():
    
    X=pd.DataFrame(
        {'a':[1,2,2,4,5],'b':[1,2,3,4,5],'c':[1,1,1,2,1],'d':['a','a','a','a','b']}
        )
    y=pd.Series([0,0,1,1,1],name='y')

    pm=prefitModel(
        tree_params={'max_depth': 2, 'learning_rate': 0.05, 'n_estimators': 10,'n_jobs':1},
        method='ceiling',
                   ).fit(X,y)    
    
    pm=prefitModel(method='floor',max_iter=10).fit(X,y)    
    
    pm=prefitModel(method='floor',col_rm='a').fit(X,y)    
    
    pm.predict_proba(X)
    

def test_preSelector():
    
    X=pd.DataFrame(
        {
         'a':np.arange(100),
         'b':np.append(np.repeat(3,30),np.repeat(4,70)),
         'c':np.concatenate([np.repeat('a',50),np.repeat('b',50)],dtype=object)}
        )
    
    ws=pd.Series(np.ones(100),name='ws')
    
    y=pd.Series(np.append(np.zeros(50),np.ones(50)),name='y')
    
    res=preSelector().fit_transform(X,y)   
    res=preSelector().fit(X,y,cat_features=['b'])   
    res=preSelector().fit(X,y,sample_weight=ws).transform(X)   
    res=preSelector(keep=['a']).fit_transform(X,y)  
    assert 'a' in res.columns
    res=preSelector(out_path='tmp').fit_transform(X,y) 

def test_lassologit():
    
    from sklearn.datasets import load_breast_cancer
    import matplotlib
    
    X=pd.DataFrame(
        load_breast_cancer()['data'],columns=['v'+str(i) for i in range(30)]
        )[['v22','v24','v21','v28','v15','v6','v11','v13','v27','v14']]
    y=pd.Series(load_breast_cancer()['target'],name='y')
    ws=pd.Series(np.ones(y.size),index=y.index)
    
    X=X[['v22','v24','v21','v28']]
    
    res=LassoLogit(c_num=10,standard=True,metric='roc_auc',n_jobs=1).fit_plot(X,y)
    
    assert isinstance(res.plot_path,matplotlib.figure.Figure)
    assert isinstance(res.plot_valscore,matplotlib.figure.Figure)
    assert hasattr(res,'cv_path_')
    assert hasattr(res,'cv_res_')
    assert hasattr(res,'feature_names_')
    assert hasattr(res,'model_refit')
    
    res.predict_proba(X)
    res.transform(X)
    
    res=LassoLogit(c_num=10,standard=True,metric='neglogloss',n_jobs=1).fit_plot(X,y)
    
    assert isinstance(res.plot_path,matplotlib.figure.Figure)
    assert isinstance(res.plot_valscore,matplotlib.figure.Figure)
    assert hasattr(res,'cv_path_')
    assert hasattr(res,'cv_res_')
    assert hasattr(res,'feature_names_')
    assert hasattr(res,'model_refit')
    
    res.predict_proba(X)
    res.transform(X)    
    
    res.refit_with_C(X, y, C=1)
    assert res.model_refit.C==1
    
    res=LassoLogit(c_num=10,standard=True,metric='neglogloss',n_jobs=1).fit_plot(X,y,ws)
    
    assert isinstance(res.plot_path,matplotlib.figure.Figure)
    assert isinstance(res.plot_valscore,matplotlib.figure.Figure)
    assert hasattr(res,'cv_path_')
    assert hasattr(res,'cv_res_')
    assert hasattr(res,'feature_names_')
    assert hasattr(res,'model_refit')    
    
    res.predict_proba(X)
    res.transform(X) 
    

def test_LgbmSeqSelector():
    
    X=pd.DataFrame(
        {'a':[1,2,2,4,5],'b':[1,2,3,4,5],'c':[1,1,1,2,1]}
        )
    y=pd.Series([0,0,1,1,1],name='y')
    
    res = LgbmSeqSelector(k_features=1,n_jobs=1,cv=2,forward=False,floating=False).fit_transform(X,y)
    res = LgbmSeqSelector(k_features=1,n_jobs=1,cv=2,forward=True,floating=False).fit_transform(X,y)
    res = LgbmSeqSelector(k_features=1,n_jobs=1,cv=2,forward=False,floating=True).fit_transform(X,y)    
    res = LgbmSeqSelector(k_features=1,n_jobs=1,cv=2,forward=False,floating=False).fit_transform(X,y)
    assert 'a' in res.columns
    
    seq = LgbmSeqSelector(k_features=1,n_jobs=1,cv=2,forward=False,floating=False).fit(X,y)
    seq.plot()
    

def test_LgbmShapRFECVSelector():
    
    X=pd.DataFrame(
        np.random.rand(100,4),columns=['a','b','c','d']
        )
    y=pd.Series(np.random.randint(0,2,100),name='y')
    
    LgbmShapRFECVSelector(n_jobs=1,cv=2).fit_transform(X,y,check_additivity=False)    
    LgbmShapRFECVSelector(n_jobs=1,cv=2,early_stopping_rounds=10).fit_transform(X,y,check_additivity=False)   
    LgbmShapRFECVSelector(n_jobs=1,cv=2,method='bs').fit_transform(X,y,check_additivity=False)       
    LgbmShapRFECVSelector(n_jobs=1,cv=2).fit(X,y).plot()   
    

def test_LgbmPISelector():
    
    X=pd.DataFrame(
        {'a':np.arange(50)}
        )
    y=pd.Series(np.append(np.zeros(25),np.ones(25)),name='y')
    
    res = LgbmPISelector(cv=2,n_jobs=1,validation_fraction=0.1).fit_transform(X,y)
    res = LgbmPISelector(cv=2,n_jobs=1,validation_fraction=None,early_stopping_rounds=10).fit_transform(X,y)
    res = LgbmPISelector(threshold=0.1,cv=2,n_jobs=1,early_stopping_rounds=None).fit_transform(X,y)
    res = LgbmPISelector(threshold=0.1,cv=2,n_jobs=1,method='bs',early_stopping_rounds=10).fit_transform(X,y)
    res = LgbmPISelector(threshold=0.1,cv=2,n_jobs=1,method='bs',early_stopping_rounds=None).fit_transform(X,y)
    res = LgbmPISelector(threshold=0.1,cv=2,n_jobs=1,method='bs',early_stopping_rounds=None,validation_fraction=None).fit_transform(X,y)
    
    assert 'a' in res
    
    
def test_stepLogit():
    
    from sklearn.datasets import load_breast_cancer
    
    X=pd.DataFrame(
        load_breast_cancer()['data'],columns=['v'+str(i) for i in range(30)]
        )[['v22','v24','v21','v28','v15','v6','v11','v13','v27','v14']]
    y=pd.Series(load_breast_cancer()['target'],name='y')
    cols=['v22','v24','v21','v28','v15','v6','v11','v13','v27','v14']
    
    res=stepLogit(method='no_stepwise',show_step=True).fit_transform(X,y)
    assert set(res.columns)==set(res)
    res=stepLogit(method='forward',show_step=True).fit_transform(X,y)
    assert set(res.columns)==set(['v22','v24','v21','v28','v15','v6','v11','v13','v27'])
    res=stepLogit(method='backward',show_step=True).fit_transform(X,y)
    assert set(res.columns)==set(['v22','v24','v21','v28','v15','v6','v11','v13'])
    res=stepLogit(method='both',show_step=True).fit_transform(X,y)
    assert set(res.columns)==set(['v22','v24','v21','v28','v15','v6','v11','v13'])
    
    res=stepLogit(method='forward',show_step=True,criterion='bic').fit_transform(X,y)
    assert set(res.columns)==set(['v22','v24','v21','v21','v13','v28'])
    res=stepLogit(method='backward',show_step=True,criterion='bic').fit_transform(X,y)
    assert set(res.columns)==set(['v22','v24','v21','v28','v15','v6'])
    res=stepLogit(method='both',show_step=True,criterion='bic').fit_transform(X,y)
    assert set(res.columns)==set(['v22','v24','v21','v28','v13'])
    
    res=stepLogit(custom_column=cols,method='no_stepwise',show_step=True).fit_transform(X,y)
    assert set(res.columns)==set(res)
    res=stepLogit(custom_column=cols,method='forward',show_step=True).fit_transform(X,y)
    assert set(res.columns)==set(['v22','v24','v21','v28','v15','v6','v11','v13','v27'])
    res=stepLogit(custom_column=cols,method='backward',show_step=True).fit_transform(X,y)
    assert set(res.columns)==set(['v22','v24','v21','v28','v15','v6','v11','v13'])
    res=stepLogit(custom_column=cols,method='both',show_step=True).fit_transform(X,y)
    assert set(res.columns)==set(['v22','v24','v21','v28','v15','v6','v11','v13'])
    
    res=stepLogit(custom_column=cols,method='forward',show_step=True,criterion='bic').fit_transform(X,y)
    assert set(res.columns)==set(['v22','v24','v21','v21','v13','v28'])
    res=stepLogit(custom_column=cols,method='backward',show_step=True,criterion='bic').fit_transform(X,y)
    assert set(res.columns)==set(['v22','v24','v21','v28','v15','v6'])
    res=stepLogit(custom_column=cols,method='both',show_step=True,criterion='bic').fit_transform(X,y)
    assert set(res.columns)==set(['v22','v24','v21','v28','v13'])
    
    res=stepLogit(custom_column=['v22','v24'],method='forward',show_step=True,criterion='bic').fit_transform(X,y)
    res=stepLogit(custom_column=['v22','v24'],method='backward',show_step=True,criterion='bic').fit_transform(X,y)
    res=stepLogit(custom_column=['v22','v24'],method='both',show_step=True,criterion='bic').fit_transform(X,y)
    
    ws=pd.Series(np.ones(y.size),index=y.index)
    res=stepLogit(method='no_stepwise',show_step=True).fit(X,y,ws).transform(X)
    assert set(res.columns)==set(res)
    res=stepLogit(method='forward',show_step=True).fit(X,y,ws).transform(X)
    assert set(res.columns)==set(['v22','v24','v21','v28','v15','v6','v11','v13','v27'])
    res=stepLogit(method='backward',show_step=True).fit(X,y,ws).transform(X)
    assert set(res.columns)==set(['v22','v24','v21','v28','v15','v6','v11','v13'])
    res=stepLogit(method='both',show_step=True).fit(X,y,ws).transform(X)
    assert set(res.columns)==set(['v22','v24','v21','v28','v15','v6','v11','v13'])
    
    stepLogit(method='no_stepwise',show_step=True).fit(X,y,ws).predict_proba(X)


def test_woeTransformer():   
    
    from sklearn.datasets import load_breast_cancer
    import numpy as np
    
    X=pd.DataFrame(load_breast_cancer()['data'],columns=['var_'+str(i) for i in range(30)])
    y=pd.Series(load_breast_cancer()['target'],name='y')
    X['var_31']=pd.Series(np.random.randint(0,5,y.size),name='var_31',dtype='str')
    X['var_32']=pd.Series(np.random.randint(0,5,y.size-20).tolist()+np.repeat(np.nan,20).tolist(),name='var_32')

    bins=binSelector(n_jobs=1,iv_limit=0,bin_num_limit=10).fit(X,y).bins
    
    X_bin=binTransformer(bins).fit_transform(X,y) 
    
    X_woe=woeTransformer(bins,woe_missing=0,distr_limit=0).fit_transform(X,y)    
    
    from category_encoders.woe import WOEEncoder
    
    X_woe1=WOEEncoder(regularization=1e-10).fit_transform(X_bin,y)
    
    for col in X_woe.columns:
        
        assert X_woe[col].round(4).equals(X_woe1[col].round(4))
        
    X_woe=woeTransformer(bins,woe_missing=0,distr_limit=0.05,woe_special=0).fit_transform(X,y)  
    assert all(X_woe['var_32'][X['var_32'].isnull()]==0)
    
    X=pd.DataFrame({'var_num':[0,1,1,3,np.nan,4,5,1,0,1],
                    'var_char':['a','a','a','b','b','missing','c','c','d','d']})
    y=pd.Series([1,0,1,1,1,0,0,0,1,1],name='y')
    bins=binSelector(n_jobs=1,iv_limit=0,bin_num_limit=5,
                     special_values={'var_num':[1],'var_char':['a']}).fit(X,y).bins
    
    X_bin=binTransformer(bins,special_values={'var_num':[1],'var_char':['a']}).fit_transform(X,y) 
    assert all(X_bin['var_num'][X['var_num']==1]=='special')
    assert all(X_bin['var_num'][X['var_num'].isnull()]=='missing')
    assert all(X_bin['var_char'][X['var_char']=='a']=='special')
    assert all(X_bin['var_char'][X['var_char'].isnull()]=='missing')
    
    
def test_cardScorer():
    
    X=pd.DataFrame(
        {'a':[1,2,2,2,1],
         'b':[np.nan,200,100,250,np.nan],
         'c':['a','b','b','c','a']}
        )
    y=pd.Series([0,0,0,0,1],name='y')

    bins=binSelector(n_jobs=1,iv_limit=0).fit(X,y).bins
    
    X_woe=woeTransformer(bins,woe_missing=0,distr_limit=0.5).fit_transform(X,y)
    assert X_woe['b'][0]==0 and X_woe['b'][4]==0
    
    X_woe=woeTransformer(bins).fit_transform(X,y)
    
    lm=stepLogit(method='no_stepwise',show_step=True).fit(X_woe,y) 
    
    res=cardScorer(lm.logit_model,bins,check_na=False).fit(X)
    res=cardScorer(lm.logit_model,bins).fit(X)
    
    assert hasattr(res,'scorecard')
    
    res.transform(X)
    
    lm=LogisticRegression().fit(X_woe,y) 
    res=cardScorer(lm,bins,check_na=False).fit(X)
    res=cardScorer(lm,bins).fit(X)
    
    assert hasattr(res,'scorecard')
    
    res.transform(X)
    