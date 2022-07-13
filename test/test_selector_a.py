#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:34:22 2022

@author: zengke
"""

from BDMLtools.selector import binSelector,faSelector,preSelector,prefitModel
import pandas as pd

def test_binSelector():

    X=pd.DataFrame(
        {'a':[1,2,2],'b':[1,2,3]}
        )
    
    y=pd.Series([0,0,1],name='y')
    
    binSelector(method='freq').fit(X,y)
    
    binSelector(method='tree',n_jobs=1,iv_limit=0).fit(X,y)    
    
    binSelector(method='tree',n_jobs=1,iv_limit=0,coerce_monotonic=True).fit(X,y) 
    
    binSelector(method='chi2',n_jobs=1,iv_limit=0).fit(X,y)    
    
    binSelector(method='chi2',n_jobs=1,iv_limit=0,coerce_monotonic=True).fit(X,y)    
    
    
def test_faSelector():

    X=pd.DataFrame(
        {'a':[1,2,2,4,5],'b':[1,2,3,4,5],'c':[1,1,1,2,1]}
        )
    y=pd.Series([0,0,1,1,1],name='y')
    
    res=faSelector(n_clusters='auto').fit(X,y)
    res=faSelector(n_clusters=2).fit(X,y)
    
    assert hasattr(res,'components_infos')
    assert hasattr(res,'rsquare_infos')
    
    
def test_preSelector():
    
    X=pd.DataFrame(
        {'a':[1,2,2,4,5],'b':[1,2,3,4,5],'c':[1,1,1,2,1]}
        )
    y=pd.Series([0,0,1,1,1],name='y')
    
    preSelector().fit(X,y)
    

def test_prefitModel():
    
    X=pd.DataFrame(
        {'a':[1,2,2,4,5],'b':[1,2,3,4,5],'c':[1,1,1,2,1]}
        )
    y=pd.Series([0,0,1,1,1],name='y')

    prefitModel().fit(X,y)    
    

