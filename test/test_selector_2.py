#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:34:22 2022

@author: zengke
"""

from BDMLtools.selector import stepLogit,binSelector,cardScorer,LgbmSeqSelector,LgbmShapRFECVSelector,LgbmPISelector
from BDMLtools.encoder import woeTransformer
import pandas as pd
import numpy as np


def test_LgbmSeqSelector():
    
    X=pd.DataFrame(
        {'a':[1,2,2,4,5],'b':[1,2,3,4,5],'c':[1,1,1,2,1]}
        )
    y=pd.Series([0,0,1,1,1],name='y')
    
    LgbmSeqSelector(k_features=1,n_jobs=1,cv=2).fit(X,y)
    

def test_LgbmShapRFECVSelector():
    
    X=pd.DataFrame(
        np.random.rand(100,4),columns=['a','b','c','d']
        )
    y=pd.Series(np.random.randint(0,2,100),name='y')
    
    LgbmShapRFECVSelector(n_jobs=1,cv=2).fit(X,y,check_additivity=False)    
    

def test_LgbmPISelector():
    
    X=pd.DataFrame(
        np.random.rand(100,4),columns=['a','b','c','d']
        )
    y=pd.Series(np.random.randint(0,2,100),name='y')
    
    LgbmPISelector(cv=2,n_jobs=1,validation_fraction=0.2).fit(X,y)
    
    
def test_stepLogit():
    
    X=pd.DataFrame(
        {'a':[1,2,2,4,5],'b':[1,2,3,4,5],'c':[1,1,1,2,1]}
        )
    y=pd.Series([1,0,1,0,1],name='y')
    
    stepLogit(no_stepwise=False,show_step=True).fit(X,y)
    
    
def test_cardScorer():
    
    X=pd.DataFrame(
        {'a':np.ones(100)}
        )
    y=pd.Series(np.random.randint(0,2,100),name='y')
    
    bins=binSelector(n_jobs=1,iv_limit=0).fit(X,y).bins
    woe=woeTransformer(bins).fit(X,y)
    X_woe=woe.transform(X)
    
    lm=stepLogit(no_stepwise=True,show_step=True).fit(X_woe,y) 
    
    cardScorer(lm.logit_model,bins).fit(X)
    
    