#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:34:22 2022

@author: zengke
"""

from BDMLtools.clearner import outliersTransformer,dtypeAllocator,nanTransformer,dtStandardization
import pandas as pd
import numpy as np


def test_dtStandardization():


    dt=pd.DataFrame(
        {'a':[1,2,2],'b':[1,2,3],'c':[1,2,3]},
        index=[0,1,2]
        ).rename(columns={'c':'b'})
    
    res=dtStandardization(id_col=['a']).fit_transform(dt)
    
    assert len(res)==2
    assert all(np.equal(res.columns,['b'])) 
    assert res.index.name=='a'
    
    
def test_dtypeAllocator():
    
    dt=pd.DataFrame(
        {'a':[1,2,3],'b':['1','2','3'],
         'c':['2021-09-01','2022-08-01','2011-12-11'],
         'd':[True,True,False]
         }
        )
    
    res=dtypeAllocator({'float':['a','d'],'str':['b'],'date':['c']}).fit_transform(dt)
    
    assert all(np.equal(res.dtypes.tolist(),[np.dtype('float64'),
                                             np.dtype('float64'), 
                                             np.dtype('O'),
                                             np.dtype('<M8[ns]')]))
    
def test_nanTransformer():
    
    dt=pd.DataFrame(
        {'a':[np.nan,-np.inf,np.inf],'b':['','missing','nan']}
        )
    
    res=nanTransformer(fill_value=(np.nan, 'missing')).fit_transform(dt)
    
    assert all(res['a'].isnull())
    assert all(res['b'].isnull())
    

def test_outliersTransformer():
    
    dt=pd.DataFrame(
        {'a':pd.Series(np.ones(100))}
        )
    
    dt.loc[99]=100;dt.loc[0]=-100
    
    res=outliersTransformer(method='nan').fit_transform(dt)
    
    assert(pd.isnull(res.loc[0,'a']))
    assert(pd.isnull(res.loc[99,'a']))
    
    
    