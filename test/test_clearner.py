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
    
    res=dtStandardization(id_col=['a'],col_rm='b').fit_transform(dt)
    assert res.columns.size==0
    
    res=dtStandardization(col_rm='b',downcast=True).fit_transform(dt)
    assert all(np.equal(res.columns,['a'])) 
    assert res.dtypes['a']==np.dtype('float32')
    
    res=dtStandardization(id_col=['a'],drop_dup=False).fit_transform(dt)
    assert all(np.equal(res.columns,['b','b'])) 
    assert all(np.equal(res.index,[1,2,2])) 
    
    
    
def test_dtypeAllocator():
    
    dt=pd.DataFrame(
        {'a':[1,2,3],'b':[1,2,3],
         'c':['2021-09-01','2022-08-01','2011-12-11'],
         'd':[True,True,False],
         'e':[pd.to_timedelta('4 days'),pd.to_timedelta('5 days'),pd.to_timedelta('6 days')],
         'f':[1,2,3]
         }
        )
    
    res=dtypeAllocator({'int':['f'],'float':['a','d'],'str':['b'],'date':['c'],'tdiff':['e']}).fit_transform(dt)
    
    assert all(np.equal(res.dtypes.tolist(),[np.dtype('int64'),
                                             np.dtype('float64'),
                                             np.dtype('float64'), 
                                             np.dtype('O'),
                                             np.dtype('<M8[ns]'),
                                             np.dtype('float64')]))
    
    res=dtypeAllocator().fit_transform(dt)
    
    assert all(np.equal(res.dtypes.tolist(),[np.dtype('int64'),
                                             np.dtype('int64'),
                                             np.dtype('int64'), 
                                             np.dtype('int64'),
                                             np.dtype('O'),
                                             np.dtype('float64')]))   
    
    res=dtypeAllocator({'int':['f'],'float':['a','d'],'str':['b'],'date':['c'],'tdiff':['e']},dtype_num='32',drop_date=True,precision=3).fit_transform(dt)
    
    assert all(np.equal(res.dtypes.tolist(),[np.dtype('int32'),
                                             np.dtype('float32'),
                                             np.dtype('float32'), 
                                             np.dtype('O'),
                                             np.dtype('float32')]))
    
    assert 'c' not in res.columns
    
    
def test_nanTransformer():
    
    dt=pd.DataFrame(
        {'a':[np.nan,-np.inf,np.inf],'b':['','missing','nan']}
        )
    
    res=nanTransformer(fill_value=(np.nan, 'missing')).fit_transform(dt)

    assert all(res['a'].isnull())
    assert all(res['b'].isnull())
    
    dt=pd.DataFrame(
        {'a':[np.nan,-np.inf,np.inf,1],'b':['','missing','nan','mode']}
        )
    
    res=nanTransformer(method=('mean','most_frequent'),fill_value=(np.nan, 'missing')).fit_transform(dt)   
    
    assert all(res['a']==1)
    assert all(res['b']=='mode')
    
    
    res=nanTransformer(method=('mean','most_frequent'),fill_value=(np.nan, 'missing')).fit_transform(dt)   
    
    assert all(res['a']==1)
    assert all(res['b']=='mode')
    
    res=nanTransformer(method=('mean','most_frequent'),indicator=True,fill_value=(np.nan, 'missing')).fit_transform(dt)   
    
    assert all(np.equal(res.columns,['a','b','a_isnan','b_isnan']))
    assert all(np.equal(res['a_isnan'],[1,1,1,0]))
    assert all(np.equal(res['b_isnan'],[1,1,1,0]))
    
    
    res=nanTransformer(method=('knn','most_frequent'),fill_value=(np.nan, 'missing'),dtype_num='float32').fit_transform(dt)  
    assert all(np.equal(res['a'],[1.,1.,1.,1.]))
    assert res['a'].dtype==np.dtype('float32')
    
    

def test_outliersTransformer():
    
    dt=pd.DataFrame(
        {'a':pd.Series(np.ones(100))}
        )
    
    dt.loc[99]=100;dt.loc[0]=-100
    
    res=outliersTransformer(method='nan').fit_transform(dt)
    
    assert(pd.isnull(res.loc[0,'a']))
    assert(pd.isnull(res.loc[99,'a']))
    
    dt=pd.DataFrame(
        {'a':pd.Series(np.arange(100))}
        )
    
    dt.loc[99]=10000;dt.loc[0]=-10000
    
    res=outliersTransformer(method='fill').fit_transform(dt)
    
    assert(res.loc[0,'a']==-123.75)
    assert(res.loc[99,'a']==222.75)
    
    
    