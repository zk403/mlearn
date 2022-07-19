#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:46:04 2022

@author: zengke
"""

from BDMLtools.base import Base,BaseEval
import pandas as pd
import numpy as np


def test_base():
    
    X=pd.DataFrame(
            {
             'a':[1,2,2,4,5,1,2,2,4,5,1,2,2,4,5,1,2,2,4,5],
             'b':[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5],
             'c':['1','1','1','2','1','1','1','1','2','1','1','1','1','2','1','1','1','1','2','1']},
            )
    y=pd.Series([0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1],name='y')
    ws=pd.Series(np.ones(20),name='ws')
    
    res=Base()
    res._is_fitted=True
    
    res._check_is_fitted()
    res._check_x(y)
    res._check_ind([X, y])
    res._check_X(X)
    res._check_colname(X)
    res._check_data(X, y)
    res._check_ws(y,ws)
    res._check_param_dtype('float32')
    res._check_yname(y)
    
    
def test_baseeval():
    
    show_plot=('ks', 'lift', 'gain', 'roc', 'lz', 'pr', 'f1', 'density')
    pred_desc=True
    
    res=BaseEval()
    res._check_plot_params(show_plot,pred_desc)
    
    pred=pd.Series([100,130,70,150,160],name='pred',dtype='float')
    true=pd.Series([0,0,1,0,1],name='true',dtype='int')
    group=pd.Series([1,1,1,1,1],name='group')
    ws=pd.Series([0.22,0.34,0.78,0.13,0.54],name='pred',dtype='float')
    
    res._check_params(pred,true,group,ws)
    res._check_values(pred,true,group,ws)