#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:46:04 2022

@author: zengke
"""

from BDMLtools.base import Base,BaseEval
from sklearn.exceptions import NotFittedError
from BDMLtools.exception import DataTypeError,XyIndexError,yValueError
from BDMLtools.fun import Specials
import pandas as pd
import numpy as np

def test_base():
    

    #_check_is_fitted
    res=Base()
    res._is_fitted=True
    res._check_is_fitted()
    
    try:
        res._is_fitted=False  
        res._check_is_fitted()
    except NotFittedError:
        pass
    
    #_check_x
    res._check_x(pd.Series([0,0,1],name='y'))
    
    try:
        res._check_x([1,2,3])
    except DataTypeError:
        pass
    
    try:
        res._check_x(pd.Series([0,0,1],name='y',dtype='float32'))
    except DataTypeError:
        pass
    
    #_check_ind
    X=pd.DataFrame({'a':[1,2,2]})
    y=pd.Series([0,0,1],name='y')
    
    res._check_ind([X, y])
    try:
        X.index=[1,2,2]
        res._check_ind([X, y])
    except XyIndexError:
        pass
    
    try:
        y.index=[1,2,2]
        res._check_ind([X, y])
    except XyIndexError:
        pass
    
    #_check_X
    X=pd.DataFrame({'a':[1,2,2]})
    res._check_X(X)
    
    try:
        res._check_X(X['a'])
    except DataTypeError:
        pass    

    try:
        X.index=[1,2,2]
        res._check_X(X)
    except XyIndexError:
        pass   
    
    try:
        X.index=[1,2,3];X=X.astype('float16')
        res._check_X(X)
    except DataTypeError:
        pass      
    
    #_check_colname
    X=pd.DataFrame({'a':[1,2,2],'b':[1,2,3]})
    res._check_colname(X)
    
    try:
        X.columns=['a','a']
        res._check_colname(X)
    except XyIndexError:
        pass   
    
    #_check_data
    X=pd.DataFrame({'a':[1,2,2]})
    y=pd.Series([0,1,0],name='y')
    res._check_data(X, y)
    
    try:
        res._check_data(y, y)
    except DataTypeError:
        pass
    
    try:
        res._check_data(X, X)
    except DataTypeError:
        pass
    
    try:
        X.index=[1,2,2]
        res._check_data(X, y)
    except XyIndexError:
        pass    

    try:
        X.index=[1,2,3];y.index=[1,2,2]
        res._check_data(X, y)
    except XyIndexError:
        pass    
    
    try:
        X.index=[1,2,3];y.index=[1,2,4]
        res._check_data(X, y)
    except XyIndexError:
        pass    
    
    try:
        X.index=[0,1,2]
        y=pd.Series([0,1,2],name='y')
        res._check_data(X, y)
    except yValueError:
        pass  
    
    try:
        X=X.astype('float32')
        y=pd.Series([0,1,1],name='y')
        res._check_data(X, y)
    except DataTypeError:
        pass  
    
    #_check_ws    
    y=pd.Series([0,1,1],name='y')
    ws=pd.Series([1,1,1],name='ws')
    res._check_ws(y,ws)
    
    try:
        res._check_ws(y, [1,2,3])
    except DataTypeError:
        pass 
    
    try:
        ws=pd.Series([1,1,1],name='ws',dtype='float32')
        res._check_ws(y, ws)
    except DataTypeError:
        pass 
    
    try:
        ws=pd.Series([1,1,1],name='ws',index=[1,2,3])
        res._check_ws(y, ws)
    except XyIndexError:
        pass     
    
    #_check_yname
    res._check_yname(y)
    
    try:
        y=pd.Series([1,1,1],index=[1,2,3])
        res._check_yname(y)
    except ValueError:
        pass         
    
    
def test_baseeval():
    
    #_check_plot_params
    show_plot=('ks', 'lift', 'gain', 'roc', 'lz', 'pr', 'f1', 'density')
    pred_desc=True
    
    res=BaseEval()
    res._check_plot_params(show_plot,pred_desc)
    
    try:
        show_plot=['ks', 'lift', 'gain', 'roc', 'lz', 'pr', 'f1', 'density']
        res._check_plot_params(show_plot,pred_desc)
    except ValueError:
        pass
    
    try:
        show_plot=('kss', 'lift', 'gain', 'roc', 'lz', 'pr', 'f1', 'density')
        res._check_plot_params(show_plot,pred_desc)
    except ValueError:
        pass
    
    try:
        show_plot=('ks', 'lift', 'gain', 'roc', 'lz', 'pr', 'f1', 'density')
        pred_desc='True'
        res._check_plot_params(show_plot,pred_desc)
    except ValueError:
        pass
    
    
    #_check_params    
    pred=pd.Series([100,130,70,150,160],name='pred',dtype='float')
    true=pd.Series([0,0,1,0,1],name='true',dtype='int')
    group=pd.Series([1,1,1,1,1],name='group')
    ws=pd.Series([0.22,0.34,0.78,0.13,0.54],name='pred',dtype='float')
    
    res._check_params(pred,true,group,ws)
    
    try:
        res._check_params([],true,group,ws)
    except ValueError:
        pass  
    
    try:
        res._check_params(pred,[],group,ws)
    except ValueError:
        pass  
    
    try:
        res._check_params(pred,true,[1,2,3],ws)
    except ValueError:
        pass  
    
    try:
        res._check_params(pred,true,group,[])
    except ValueError:
        pass  
    
    try:
        pred.index=[1,2,3,4,5]
        res._check_params(pred,true,group,ws)
    except XyIndexError:
        pass  
    

    try:
        pred.index=true.index
        group.index=[1,2,3,4,5]
        res._check_params(pred,true,group,ws)
    except XyIndexError:
        pass  
    
    try:
        group.index=true.index
        ws.index=[1,2,3,4,5]
        res._check_params(pred,true,group,ws)
    except XyIndexError:
        pass  
    
    #_check_values    
    pred=pd.Series([100,130,70,150,160],name='pred',dtype='float')
    true=pd.Series([0,0,1,0,1],name='true',dtype='int')
    group=pd.Series([1,1,1,1,1],name='group')
    ws=pd.Series([0.22,0.34,0.78,0.13,0.54],name='pred',dtype='float')

    res._check_values(pred,true,group,ws)
    
    try:
        pred=pd.Series([np.nan,130,70,150,160],name='pred',dtype='float')
        res._check_values(pred,true,group,ws)
    except ValueError:
        pass      
    
    try:
        pred=pd.Series([100,130,70,150,160],name='pred',dtype='float')
        true=pd.Series([np.nan,0,1,0,1],name='true',dtype='float')
        res._check_values(pred,true,group,ws)
    except ValueError:
        pass      
    
    try:
        pred=pd.Series([100,130,70,150,160],name='pred',dtype='float')
        true=pd.Series([0,0,1,0,1],name='true',dtype='int')
        group=pd.Series([np.nan,1,1,1,1],name='group')
        ws=pd.Series([0.22,0.34,0.78,0.13,0.54],name='pred',dtype='float')
        res._check_values(pred,true,group,ws)
    except ValueError:
        pass      
    
    
    try:
        pred=pd.Series([100,130,70,150,160],name='pred',dtype='float')
        true=pd.Series([0,0,1,2,1],name='true',dtype='int')
        group=pd.Series([1,1,1,1,1],name='group')
        ws=pd.Series([0.22,0.34,0.78,0.13,0.54],name='pred',dtype='float')
        res._check_values(pred,true,group,ws)
    except ValueError:
        pass  


    try:
        pred=pd.Series([100,130,70,150,160],name='pred',dtype='float')
        true=pd.Series([0,0,1,1,1],name='true',dtype='int')
        group=pd.Series([1,1,1,1,1],name='group')
        ws=pd.Series([0.34,0.78,0.13,0.54],name='pred',dtype='float')
        res._check_values(pred,true,group,ws)
    except ValueError:
        pass      
    
    try:
        pred=pd.Series([100,130,70,150,160],name='pred',dtype='float')
        true=pd.Series([0,0,1,1,1],name='true',dtype='int')
        group=pd.Series([1,1,1,1,1],name='group')
        ws=pd.Series([-1,0.34,0.78,0.13,0.54],name='pred',dtype='float')
        res._check_values(pred,true,group,ws)
    except ValueError:
        pass      
    
    try:
        pred=pd.Series([100,130,70,150,160],name='pred',dtype='float')
        true=pd.Series([0,0,1,1,1],name='true',dtype='int')
        group=pd.Series([1,1,1,1,1],name='group')
        ws=pd.Series([np.nan,0.34,0.78,0.13,0.54],name='pred',dtype='float')
        res._check_values(pred,true,group,ws)
    except ValueError:
        pass      
    
    try:
        pred=pd.Series([100,70,150,160],name='pred',dtype='float')
        true=pd.Series([0,0,1,1,1],name='true',dtype='int')
        group=pd.Series([1,1,1,1,1],name='group')
        ws=pd.Series([-1,0.34,0.78,0.13,0.54],name='pred',dtype='float')
        res._check_values(pred,true,group,ws)
    except ValueError:
        pass      
    
    
def test_specials():

    res=Specials()    
    
    X=pd.DataFrame({'a':[1.,2.,2.],'b':['1','2','3']})
    X_1=res._sp_replace_df(X,[1,'1'])
    assert X_1.loc[0,'a']==np.finfo(np.float32).max
    assert X_1.loc[0,'b']=='special'
    
    X=pd.DataFrame({'a':[1.,2.,2.],'b':['1','2','3']})
    X_1=res._sp_replace(X,[1,'1'])
    assert X_1.loc[0,'a']==np.finfo(np.float32).max
    assert X_1.loc[0,'b']=='special'
    
    X=pd.DataFrame({'a':[1.,2.,2.],'b':['1','2','3']})
    X_1=res._sp_replace(X,{'a':[1],'b':['1']})
    assert X_1.loc[0,'a']==np.finfo(np.float32).max
    assert X_1.loc[0,'b']=='special'
    
    assert res._check_spvalues('a',None) is None    
    assert np.equal(res._check_spvalues('a',[1]),[1]).all()
    assert np.equal(res._check_spvalues('a',{'a':[1],'b':['a']}),[1]).all()
    
    try:
        res._check_spvalues('a',(1)) 
    except ValueError:
        pass
    
    
    
    