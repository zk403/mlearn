#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:44:35 2022

@author: zengke
"""

from BDMLtools.plotter import  perfEval,perfEval2
from plotnine.ggplot import ggplot
import pandas as pd

def test_eval():
    
    pred_prob=pd.Series([0.22,0.34,0.78,0.13,0.54],name='pred',dtype='float')
    pred=pd.Series([100,130,70,150,160],name='pred',dtype='float')
    true=pd.Series([0,0,1,0,1],name='true',dtype='int')
    group=pd.Series([1,1,1,1,1],name='group')
    ws=pd.Series([0.2,0.3,0.1,0.9,0.5],name='ws')
    
    res=perfEval(show_plot=('ks','roc','gain','lift','roc','pr','lz','f1','density')).plot(pred,true)
    res=perfEval(show_plot=('ks','roc','gain','lift','roc','pr','lz','f1','density')).plot(pred_prob,true)
    res=perfEval(show_plot=('ks','roc','gain','lift','roc','pr','lz','f1','density')).plot(pred,true,group)
    res=perfEval(show_plot=('ks','roc','gain','lift','roc','pr','lz','f1','density')).plot(pred_prob,true,group)
    res=perfEval(show_plot=('ks','roc','gain','lift','roc','pr','lz','f1','density')).plot(pred_prob,true,group,ws)
    res=perfEval(title='test',pred_desc=True).plot(pred,true,group)

    
    assert all([isinstance(res[key],ggplot) for key in res])
    

def test_eval2():
    
    pred_prob=pd.Series([0.22,0.34,0.78,0.13,0.54],name='pred',dtype='float')
    pred=pd.Series([1,1,0,1,1],name='pred',dtype='float')
    true=pd.Series([0,0,1,0,1],name='true',dtype='int')
    group=pd.Series(['g1','g1','g1','g1','g1'],name='group')
    ws=pd.Series([0.2,0.3,0.1,0.9,0.5],name='ws')
    
    res=perfEval2().plot(pred,true)
    res=perfEval2(pred_prob=True).plot(pred_prob,true)
    res=perfEval2().plot(pred,true,group)
    res=perfEval2(pred_prob=True).plot(pred_prob,true,group)
    res=perfEval2(labels=['g','b'],sort_group=['g','b']).plot(pred,true,group)
    res=perfEval2(labels=['g','b'],sort_group=['g','b']).plot(pred,true,group,sample_weight=ws)
    assert isinstance(res,ggplot)
    

