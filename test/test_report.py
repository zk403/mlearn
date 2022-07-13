#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:34:22 2022

@author: zengke
"""

from BDMLtools.report import EDAReport,varReport,varGroupsReport,varReportSinge,GainsTable
import pandas as pd


def test_EDAReport():

    dt=pd.DataFrame(
        {'a':[1,2,2],'b':[1,2,3],'c':['1','2','3']},
        index=[0,1,2])
    
    res=EDAReport().fit(dt)
    
    print(res.num_report)
    print(res.char_report)
    print(res.na_report)
    
def test_varReportSinge():
    
    X=pd.DataFrame(
        {'a':[1,2,2],'c':['1','2','3']})
    
    y=pd.Series([0,1,1],name='y')
    
    varReportSinge().report(X['a'],y,[1])
    varReportSinge().report(X['c'],y,['1','2','3'])


def test_varReport():
    
    X=pd.DataFrame(
        {'a':[1,2,2],'c':['1','2','3']})
    
    y=pd.Series([0,1,1],name='y')
    
    res=varReport({'a':[1],'c':['1','2','3']},n_jobs=1).fit(X,y)
    
    assert hasattr(res,'var_report_dict')
    assert hasattr(res,'breaks_list_dict')
    

def test_varGroupsReport():  
    
    X=pd.DataFrame(
        {'a':[1,2,2],'c':['1','2','3'],'g':['g1','g1','g2'],'y':[0,1,1]})
    
    res=varGroupsReport({'a':[1],'c':['1','2','3']},columns=['g'],target='y',n_jobs=1).fit(X)
    
    assert hasattr(res,'report_dict')
    
    
def test_GainsTable():      
    
    X=pd.Series([100,200,300,400,500],name='score')
    y=pd.Series([1,0,0,1,1],name='y')
    
    print(GainsTable().fit_report(X, y))   