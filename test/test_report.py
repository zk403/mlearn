#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:34:22 2022

@author: zengke
"""

from BDMLtools.report import EDAReport,varReport,varGroupsReport,varReportSinge,varGroupsPlot,GainsTable,businessReport
import pandas as pd
import numpy as np
import mock

def test_EDAReport():

    dt=pd.DataFrame(
        {'a':[1,2,2],'b':[1,2,3],'c':['1','2',np.nan],'d':[1,2,np.nan]},
        index=[0,1,2])
    
    res=EDAReport().report(dt)
    res=EDAReport(out_path='tmp').report(dt)
    res=EDAReport(missing_values=[1,'1']).report(dt)
    res=EDAReport(missing_values={'a':[1],'b':[1,2,3]}).report(dt) 
    res=EDAReport(is_nacorr=True).report(dt)
    assert hasattr(res,'num_report')
    assert hasattr(res,'char_report')
    assert hasattr(res,'na_report')
    assert hasattr(res,'nacorr_report')

@mock.patch('matplotlib.pyplot.show')
def test_varReportSinge(mock_show):
    
    X=pd.DataFrame(
        {'a':[1,2,2],'c':['1','2','3']})
    
    ws=pd.Series([1,1,1],name='y')
    y=pd.Series([0,1,1],name='y')
    
    varReportSinge().report(X['a'],y,[1])
    varReportSinge().report(X['c'],y,['1','2','3'])
    varReportSinge().report(X['a'],y,[1],sample_weight=ws)
    varReportSinge().woe_plot(X['c'],y,['1','2','3'])
    varReportSinge().woe_plot(X['c'],y,['1','2','3'],sample_weight=ws)
    varReportSinge().woe_plot(X['c'],y,['1','2%,%3'])
    varReportSinge().woe_plot(X['c'],y,['1%,%2','3'],sample_weight=ws)

    
@mock.patch('matplotlib.pyplot.show')
def test_varReport(mock_show):
    
    X=pd.DataFrame(
        {'a':[1,2,2],'c':['1','2','3']})
    
    ws=pd.Series([1,1,1],name='y')
    y=pd.Series([0,1,1],name='y')

    res=varReport({'a':[1],'c':['1','2','3']},n_jobs=1).fit(X,y)
    res=varReport({'a':[1],'c':['1','2','3']},sample_weight=ws,n_jobs=1).fit(X,y)
    res=varReport({'a':[1],'c':['1','2','3']},out_path='tmp',n_jobs=1).fit(X,y)
    
    assert hasattr(res,'var_report_dict')
    assert hasattr(res,'breaks_list_dict')
    
    res.woe_plot(n_jobs=1)
    

def test_businessReport():
    
    X=pd.DataFrame(
        {'a':[1,2,2],'c':['1','2','3'],'g':['g1','g1','g2'],'y':[0,1,1]})
    
    businessReport('y',['g'],None).report(X)
    
    businessReport('y',['g'],None,rename_columns=['group']).report(X)
    
    businessReport('y',['g'],None,rename_index=['group']).report(X)
    
    businessReport('y',['g'],None,out_path='tmp').report(X)

    
@mock.patch('matplotlib.pyplot.show')
def test_varGroupsReport(mock_show):  
    
    X=pd.DataFrame(
        {'a':[1,2,2],'c':['1','2','3'],'g':['g1','g1','g2'],'y':[0,1,1]})
    ws=pd.Series([1,1,1],name='y')    
    
    res=varGroupsReport({'a':[1],'c':['1','2','3']},columns=['g'],target='y',n_jobs=1).fit(X)
    res=varGroupsReport({'a':[1],'c':['1','2','3']},sample_weight=ws,columns=['g'],target='y',n_jobs=1).fit(X)
    res=varGroupsReport({'a':[1],'c':['1','2','3']},columns=['g'],row_limit=1,target='y',n_jobs=1).fit(X)
    res=varGroupsReport({'a':[1],'c':['1','2','3']},columns=['g'],sort_columns={'g':['g1','g2']},target='y',n_jobs=1).fit(X)   
    res=varGroupsReport({'a':[1],'c':['1','2','3']},out_path='tmp',columns=['g'],target='y',n_jobs=1).fit(X)
    
    assert hasattr(res,'report_dict')
    assert not len(set(res.report_dict.keys())-set(['report_all','report_brief','report_count',
                                                    'report_badprob','report_iv','report_ks','report_lift']))
             
    res=varGroupsReport({'a':[1],'c':['1','2','3']},columns=['g'],output_psi=True,target='y',n_jobs=1).fit(X)
    assert hasattr(res,'report_dict')
    assert not len(set(res.report_dict.keys())-set(['report_all','report_brief','report_count',
                                                    'report_badprob','report_iv','report_ks','report_psi','report_lift']))
    
    varGroupsPlot({'a':[1],'c':['1','2','3']},column='g',target='y',n_jobs=1).plot(X)
    varGroupsPlot({'a':[1],'c':['1','2','3']},column='g',sort_column=['g1','g2'],target='y',n_jobs=1).plot(X)
    

def test_GainsTable():      
    
    X=pd.Series([100,250,280,310,460],name='score')
    g=pd.Series(['g1','g1','g2','g2','g2'])
    y=pd.Series([1,0,0,1,1],name='y')

    GainsTable().report(X, y)   
    GainsTable(order='prob').report(X, y)   
    GainsTable(order='score').report(X, y)   
    GainsTable(method='f').report(X, y)   
    GainsTable().report(X, y, g)   
    GainsTable().report(X, y, g, ['g1'])   
    
    
    
    
    
    