#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:46:24 2022

@author: zengke
"""

from BDMLtools.clearner import outliersTransformer,dtypeAllocator,nanTransformer,dtStandardization
from BDMLtools.report import businessReport,EDAReport,varReport,varGroupsReport,varReportSinge
from BDMLtools.selector import binSelector,binFreq,binPretty,binTree,binChi2,binKmeans
from BDMLtools.selector import faSelector
from BDMLtools.selector import stepLogit,cardScorer
from BDMLtools.selector import preSelector,corrSelector,prefitModel
#from BDMLtools.selector import RFECVSelector
from BDMLtools.selector import lassoSelector
from BDMLtools.encoder import woeTransformer
from BDMLtools.tuner import girdTuner,hgirdTuner
from BDMLtools.tuner import BayesianXGBTuner,BayesianLgbmTuner,shapCheck

import scorecardpy as sc
from sklearn.linear_model import LogisticRegression 
import pandas as pd
import numpy as np

def test_bin():
    
    dt=sc.germancredit().copy()
    dt['creditability']=dt['creditability'].map({'good':0,'bad':1})
    
    
    da=dtypeAllocator().fit(dt)
    dt=da.transform(dt)
    
    X=dt.drop('creditability',axis=1)
    y=dt['creditability']
    
    bins_bm=binSelector(method='tree',bin_num_limit=5,distr_limit=0.05,iv_limit=0,n_jobs=1).fit(X,y).bins
    
    bins_bm_chi=binSelector(method='chi2',bin_num_limit=5,distr_limit=0.05,iv_limit=0,n_jobs=1).fit(X,y).bins
    
    bin_freq=binSelector(method='freq',bin_num_limit=5,iv_limit=0,n_jobs=1).fit(X,y).bins
    
    bin_kmeans=binSelector(method='freq-kmeans',max_bin=5,bin_num_limit=4,iv_limit=0,n_jobs=1).fit(X,y).bins
    
    bin_pretty=binSelector(method='pretty',bin_num_limit=5,distr_limit=0.05,iv_limit=0,n_jobs=1).fit(X,y).bins
    
    
    bin_monotonic=binSelector(method='tree',bin_num_limit=5,distr_limit=0.05,
                   iv_limit=0,n_jobs=1,coerce_monotonic=True).fit(X,y).bins
    
    for key in bin_monotonic:
        
        vtab=bin_monotonic[key]
        
        badprob=vtab.loc[~vtab.index.isin(['missing','special'])]['badprob']
        
        if badprob.is_monotonic_decreasing or badprob.is_monotonic_increasing:
            
            print('monotonic trend shows in {}'.format(key))
        
        else:
            
            print('no monotonic trend shows in {}'.format(key))
            
def test_scorecard():        
    
    dt=sc.germancredit().copy()
    dt['creditability']=dt['creditability'].map({'good':0,'bad':1})
    
    
    da=dtypeAllocator().fit(dt)
    dt=da.transform(dt)
    
    X=dt.drop('creditability',axis=1)
    y=dt['creditability']
        
    breaks_list_user={'age.in.years': [26.0, 30.0, 35.0],
     'credit.amount': [4000.0, 6200.0, 8000.0],
     'credit.history': [2.0, 3.0, 4.0],
     'duration.in.month': [8.0, 16.0, 44.0],
     'foreign.worker': [1],
     'housing': [1.0],
     'installment.rate.in.percentage.of.disposable.income': [2.0, 3.0, 4.0],
     'job': [2.0, 3.0],
     'number.of.existing.credits.at.this.bank': [2.0],
     'number.of.people.being.liable.to.provide.maintenance.for': [2.0],
     'other.debtors.or.guarantors': [2.0],
     'other.installment.plans': [2.0],
     'personal.status.and.sex': [2.0, 3.0],
     'present.employment.since': [2.0, 3.0],
     'present.residence.since': [2.0],
     'property': [1.0, 2.0, 3.0],
     'purpose': ['retraining%,%car (used)',
      'radio/television',
      'furniture/equipment%,%domestic appliances%,%business',
      'repairs%,%car (new)%,%others%,%education'],
     'savings.account.and.bonds': [1.0, 2.0, 3.0],
     'status.of.existing.checking.account': [1.0, 2.0, 3.0],
     'telephone': [1.0]}
    
    
    bin_sc=sc.woebin(dt,y='creditability',breaks_list=breaks_list_user,no_cores=1)
    
    bin_bm=binSelector(breaks_list_adj=breaks_list_user,n_jobs=1).fit(X,y).adjbin
    
    dt_woe_sc = sc.woebin_ply(dt, bins=bin_sc,no_cores=1)
    
    dt_woe_bm = woeTransformer(varbin=bin_bm,n_jobs=1).transform(X,y)
    
    dt_woe_sc_1=dt_woe_sc.loc[:,dt_woe_sc.columns.str.contains('woe')]
    dt_woe_sc_1.columns=[i[:-4] for i in dt_woe_sc_1.columns]
    dt_woe_sc_1=dt_woe_sc_1[dt_woe_bm.columns]
    
    
    print("woe_equal:{}".format(dt_woe_sc_1.astype('float32').equals(dt_woe_bm))) 
    
    lr_sc = LogisticRegression(penalty='l1',C=0.9,solver='saga').fit(dt_woe_sc_1, y)

    lr_bm = LogisticRegression(penalty='l1',C=0.9,solver='saga').fit(dt_woe_bm, y)
    
    
    card_sc = sc.scorecard(bin_sc, lr_sc, dt_woe_sc_1.columns,
                           points0=600,
                           odds0=0.05263157894736842,
                           pdo=50)
    
    
    card_obj = cardScorer(lr_bm,bin_bm,
                        odds0=0.05263157894736842,
                        pdo=50,
                        points0=600).fit(X)

    card_bm = card_obj.scorecard
    
    print(len(card_sc),len(card_bm))
    
    dt_score_sc=sc.scorecard_ply(dt,card_sc)
    
    dt_score_bm=card_obj.transform(X)
    
    print("score_equal:{}".format(dt_score_sc['score'].astype('float32').equals(dt_score_bm['score'])))
    
    
    
def test_tab():
    
    dt=sc.germancredit().copy()
    dt['creditability']=dt['creditability'].map({'good':0,'bad':1})
    
    dtypes_dict={
        'num':['age.in.years',
             'credit.amount',
             'creditability',
             'duration.in.month',
             'installment.rate.in.percentage.of.disposable.income',
             'number.of.existing.credits.at.this.bank',
             'number.of.people.being.liable.to.provide.maintenance.for',
             'present.residence.since'],
        'str':['housing','telephone','foreign.worker','purpose','job','personal.status.and.sex','property',
               'credit.history','savings.account.and.bonds','present.employment.since',
               'status.of.existing.checking.account',
               'other.installment.plans','other.debtors.or.guarantors'],
        'date':[]
    }
    
    da=dtypeAllocator(dtypes_dict=dtypes_dict).fit(dt)
    dt=da.transform(dt)
    X=dt.drop('creditability',axis=1)
    y=dt['creditability']
    
    bin_tree=binSelector(method='tree',bin_num_limit=8,n_jobs=1,iv_limit=0).fit(X,y)
    
    vtab=varReport(bin_tree.breaks_list,n_jobs=1).fit(X,y)
    
    varReportSinge().report(X['age.in.years'],y,[20,30,40,50])
    
    X_all=X.join(y).assign(
        month=np.random.randint(9,13,y.size),
        client_group=pd.Series(np.random.randint(0,3,y.size),index=y.index).map({0:'g1',1:'g2',2:'g3'})
    )
    
    vtab_g=varGroupsReport(bin_tree.breaks_list,columns=['month','client_group'],target=y.name,row_limit=0,
                          n_jobs=1).fit(X_all)
    
    sort_columns={
    'month':['9','10','11','12'],
    'client_group':['g3','g2','g1']
    }
    
    vtab_g=varGroupsReport(bin_tree.breaks_list,columns=['month','client_group'],target=y.name,row_limit=0,
                          sort_columns=sort_columns,
                          n_jobs=1).fit(X_all)
    
    vtab_g=varGroupsReport(bin_tree.breaks_list,columns=['month','client_group'],target=y.name,
                          row_limit=80,
                          sort_columns=sort_columns,
                          n_jobs=1).fit(X_all)

    vtabs_g=varGroupsReport(bin_tree.breaks_list,columns=['client_group'],target=y.name,
                        row_limit=0,output_psi=True,n_jobs=1).fit(X_all)
    
    print("vtabs finish")
    

def test_selector():    
    
    dt=sc.germancredit().copy()
    dt['creditability']=dt['creditability'].map({'good':0,'bad':1})
    
    
    da=dtypeAllocator().fit(dt)
    dt=da.transform(dt)
    
    X=dt.drop('creditability',axis=1)
    y=dt['creditability']    
    
    preSelector().fit(X,y)
    
    prefitModel().fit(X,y)
    
    bin_bm=binSelector(method='tree',n_jobs=1).fit(X,y).bins
    
    dt_woe_bm = woeTransformer(varbin=bin_bm,n_jobs=1).transform(X,y)
    
    faSelector().fit(dt_woe_bm,y)
    
    corrSelector().fit(dt_woe_bm,y)
    
    lassoSelector().fit(dt_woe_bm,y)
    
    BayesianXGBTuner(para_space={
                 'n_estimators': (80, 150),
                 'learning_rate': (0.05, 0.2),
                 'max_depth': (3, 10),
                 'gamma': (0, 20),
                 'min_child_weight': (0, 10),
                 'max_delta_step': (0, 0),
                 'scale_pos_weight': (11,11),
                 'subsample': (0.5, 1),
                 'colsample_bytree': (0.5, 1),
                 'reg_lambda': (0, 10)
                           }).fit(dt_woe_bm,y)
    
    BayesianLgbmTuner(para_space={
                     'boosting_type':'gbdt', 
                     'n_estimators':(30,120),
                     'learning_rate':(0.05,0.2), 
                    
                     'max_depth':(2,4),
                     'min_split_gain': (0,20),
                     'min_sum_hessian_in_leaf': (0,20),
                     
                     'scale_pos_weight':(1,1),
                     'subsample':(0.5,1),
                     'colsample_bytree' :(0.5,1),
                     'reg_lambda':(0,10), 
                     }).fit(dt_woe_bm,y)
    
    
test_bin()

test_scorecard()

test_tab()

test_selector()