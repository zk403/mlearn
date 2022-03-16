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
from BDMLtools.selector import preSelector,prefitModel
#from BDMLtools.selector import RFECVSelector
from BDMLtools.selector import lassoSelector,LgbmRFECVSelector,LgbmSeqSelector
from BDMLtools.plotter import  perfEval
from BDMLtools.encoder import woeTransformer
from BDMLtools.tuner import girdTuner,hgirdTuner
from BDMLtools.tuner import BayesianXGBTuner,BayesianLgbmTuner,shapCheck
import scorecardpy as sc
from sklearn.linear_model import LogisticRegression 
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class test:
    
    def test_all(self):
        
        self.test_dtStandardization()
        self.test_dtypeAllocator()
        self.test_outliersTransformer()
        self.test_nanTransformer()
        self.test_prefitModel()
        self.test_preSelector()
        self.test_binSelector()
        self.test_scorecard()
        self.test_perfEval()
        self.test_tab()
        self.test_tunner()       

    def test_dtStandardization(self):
    
        dt=sc.germancredit().copy().assign(ids=range(len(sc.germancredit())))
        dt['creditability']=dt['creditability'].map({'good':0,'bad':1})
    
        dt_1=dtStandardization(id_col=['ids'],col_rm=['status.of.existing.checking.account','age.in.years'],
                          downcast=False,set_index=True,drop_dup=True).fit_transform(dt)
        
        
        if np.isin(['status.of.existing.checking.account','age.in.years'],dt_1.columns).any():
            
            raise ValueError('param col_rm in dtStandardization error')
    
        if not pd.Series(dt_1.index).equals(dt['ids']):
            
            raise ValueError('param id_col in dtStandardization error')
            
        dt_dup=pd.concat([dt,dt])
        
        dt_2=dtStandardization(id_col=None,col_rm=None,
                          downcast=False,set_index=True,drop_dup=True).fit_transform(dt_dup)    
        
        if dt_2.index.duplicated().any():
            
            raise ValueError('param drop_dup in dtStandardization error')
        
        dt_dup=pd.concat([dt,dt],axis=1)
        
        dt_2=dtStandardization(id_col=None,col_rm=None,
                          downcast=False,set_index=True,drop_dup=True).fit_transform(dt_dup)   
        
        if dt_2.columns.duplicated().any():
            
            raise ValueError('param drop_dup in dtStandardization error')
            
        dt_dup_1=pd.concat([dt_dup,dt_dup],axis=0)  
        
        dt_2=dtStandardization(id_col=['ids'],col_rm=None,
                          downcast=False,set_index=True,drop_dup=True).fit_transform(dt_dup_1)       
    
        
        if dt_2.columns.duplicated().any() or  dt_2.index.duplicated().any():           
            
            raise ValueError('param drop_dup in dtStandardization error')
            
            
        print('dtStandardization test successfully')


    def test_dtypeAllocator(self):       
        
        dt=sc.germancredit().copy().assign(date='2021-10-10',
                                           booltest=True,
                                           datediff=pd.to_timedelta('15 Days'),
                                           floattest=0.111111111)
        dt['creditability']=dt['creditability'].map({'good':0,'bad':1})
        
        
        
        dtypes_dict={
            'num':['age.in.years',
                 'credit.amount',
                 'creditability',
                 'duration.in.month',
                 'installment.rate.in.percentage.of.disposable.income',
                 'number.of.existing.credits.at.this.bank',
                 'number.of.people.being.liable.to.provide.maintenance.for',
                 'present.residence.since',
                 'floattest',
                 'booltest',
                 ],
            'str':[
                   #'housing',
                   #'telephone',
                   'foreign.worker','purpose','job','personal.status.and.sex','property',
                   'credit.history','savings.account.and.bonds','present.employment.since',
                   'status.of.existing.checking.account',
                   'other.installment.plans','other.debtors.or.guarantors'],
            'date':['date'],
            'tdiff':['datediff']
        }
        
        da=dtypeAllocator(dtypes_dict=dtypes_dict,col_rm=['housing','telephone'],t_unit='1 D',dtype_num='float64',
                                                          drop_date=False,precision=3).fit(dt)
        dt_1=da.transform(dt)
        
       

        if not np.equal(dt_1[['housing','telephone']].dtypes,['category','category']).all():
            
            raise ValueError('param col_rm in dtypeAllocator error')
            
        if not np.equal(dt_1[['date']].dtypes,['datetime64[ns]']).all():
            
            raise ValueError('date column setting in dtypeAllocator error')

        if not np.equal(dt_1[['datediff']].dtypes,['float64']).all():
            
            raise ValueError('date column setting in dtypeAllocator error')            
            

        dtypes_dict={
            'num':['age.in.years',
                 'credit.amount',
                 'creditability',
                 'duration.in.month',
                 'installment.rate.in.percentage.of.disposable.income',
                 'number.of.existing.credits.at.this.bank',
                 'number.of.people.being.liable.to.provide.maintenance.for',
                 'present.residence.since',
                 'floattest',
                 'booltest',
                 ],
            'str':[
                   'housing',
                   'telephone',
                   'foreign.worker','purpose','job','personal.status.and.sex','property',
                   'credit.history','savings.account.and.bonds','present.employment.since',
                   'status.of.existing.checking.account',
                   'other.installment.plans','other.debtors.or.guarantors'],
            'date':['date'],
            'tdiff':['datediff']
        }
        
        da=dtypeAllocator(dtypes_dict=dtypes_dict,col_rm=None,t_unit='15 D',dtype_num='float64',
                                                          drop_date=True,precision=3).fit(dt)
        dt_1=da.transform(dt)

        if 'date' in dt_1.columns:
            
            raise ValueError('param drop_date in dtypeAllocator error')
            
        if not np.equal(dt_1.dtypes.unique(),[np.dtype('float64'),np.dtype('O')]).all():

            raise ValueError('dtypes of out contain unknown type')
            
            
            
            
        da=dtypeAllocator(dtypes_dict={},col_rm=None,t_unit='1 D',dtype_num='float64',
                                                          drop_date=True,precision=3).fit(dt)
        dt_1=da.transform(dt)  

        if not np.equal(dt_1.dtypes.unique(),[np.dtype('float64'),np.dtype('O')]).all():

            raise ValueError('dtypes of out contain unknown type')             
            
            
        da=dtypeAllocator(dtypes_dict={},col_rm=None,t_unit='1 D',dtype_num='float32',
                                                          drop_date=True,precision=3).fit(dt)
        dt_1=da.transform(dt)  

        if not np.equal(dt_1.dtypes.unique(),[np.dtype('float32'),np.dtype('O')]).all():

            raise ValueError('dtypes of out contain unknown type') 


        da=dtypeAllocator(dtypes_dict={},col_rm=None,t_unit='1 D',dtype_num='float64',
                                                          drop_date=True,precision=10).fit(dt)
        dt_1=da.transform(dt)  

                 
        if not len(str(dt_1.floattest[0]))==11:
        
            raise ValueError('param precision not work') 

        print('dtypeAllocator test successfully')
        
        
        
    def test_outliersTransformer(self):
        
        dt=sc.germancredit().copy()
        da=dtypeAllocator().fit(dt)
        dt=da.transform(dt)
        
        dt_nan=outliersTransformer(columns=['credit.amount'],method='nan').fit_transform(dt)
        dt_fill=outliersTransformer(columns=['credit.amount'],method='fill').fit_transform(dt)
        
        dt['credit.amount'].hist()    
        
        dt_nan['credit.amount'].hist()    
        
        dt_fill['credit.amount'].hist()
        
        if dt_nan['credit.amount'].isnull().sum()!=24:
            
            raise ValueError('outliersTransformer error') 
            
        if dt_fill['credit.amount'][dt_nan['credit.amount'].isnull()].unique()!=[11792.5]:
        
            raise ValueError('outliersTransformer error') 
            
        print('outliersTransformer test successfully')
        
    
    def test_nanTransformer(self):
        
        dt=pd.DataFrame(
            {
                'v1':np.arange(5),
                'v2':np.array([np.inf,-np.inf,np.nan,5.1,1.0]),
                'v3':np.array(['g1','','special','g3','nan'])
            })
        
        missing_list=[1,np.nan, -np.inf,np.inf,'nan', '', 'special', 'missing']
        
        dt_1=nanTransformer(missing_values=missing_list).fit_transform(dt)
        
        if not np.isnan(dt_1.v2[0:2]).all():
            raise ValueError('nanTransformer error:inf')
        
        if not (dt_1.v3.unique()==['g1', 'missing', 'g3']).all():
            raise ValueError('nanTransformer error:string')
            
        if not np.isnan(dt_1['v1'][dt['v1']==1].values).all() :   
            raise ValueError('nanTransformer error:int')
            
        if not np.isnan(dt_1['v2'][dt['v2']==1].values).all() :   
            raise ValueError('nanTransformer error:float')    
            
            
        missing_dict={
            'v1':[1,np.nan, -np.inf,np.inf,'nan', '', 'special', 'missing'],
            'v2':[1,np.nan, -np.inf,np.inf,'nan', '', 'special', 'missing'],
            'v3':[1,np.nan, -np.inf,np.inf,'nan', '', 'special', 'missing']
        }    
                    
        dt_1=nanTransformer(missing_values=missing_dict,fill_value=(-9999,'na')).fit_transform(dt)
        
        if not np.isin(dt_1.v2[0:2],[-9999]).all():
            raise ValueError('nanTransformer error:inf')
        
        if not (dt_1.v3.unique()==['g1', 'na', 'g3']).all():
            raise ValueError('nanTransformer error:string')
            
        if not np.isin(dt_1['v1'][dt['v1']==1].values,[-9999]).all() :   
            raise ValueError('nanTransformer error:int')
            
        if not np.isin(dt_1['v2'][dt['v2']==1].values,[-9999]).all() :   
            raise ValueError('nanTransformer error:float') 
            
        print('nanTransformer test successfully')
        
    
    def test_preSelector(self):
        
        X=pd.DataFrame(
            {
                'na_test':np.append(np.ones(100),np.repeat(np.nan,900)),#nan pop=0.9
                'uni_test':np.append(np.repeat('-1',800),np.arange(200)),#-1 unique pop=0.8
                'var_test':np.repeat(1,1000), #variance = 0
                'chi2_test':np.repeat('g1',1000), #p_value no significancne
                'f_test':np.random.randint(0,100,1000), #p_value no significancne      
            }
        )
        
        y=pd.Series(np.random.randint(0,2,1000),name='target')
        
        keep_col=preSelector(na_pct=0.89,
                   unique_pct=None,
                   variance=None,
                   chif_pvalue=None,
                   tree_imps=None,
                   auc_limit=None,
                   iv_limit=None).fit(X[['na_test']],y).keep_col

        if 'na_test' in keep_col:
            
            raise ValueError("param na_pct error")
            
        keep_col=preSelector(na_pct=None,
                           unique_pct=0.81,
                           variance=None,
                           chif_pvalue=None,
                           tree_imps=None,
                           auc_limit=None,
                           iv_limit=None).fit(X[['uni_test']],y).keep_col
        
        if not 'uni_test' in keep_col:
            
            raise ValueError("param unique_pct error")
            
        keep_col=preSelector(na_pct=None,
                           unique_pct=None,
                           variance=0,
                           chif_pvalue=None,
                           tree_imps=None,
                           auc_limit=None,
                           iv_limit=None).fit(X[['var_test']],y).keep_col
        
        if 'var_test' in keep_col:
            
            raise ValueError("param variance error")
            
        keep_col=preSelector(na_pct=None,
                           unique_pct=None,
                           variance=None,
                           chif_pvalue=0.05,
                           tree_imps=None,
                           auc_limit=None,
                           iv_limit=None).fit(X[['chi2_test','f_test']],y).keep_col
        
        if 'chi2_test' in keep_col or 'f_test' in keep_col:
            
            raise ValueError("param chif_pvalue error")
            
        print('preSelector test successfully')
        

    def test_prefitModel(self):
        
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
                 'present.residence.since',
                 ],
            'str':[
                   'housing',
                   'telephone',
                   'foreign.worker','purpose','job','personal.status.and.sex','property',
                   'credit.history','savings.account.and.bonds','present.employment.since',
                   'status.of.existing.checking.account',
                   'other.installment.plans','other.debtors.or.guarantors']
        }
        
        da=dtypeAllocator(dtypes_dict=dtypes_dict,t_unit='1 D',dtype_num='float64',
                                            drop_date=True,precision=3).fit(dt)
        
        dt_1=da.transform(dt)
        
        X=dt_1.drop('creditability',axis=1)
        y=dt_1['creditability']    
        
        from sklearn.metrics import roc_auc_score
        res=prefitModel(method='floor',max_iter=300,col_rm=['credit.amount','telephone']).fit(X,y)
        auc=roc_auc_score(y,res.predict_proba(X))
        print('floor auc:{}'.format(auc)) 
        
        res=prefitModel(method='ceiling',tree_params={'max_depth': 2, 'learning_rate': 0.1, 'n_estimators': 50},col_rm=['credit.amount','telephone']).fit(X,y)
        auc=roc_auc_score(y,res.predict_proba(X))
        print('ceiling auc:{}'.format(auc)) 
        
        print('prefitModel test successfully')


    def test_binSelector(self):
        
        dt=sc.germancredit().copy()
        da=dtypeAllocator().fit(dt)
        dt=da.transform(dt)
        dt['creditability']=dt['creditability'].map({'good':0,'bad':1})
        
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
                
        print('binSelector test successfully')
               
            
    def test_scorecard(self):        
    
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
        
        bin_bm=varReport(breaks_list_dict=breaks_list_user,n_jobs=1).fit(X,y).var_report_dict
        
        dt_woe_sc = sc.woebin_ply(dt, bins=bin_sc,no_cores=1)
        
        dt_woe_bm = woeTransformer(varbin=bin_bm,n_jobs=1).transform(X,y)
        
        dt_woe_sc_1=dt_woe_sc.loc[:,dt_woe_sc.columns.str.contains('woe')]
        dt_woe_sc_1.columns=[i[:-4] for i in dt_woe_sc_1.columns]
        dt_woe_sc_1=dt_woe_sc_1[dt_woe_bm.columns]
        
        
        print("woe_equal:{}".format(dt_woe_sc_1.astype('float32').equals(dt_woe_bm.astype('float32')))) 
        
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
        
        
        
        print("score_equal:{}".format(dt_score_sc['score'].equals(dt_score_bm['score'])))
        
        print('scorecard test successfully')    
    
    
    def test_tab(self):
        
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
        figs=vtab.woe_plot()
        
        varReportSinge().report(X['age.in.years'],y,[20,30,40,50])
        varReportSinge().woe_plot(X['age.in.years'],y,[20,30,40,50])
        
        
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
        
        
        
        print("vtabs test successfully")
    

    def test_tunner(self):    
        
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
        
        lassoSelector().fit(dt_woe_bm,y)
        
        LgbmRFECVSelector().fit(dt_woe_bm,y)
        
        LgbmSeqSelector().fit(dt_woe_bm,y)
        
        
        #Bayesian based tunner
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
        
        #Gird based tunner
        girdTuner(XGBClassifier,para_space={
                             'n_estimators':[100],
                             'learning_rate':[0.1],
                            
                             'max_depth':[3],
                             'gamma': [0,10],
                             'min_child_weight':[0],
                             
                             'subsample':[0.6,0.8],
                             'colsample_bytree' :[0.6,0.8],
                             'reg_lambda':[0,10], 
                             'scale_pos_weight':[1,10],
                             'max_delta_step':[0]
                             },method='gird'           
            ).fit(dt_woe_bm,y)
        
        
        from scipy.stats import randint as sp_randint
        from scipy.stats import uniform as sp_uniform 
        from BDMLtools.tuner.base import sLGBMClassifier
        
        girdTuner(XGBClassifier,para_space={
                  'n_estimators':sp_randint(low=60,high=120),#迭代次数
                  'learning_rate':sp_uniform(loc=0.05,scale=0.15), #学习率
                 
                  'max_depth':sp_randint(low=2,high=4),
                  'gamma': sp_uniform(loc=0,scale=21),
                  'min_child_weight': sp_uniform(loc=0,scale=21),
                  
                  'subsample':sp_uniform(loc=0.5,scale=0.5),
                  'colsample_bytree' :sp_uniform(loc=0.5,scale=0.5),
                  
                  'reg_lambda':sp_randint(low=0,high=1), 
                  'scale_pos_weight':sp_uniform(loc=1,scale=0), 
                  'max_delta_step':sp_uniform(loc=0,scale=0)
                  } ,method='random_gird'           
            ).fit(dt_woe_bm,y)        
        
        
        girdTuner(sLGBMClassifier,para_space={
                     'boosting_type':['gbdt','goss'], 
                     'n_estimators':[100],
                     'learning_rate':[0.1], 
                    
                     'max_depth':[3],#[0,∞],
                     'min_split_gain': [0],
                     'min_child_weight':[0],
                     
                     'scale_pos_weight':[1],
                     'subsample':[0.6,0.8],
                     'colsample_bytree' :[0.6,0.8],
                     'reg_lambda':[0,10], 
                             },method='gird'           
            ).fit(dt_woe_bm,y)

        
        girdTuner(sLGBMClassifier,para_space={
                     'boosting_type':['gbdt','goss'], #'goss','gbdt'
                     'n_estimators':sp_randint(low=100,high=110),
                     'learning_rate':sp_uniform(loc=0.1,scale=0), 
                    
                     'max_depth':sp_randint(low=2,high=4),#[0,∞],
                     'min_split_gain': sp_uniform(loc=0,scale=0),
                     'min_child_weight': sp_uniform(loc=0,scale=0),
                     
                     'scale_pos_weight':[1,11],
                     'subsample':sp_uniform(loc=0.5,scale=0.5),
                     'colsample_bytree' :sp_uniform(loc=0.5,scale=0.5),
                     'reg_lambda':sp_uniform(loc=0,scale=20),

                  } ,method='random_gird'           
            ).fit(dt_woe_bm,y)        
        
        #Halving based tunner
        hgirdTuner(XGBClassifier,para_space={
                             'n_estimators':[100],
                             'learning_rate':[0.1],
                            
                             'max_depth':[3],
                             'gamma': [0,10],
                             'min_child_weight':[0],
                             
                             'subsample':[0.6,0.8],
                             'colsample_bytree' :[0.6,0.8],
                             'reg_lambda':[0,10], 
                             'scale_pos_weight':[1,10],
                             'max_delta_step':[0]
                             },method='h_gird'           
            ).fit(dt_woe_bm,y)
        
        
        hgirdTuner(XGBClassifier,para_space={
                  'n_estimators':sp_randint(low=60,high=120),#迭代次数
                  'learning_rate':sp_uniform(loc=0.05,scale=0.15), #学习率
                 
                  'max_depth':sp_randint(low=2,high=4),
                  'gamma': sp_uniform(loc=0,scale=21),
                  'min_child_weight': sp_uniform(loc=0,scale=21),
                  
                  'subsample':sp_uniform(loc=0.5,scale=0.5),
                  'colsample_bytree' :sp_uniform(loc=0.5,scale=0.5),
                  
                  'reg_lambda':sp_randint(low=0,high=1), 
                  'scale_pos_weight':sp_uniform(loc=1,scale=0), 
                  'max_delta_step':sp_uniform(loc=0,scale=0)
                  } ,method='h_random'           
            ).fit(dt_woe_bm,y)   
        
        
        hgirdTuner(sLGBMClassifier,para_space={
                     'boosting_type':['gbdt','goss'], 
                     'n_estimators':[100],
                     'learning_rate':[0.1], 
                    
                     'max_depth':[3],#[0,∞],
                     'min_split_gain': [0],
                     'min_child_weight':[0],
                     
                     'scale_pos_weight':[1],
                     'subsample':[0.6,0.8],
                     'colsample_bytree' :[0.6,0.8],
                     'reg_lambda':[0,10], 
                             },method='h_gird'           
            ).fit(dt_woe_bm,y)

        
        hgirdTuner(sLGBMClassifier,para_space={
                     'boosting_type':['gbdt','goss'], #'goss','gbdt'
                     'n_estimators':sp_randint(low=100,high=110),
                     'learning_rate':sp_uniform(loc=0.1,scale=0), 
                    
                     'max_depth':sp_randint(low=2,high=4),#[0,∞],
                     'min_split_gain': sp_uniform(loc=0,scale=0),
                     'min_child_weight': sp_uniform(loc=0,scale=0),
                     
                     'scale_pos_weight':[1,11],
                     'subsample':sp_uniform(loc=0.5,scale=0.5),
                     'colsample_bytree' :sp_uniform(loc=0.5,scale=0.5),
                     'reg_lambda':sp_uniform(loc=0,scale=20),

                  } ,method='h_random'           
            ).fit(dt_woe_bm,y)        

        
        hgirdTuner(CatBoostClassifier,
                         para_space={
                             'nan_mode':['Min'],
                             'n_estimators': [80, 100],
                             'learning_rate': [0.03,0.05, 0.1],
                             'max_depth': [2,3],
                             'scale_pos_weight': [1],
                             'subsample': [1],
                             'colsample_bylevel': [1],
                             'reg_lambda': [0]
                         } ,method='h_gird'           
            ).fit(dt_woe_bm,y)                
        
        
        hgirdTuner(CatBoostClassifier,para_space={         
                             'nan_mode':['Min'],
                             'n_estimators':sp_randint(low=100,high=110),
                             'learning_rate':sp_uniform(loc=0.1,scale=0),                     
                             'max_depth':sp_randint(low=2,high=4),#[0,∞],                     
                             'scale_pos_weight':[1],
                             'subsample':sp_uniform(loc=0.5,scale=0.5),
                             'colsample_bylevel' :sp_uniform(loc=0.5,scale=0.5),
                             'reg_lambda':sp_uniform(loc=0,scale=20),
                             },
                    method='h_random'           
            ).fit(dt_woe_bm,y)        
        
        print("tunner test successfully")

        
        
    def test_perfEval(self):
        
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
          
        bin_bm=varReport(breaks_list_dict=breaks_list_user,n_jobs=1).fit(X,y).var_report_dict
        
        dt_woe_bm = woeTransformer(varbin=bin_bm,n_jobs=1).transform(X,y)
          
        lr_bm = LogisticRegression(penalty='l1',C=0.9,solver='saga').fit(dt_woe_bm, y)  
        
        y_pred=pd.Series(lr_bm.predict_proba(dt_woe_bm)[:,1],index=dt_woe_bm.index)
        
        y_true=y
        
        group=pd.Series(np.repeat('g-credit data',len(dt_woe_bm)),index=dt_woe_bm.index)
        
        sample_weight=pd.Series(np.ones(len(dt_woe_bm)),index=dt_woe_bm.index)
        
        figs=perfEval(show_plot=('density',),title='g-credit').plot(y_pred, y_true,group,sample_weight,figure_size=(6,6))
        
        card_obj = cardScorer(lr_bm,bin_bm,
                            odds0=0.05263157894736842,
                            pdo=50,
                            points0=600).fit(X)
        
        card_bm = card_obj.scorecard
        
        dt_score_bm=card_obj.transform(X)
        
        figs_score=perfEval(show_plot=('density',),title='g-credit score').plot(dt_score_bm['score'],y_true,group,sample_weight,figure_size=(6,6))
        
        print(figs)
        print(figs_score)
        
        print('perfEval test successfully')            
        
        

test().test_all()