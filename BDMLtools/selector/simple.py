#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 23:08:38 2021

@author: zengke
"""
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.feature_selection import f_classif,chi2
from category_encoders.ordinal import OrdinalEncoder
from category_encoders import WOEEncoder
from joblib import Parallel,delayed,effective_n_jobs
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from BDMLtools.fun import Specials
from BDMLtools.selector.bin_fun import binFreq
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype,is_string_dtype
import numpy as np
import pandas as pd
import toad
import os
from glob import glob
from BDMLtools.tuner.fun import sLGBMClassifier
from BDMLtools.base import Base


class prefitModel(Base,BaseEstimator):
    """ 
    预拟合数据，在不进行任何特征工程的前提下，使用全量特征预拟合数据并输出模型指标
    此功能用于在建模之前预估现有取数范围下模型的最终的性能，为取数、y定义的合理性提供参考
    若X存在object或number以外类型的列，那么这些列将被算法忽略
    
    Parameters:
    ----------
        method='ceiling',预拟合数据方法，可选‘floor’,‘ceiling’
            floor:地板算法，这里使用线性模型(sklearn对的logit回归(C=0.1 & solver='saga'))进行prefit  
                 + 分类变量处理方式:进行woe编码
                 + 数值特征缺失值:均值填补,当数据缺失值较多时，建议使用ceiling的lightbgm,其可应对缺失数据
            ceiling:天花板算法,这里使用lightgbm进行prefit，且不进行任何交叉验证
                + 分类变量处理方式:进行woe编码
                + 数值特征缺失值:不处理        
        max_iter:100,logit回归最大迭代次数
        tree_params={'max_depth':3,'learning_rate':0.05,'n_estimators':100},method='ceiling'时lightgbm的参数设定        
        col_rm=None or list,需要移除的列名list，例如id类
        sample_weight=None or pd.Series,样本权重
        
    Attribute:
    ----------
        encoder:category_encoders.WOEEncoder object,使用回归时分类变量的woe编码器
        imputer:sklearn.impute.SimpleImputer object,使用回归时数值特征缺失值处理器
        model:sklearn.linear_model.LogisticRegression or lightgbm.LGBMClassifier object拟合的模型对象
    """      
    def __init__(self,method='ceiling',max_iter=100,tree_params={'max_depth':3,'learning_rate':0.05,'n_estimators':100},
                 col_rm=None,sample_weight=None):

        self.method=method
        self.tree_params=tree_params
        self.max_iter=max_iter
        self.col_rm=col_rm
        self.sample_weight=sample_weight    
        
        self._is_fitted=False

        
    def transform(self,X,y=None):
        
        self._check_is_fitted()
           
        f_names=self.model.feature_names_in_ if self.method=='floor' else self.model.feature_name_
        
        X=X[f_names]
        
        X_numeric=X.select_dtypes('number')
        X_categoty=X.select_dtypes('object')
        
        if self.method=='floor':
            
            X_numeric=pd.DataFrame(self.imputer.transform(X_numeric),
                                   columns=self.imputer.feature_names_in_,
                                   index=X_numeric.index,dtype='float32') 
            

        if self.encoder:
            
            X_categoty=self.encoder.transform(X_categoty)

        
        X_new=pd.concat([X_numeric,X_categoty],axis=1)[f_names]
        
        if self.method=='floor':
            
            X_new=pd.DataFrame(self._scaler.transform(X_new),columns=self._scaler.feature_names_in_,
                                       index=X_new.index,dtype='float32')
        
        return X_new
    
    
    def predict_proba(self,X,y=None):
        
        X_model=self.transform(X)
        
        pred=self.model.predict_proba(X_model)[:,1]        
        
        return pred
    
       
    def fit(self,X,y):
        
        X=X.drop(self.col_rm,axis=1) if self.col_rm else X
        
        if self.method=='ceiling':
            
            self.model=self._fit_lgbm(X,y,self.tree_params,self.sample_weight)
            
        elif self.method=='floor':
            
            self.model=self._fit_reg(X,y,self.max_iter,self.sample_weight)
            
        else:
            
            raise ValueError("method in ('ceiling','floor')")   
            
        self._is_fitted=True
        
        return self
    
    def _fit_lgbm(self,X,y,params,sample_weight):
        
        X_numeric,X_categoty_encode=self._get_X(X,y)
        
        X_new=pd.concat([X_numeric,X_categoty_encode],axis=1)
        
        
        max_depth=params['max_depth']
        learning_rate=params['learning_rate']
        n_estimators=params['n_estimators']
        

        lgb=sLGBMClassifier(
                boosting_type='gbdt',
                objective = 'binary',
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,
            ).fit(X_new,y,sample_weight=sample_weight)         
            
        return lgb
    
    
    def _fit_reg(self,X,y,max_iter,sample_weight):
        
        X_numeric,X_categoty_encode=self._get_X(X,y)
        
        X_new=pd.concat([X_numeric,X_categoty_encode],axis=1)
        
        self._scaler=StandardScaler().fit(X_new)

        X_new=pd.DataFrame(self._scaler.transform(X_new),
                       columns=X_new.columns,
                       index=X_numeric.index,dtype='float32')

        logit=LogisticRegression(C=0.1,penalty='l2',solver='saga',max_iter=max_iter).fit(X_new,y,
                                                                                         sample_weight=sample_weight)  
        
        return logit
                
    
    def _get_X(self,X,y):
        
        X_numeric=X.select_dtypes('number')
        X_categoty=X.select_dtypes('object')
        
        if self.method=='floor':
            
            self.imputer=SimpleImputer(missing_values=np.nan,
                      strategy='median').fit(X_numeric)
            
            X_numeric=pd.DataFrame(self.imputer.transform(X_numeric),
                                            columns=self.imputer.feature_names_in_,
                                            index=X_numeric.index,dtype='float32') 
        
        if X_categoty.columns.size:
                
            self.encoder=WOEEncoder(regularization=1e-3).fit(X_categoty,y)
                
            X_categoty=self.encoder.transform(X_categoty)   
            
        else:
            
            self.encoder=None
            
            
        return X_numeric,X_categoty

    
    
class fliterByShuffle(Base,BaseEstimator):
    
    """ 
    shuffle法进行特征筛选:无需做任何特征处理，比较原始数据与与打乱顺序后原始数据的预测能力的差异
    
    Parameters:
    ----------
        s_times=1,int,随机乱序次数，次数越多auc差异越具备统计学意义，但会导致运算量增加
        auc_val=0,float,mean_decreasing_auc阈值,小于等于auc_val的特征将被踢出,默认0,建议范围0-0.005
        n_jobs=-1,int,列并行时任务数
        verbose=0,int,并行信息输出等级  
        
    Attribute:
    ----------

    """          
    
    def __init__(self,s_times=1,auc_val=0,n_jobs=1,verbose=0):
      
        self.s_times=s_times
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.auc_val=auc_val
        
        self._is_fitted=False

    def transform(self,X,y=None): 
        
        self._check_is_fitted()

        return X[self.keep]
    
    
    def fit(self,X,y):

        n_jobs=effective_n_jobs(self.n_jobs)
        p=Parallel(n_jobs=n_jobs,verbose=self.verbose)
            
        res=p(delayed(self._shuffle_single)(col,y,self.s_times) 
                              for name,col in X.iteritems())
        
        self.mean_decreasing_auc=pd.Series({name:auc for name,auc in res},
                                           name='mean_decreasing_auc')
        
        self.keep=self.mean_decreasing_auc[self.mean_decreasing_auc>self.auc_val].index.tolist()
        
        self._is_fitted=True
        
        return self
    
    def _shuffle(self,x,y,s_times,is_str_type=False):

        categorical_feature=[0] if is_str_type else None
        
        diff_auc=[]
 
        for i in range(s_times):
            
            #shuffle 
            col=x.copy()   
            col_shuffle=x.copy()   
            np.random.shuffle(col_shuffle)
        

            #no-shuffle
            lgb_ns=sLGBMClassifier(
                boosting_type='gbdt',
                objective = 'binary',
                learning_rate=0.1,
                n_estimators=20,
                colsample_bytree=1,
                random_state=123
            ).fit(col.reshape(col.size,1),y,categorical_feature=categorical_feature)   

            #shuffle
            lgb_s=sLGBMClassifier(
                boosting_type='gbdt',
                objective = 'binary',
                learning_rate=0.1,
                n_estimators=20,
                colsample_bytree=1,
                random_state=123
            ).fit(col_shuffle.reshape(col.size,1),y,categorical_feature=categorical_feature)   

            #prediction
            pred=lgb_ns.predict_proba(col.reshape(col.size,1))[:,1]

            pred_shuffle=lgb_s.predict_proba(col_shuffle.reshape(col.size,1))[:,1]

            #get auc
            auc_ns=roc_auc_score(y,pred)

            auc_s=roc_auc_score(y,pred_shuffle)

            #get diff
            diff_auc.append(auc_ns-auc_s)        
        
        return np.nanmean(diff_auc)


    def _shuffle_single(self,x,y,s_times=1):
        
        name=x.name

        if is_numeric_dtype(x):

            diff_auc=self._shuffle(x.values,y.values,s_times)

        elif is_string_dtype(x):

            x_o=OrdinalEncoder().fit_transform(x)[name]
            
            diff_auc=self._shuffle(x_o.values,y.values,s_times,True)
            
        else:
            
            raise ValueError('dtype only in ("number" or "object")')

        return name,np.nanmean(diff_auc)

        

class corrSelector(Base,TransformerMixin):
    
    """ 
    相关系数筛选
    Parameters:
    ----------
        corr_limit:float,相关系数阈值
        by:str or pd.Series,
            + ’IV‘:按照iv筛选
            + pd.Series:用户自定义权重,要求index为列名，value为权重值
        keep:需保留的列名list
        
    Attribute:
    ----------
        keep_col:list,保留的列名list
    """    
    
    def __init__(self,corr_limit=0.8,by='IV',keep=None):

        self.corr_limit=corr_limit
        self.by=by
        self.keep=keep
        
        self._is_fitted=False
        
    def transform(self,X,y=None):
        
        self._check_is_fitted()
        
        if self.keep and isinstance(self.keep,list):
            
            self.keep_col=list(set(self.keep_col+self.keep))
        
        return X[self.keep_col]
          
    def fit(self,X,y):
        """ 
        变量筛选
        Parameters:
        ----------
            varbin:分箱结果,计算特征IV使用,由sc.woebin产生      
        """          
        self.keep_col=self._filterByCorr(X,y)
        
        self._is_fitted=True
        
        return self
    
    def _filterByCorr(self,X,y):
        """
        特征共线性检查,将剔除共线性较强但iv较低的特征,保留共线性较强但iv较高的特征 
        Parameters:
        ----------      
        X:训练数据
        """        
#         #递归式剔除,速度较慢        
#         iv_t=toad.quality(X.join(y),target=self.y,iv_only=True)[['iv']]

#         self.var_keep=[]
#         count=0
#         while (X.corr().abs()>corr_limit).sum().sum()>X.columns.size or count==0: #相关系数矩阵中所有特征的相关系数需小于阈值时才停止迭代(非自身)
#             print(X.columns.size)
#             count=count+1
            
#             #相关性系数
#             corrtable=X.corr()
#             var_corr_max=corrtable.apply(lambda x:x[x.index!=x.name].abs().max())
#             var_highcorr=var_corr_max[var_corr_max>corr_limit].index.tolist()
#             #var_lowcorr=var_corr_max[var_corr_max<=corr_limit].index.tolist()

#             self.var_del=[]
            
#             for i in var_highcorr:                
#                 pairs=iv_t.join(corrtable[i][corrtable[i]>corr_limit],how='right').sort_values(i) #变量与其corr最高的变量
#                 #self.var_del.append(pairs[0:1].index[0]) #剔除IV较小的那个
#                 var_del=pairs[0:1].index[0]
            
#             X=X.drop(var_del,axis=1)
        
        #使用toad
        X_drop=toad.selection.drop_corr(X.join(y),target=y.name,threshold=self.corr_limit,by=self.by).drop(y.name,axis=1)
    
        keep_col=X_drop.columns.tolist()
        
        return keep_col

    
class preSelector(Base,Specials,TransformerMixin):
    
    """ 
    线性预筛选,适用于二分类模型
    
    筛选过程(设定为None时代表跳过相应步骤):
    Step 1.缺失值(所有):缺失率高于用户定义值的列将被筛除
    Step 2.唯一值(所有):唯一值占比高于用户定义值列将被筛除
    Step 3.方差(数值特征):方差低于用户定义值列的列将被筛除
    Step 4.卡方独立性检验p值(字符)/方差分析p值(数值):p值大于用户定义值的列将被剔除
    Step 5.乱序筛选(所有):原始顺序与随机顺序后使用模型预测的auc差异小于用户定义值的列将被剔除
    Step 6.Lightgbm筛选(所有):split重要性低于用户定义值的列将被剔除
    Step 7.Iv值筛选(所有):等频30箱后iv值低于用户定义值的列将被剔除
    
    Parameters:
    ----------
        na_pct:float or None,(0,1),默认0.99,缺失率高于na_pct的列将被筛除，设定为None将跳过此步骤
        unique_pct:float or None,(0,1),默认0.99,唯一值占比高于unique_pct的列将被筛除,unique_pct与variance需同时输入，任一设定为None将跳过此步骤
        variance:float or None,默认0,方差低于variance的列将被筛除,将忽略缺失值,unique_pct与variance需同时输入，任一设定为None将跳过此步骤
        chif_pvalue:float or None,(0,1),默认0.05,大于chif_pvalue的列将被剔除,为None将跳过此步骤
                    + 卡方计算中，缺失值将被视为单独一类,
                    + f值计算中，缺失值将被填补为接近+inf和-inf，计算两次，两次结果都不显著的列都将被剔除
        auc_limit:float,使用shuffle法计算原始数据与乱序数据的mean_decreasing_auc,小于等于auc_val的特征将被踢出,默认0,建议范围0-0.005,设定为None将跳过此步骤
        s_times:int,shuffle法乱序次数，越多的s_times的mean_decreasing_auc越具备统计意义，但这会增加计算量
        tree_imps:int or None,lightgbm树的split_gain小于等于tree_imps的列将被剔除,默认1，设定为None将跳过此步骤
        tree_size:int,lightgbm树个数,若数据量较大可降低树个数，若tree_imps为None时该参数将被忽略
        iv_limit:float or None使用进行iv快速筛选的iv阈值(数值等频30箱，分类则按照类别分箱)
        out_path:str or None,模型报告路径,将预筛选过程每一步的筛选过程输出到模型报告中
        missing_values:缺失值指代值
                + None
                + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换且会被计入na_pct
                + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换且会被计入na_pct
        keep:list or None,需保留列的列名list
        
    Attribute:
    ----------
        features_info:dict,每一步筛选的特征进入记录
        preSelector_report:pd.DataFrame,outpath非None时产生的features_info数据框格式
    """    
    
    
    def __init__(self,na_pct=0.99,unique_pct=0.99,variance=0,chif_pvalue=0.05,tree_imps=1,
                 tree_size=100,auc_limit=None,s_times=1,iv_limit=0.02,out_path=None,missing_values=None,keep=None
                 ):

        self.na_pct=na_pct
        self.unique_pct=unique_pct #
        self.variance=variance
        self.chif_pvalue=chif_pvalue #
        self.tree_imps=tree_imps #
        self.tree_size=tree_size
        self.auc_limit=auc_limit
        self.s_times=s_times
        self.iv_limit=iv_limit
        self.out_path=out_path
        self.missing_values=missing_values
        self.keep=keep
        
        self._is_fitted=False
        
    def transform(self,X,y=None):
        """ 
        变量筛选
        """  
        
        self._check_is_fitted()
        
        if self.keep:
        
            keep_col=np.unique(self.keep_col + self.keep).tolist()  
            
        else:
            
            keep_col=self.keep_col       
        
        return X[keep_col]
    
    def fit(self,X,y):
        
        self._check_data(X,y)
        self._check_values()
        
        X=X.drop(self.keep,axis=1) if self.keep else X
       
        X=self._sp_replace(X,self.missing_values,fill_num=np.nan,fill_str=np.nan)
        
        self.features_info={}
    
        #开始筛选
        self.features_info['0.orgin']=X.columns.tolist()
        
        keep_col=self.features_info['0.orgin']
        
        print('0.start__________________________________complete')
        
        #fliter by nan
        if self.na_pct is not None:
            
            keep_col=self._filterByNA(X[keep_col],self.na_pct)
            
            self.features_info['1.filterbyNA']=keep_col
          
            print('1.filterbyNA_____________________________complete')
            
        #fliter by unique_pct
        if self.unique_pct is not None: 
            
            keep_col=self._filterByUnique(X[keep_col],self.unique_pct)
            
            self.features_info['2.filterbyUnique']=keep_col
            
            print('2.filterbyUniquepct______________________complete')    
                      
        #fliter by variance
        if self.variance is not None: 
            
            keep_col=self._fliterByVariance(X[keep_col],self.variance)
            
            self.features_info['3.filterbyVariance']=keep_col
            
            print('3.filterbyVariance_______________________complete')
            
       
        #fliter by chi and f-value
        if self.chif_pvalue is not None:
            
            keep_col=self._filterByChif(X[keep_col],y,self.chif_pvalue)  
            
            self.features_info['4.filterbyChi2Oneway']=keep_col
            
            print('4.filterbyChi2Oneway_____________________complete')     
            
            
        #fliter by shuffle  
        if self.auc_limit is not None:
            
            keep_col=fliterByShuffle(s_times=self.s_times,auc_val=self.auc_limit,n_jobs=-1).fit(X[keep_col],y).keep
            
            self.features_info['5.filterbyShuffle']=keep_col

            print('5.filterbyShuffle________________________complete')  
        
        
        #fliter by lgbm-tree-imp  
        if self.tree_imps is not None:

            keep_col=self._filterByTrees(X[keep_col],y,self.tree_size,self.tree_imps)
            
            self.features_info['6.filterbyTrees']=keep_col
            
            print('6.filterbyTrees__________________________complete')
            
        
        #fliter by iv 
        if self.iv_limit is not None:   
            
            keep_col=self._filterbyIV(X[keep_col],y,self.iv_limit)
            
            self.features_info['7.filterbyIV']=keep_col
            
            print('7.filterbyIV_____________________________complete')
   
        
        print('_____________________________________________Done')  
        
        #summary
        for key in self.features_info:
            
            print('步骤{},保留的特征数:{}'.format(key,len(self.features_info[key])))
        
        #report
        if self.out_path: 
            
            self.preSelector_report=pd.concat([pd.Series(self.features_info[key],name=key) for key in self.features_info.keys()],axis=1)
                
            self._writeExcel() 
        
        self.keep_col=keep_col
        
        self._is_fitted=True
                    
        return self
    
    def _filterByNA(self,X,na_pct):
        
        NAreport=X.isnull().sum().div(len(X))
        
        return NAreport[NAreport<=na_pct].index.tolist() #返回满足缺失率要求的列名
    
    def _filterByUnique(self,X,unique_pct):
  
        X=X.select_dtypes(include=['object','number'])
        X_oth=X.select_dtypes(exclude=['object','number'])
        
        if X.columns.size:
            
            X_unique_pct=X.apply(lambda x:x.value_counts(dropna=False).div(len(X)).max())   
            
            return X_unique_pct[X_unique_pct<unique_pct].index.tolist()+X_oth.columns.tolist()
        
        else:
            
            return X_oth.columns.tolist()

    
    def _fliterByVariance(self,X,variance):

        X_numeric=X.select_dtypes('number')
        X_oth=X.select_dtypes(exclude='number')
        
        #drop constant columns
        X_numeric=X_numeric.loc[:,X_numeric.apply(lambda col: False if col.unique().size==1 else True)]
        
        if X_numeric.columns.size:
            
            X_var=X_numeric.var(ddof=0)
            
            return X_var[X_var>variance].index.tolist()+X_oth.columns.tolist()
        
        else:
            
            return X_oth.columns.tolist()
        

    def _filterByChif(self,X,y,chif_pvalue):
        
        #drop constant columns
        X=X.loc[:,X.apply(lambda col: False if col.unique().size==1 else True)]
        
        
        X_categoty=X.select_dtypes('object')
        
        X_numeric=X.select_dtypes('number')
        
        X_oth=X.select_dtypes(exclude=['object','number'])

        
        #filter by chi2
        if X_categoty.columns.size:      
            
            X_categoty_encode=OrdinalEncoder().fit_transform(X_categoty.replace(np.nan,'missing'))
            
            p_values=chi2(X_categoty_encode,y)[1]
            
            cate_cols=X_categoty_encode.columns[p_values<chif_pvalue].tolist()
            
        else:
            
            cate_cols=[]
            
        
        #filter by oneway
        if X_numeric.columns.size:
            
            cols_fill=X_numeric.median().to_dict()
            
            p_values_pos=f_classif(X_numeric.fillna(2**31),y)[1]
            
            p_values_mean=f_classif(X_numeric.fillna(cols_fill),y)[1]
            
            p_values_neg=f_classif(X_numeric.fillna(-2**31),y)[1]             
            
            num_cols=X_numeric.columns[(p_values_pos<chif_pvalue) | (p_values_neg<chif_pvalue) | (p_values_mean<chif_pvalue)].tolist()
            
        else:
                
            num_cols=[]
            
        
        return cate_cols+num_cols+X_oth.columns.tolist()
    

    def _filterByTrees(self,X,y,tree_size,tree_imps):

        X_numeric=X.select_dtypes(include='number')
        X_category=X.select_dtypes(include='object')
        X_oth=X.select_dtypes(exclude=['number','object'])
        
        
        if X_category.columns.size:
            
            X_category_encode=OrdinalEncoder().fit_transform(X_category).sub(1)
            
            X_new=pd.concat([X_numeric,X_category_encode],axis=1)

            lgb=sLGBMClassifier(
                boosting_type='gbdt',
                objective = 'binary',
                learning_rate=0.1,
                n_estimators=tree_size,
                subsample=0.7,
                colsample_bytree=1,
            ).fit(X_new,y,categorical_feature=X_category_encode.columns.tolist())         

            lgb_imps=lgb.booster_.feature_importance(importance_type='split')
        
            return X_new.columns[lgb_imps>tree_imps].tolist()+X_oth.columns.tolist()
        
        elif X_numeric.columns.size:
            
            lgb=sLGBMClassifier(
                boosting_type='gbdt',
                objective = 'binary',
                learning_rate=0.1,
                n_estimators=tree_size,
                subsample=0.7,
                colsample_bytree=1,
            ).fit(X_numeric,y)         

            lgb_imps=lgb.booster_.feature_importance(importance_type='split')
        
            return X_numeric.columns[lgb_imps>tree_imps].tolist()+X_oth.columns.tolist()
        
        else:
            
            return X_oth.columns.tolist()
        
    
    def _filterbyIV(self,X,y,iv_limit):        
        
        _,vtabs=binFreq(X,y,bin_num_limit=30)
        
        iv_t=pd.Series({key:vtabs[key]['total_iv'].max() for key in vtabs},name='iv_infos')

        return iv_t[iv_t>iv_limit].index.tolist()
    
    
    def _check_values(self):

        if self.na_pct is not None:
            
            if not 0<self.na_pct<1:
            
                raise ValueError("na_pct is float and in (0,1)")
            
        if self.variance is not None:
            
            if not self.variance>=0:
                
                raise ValueError("variance is in (0,1)")

        if self.unique_pct is not None:
            
            if not 0<=self.unique_pct<1:
                
                raise ValueError("variance is in (0,1)")
                
                
        if self.chif_pvalue is not None:
            
            if not 0<self.chif_pvalue<=1:
            
                raise ValueError("chif_pvalue is float and in (0,1]")
                
        if self.tree_imps is not None:
            
            if not self.tree_imps>=0:
                
                raise ValueError("tree_imps is in [0,inf]")
                
        if self.auc_limit is not None:
            
            if not self.auc_limit>=0:
                
                raise ValueError("auc_limit is in [0,inf]")
            
    
        if self.iv_limit is not None:
            
            if not self.iv_limit>0:
                
                raise ValueError("auc_limit is in (0,inf]")           


    def _writeExcel(self):
        
        if not glob(self.out_path):
            
            os.mkdir(self.out_path)
                
        if not glob(self.out_path+"/model_report.xlsx"):
            
            #print(self.out_path+"/model_report.xlsx")
            
            writer = pd.ExcelWriter(self.out_path+"/model_report.xlsx")       
            pd.DataFrame(None).to_excel(writer,sheet_name='summary')
            writer.save()                    
            
        writer=pd.ExcelWriter(self.out_path+'/model_report.xlsx',
                              mode='a',
                              if_sheet_exists='replace',
                              #engine_kwargs={'mode':'a','if_sheet_exists':'replace'},
                              engine='openpyxl')
        
        self.preSelector_report.to_excel(writer,sheet_name='3.preSelect')
            
        writer.save()     
        print('to_excel done') 