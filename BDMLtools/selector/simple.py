#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 23:08:38 2021

@author: zengke
"""
from sklearn.base import TransformerMixin
from sklearn.feature_selection import f_classif,chi2
from category_encoders.ordinal import OrdinalEncoder
from category_encoders import WOEEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from BDMLtools.fun import Specials
from BDMLtools.selector.bin_fun import binFreq
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
from glob import glob
from BDMLtools.tuner.base import sLGBMClassifier
from BDMLtools.base import Base
#from lofo import LOFOImportance, Dataset
from sklearn.feature_selection import SelectFpr,SelectFdr,SelectFwe
#from BDMLtools.selector.lgbm import LgbmPISelector
from joblib import effective_n_jobs


class prefitModel(Base):
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
        
    Attribute:
    ----------
        encoder:category_encoders.WOEEncoder object,使用回归时分类变量的woe编码器
        imputer:sklearn.impute.SimpleImputer object,使用回归时数值特征缺失值处理器
        model:sklearn.linear_model.LogisticRegression or lightgbm.LGBMClassifier object拟合的模型对象
    """      
    def __init__(self,method='ceiling',max_iter=100,tree_params={'max_depth':3,'learning_rate':0.05,'n_estimators':100},
                 col_rm=None):

        self.method=method
        self.tree_params=tree_params
        self.max_iter=max_iter
        self.col_rm=col_rm 
        
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
                                   index=X_numeric.index,dtype='float32') #downcast to save memory
            

        if self.encoder:
            
            X_categoty=self.encoder.transform(X_categoty)

        
        X_new=pd.concat([X_numeric,X_categoty],axis=1)[f_names]
        
        if self.method=='floor':
            
            X_new=pd.DataFrame(self._scaler.transform(X_new),columns=self._scaler.feature_names_in_,
                                       index=X_new.index,dtype='float32') #downcast to save memory
        
        return X_new
    
    
    def predict_proba(self,X,y=None):
        
        X_model=self.transform(X)
        
        pred=self.model.predict_proba(X_model)[:,1]        
        
        return pred
    
       
    def fit(self,X,y,sample_weight=None):
        
        X=X.drop(self.col_rm,axis=1) if self.col_rm else X
        
        X=X.dropna(how='all',axis=1)
        
        if X.columns.size==0:
            
            raise ValueError('All columns in X are nan.')
        
        if self.method=='ceiling':
            
            self.model=self._fit_lgbm(X,y,self.tree_params,sample_weight)
            
        elif self.method=='floor':
            
            self.model=self._fit_reg(X,y,self.max_iter,sample_weight)
            
        else:
            
            raise ValueError("method in ('ceiling','floor')")   
            
        self._is_fitted=True
        
        return self
        
    
    def _fit_lgbm(self,X,y,params,sample_weight):
        
        X_numeric,X_categoty_encode=self._get_X(X,y)
        
        X_new=pd.concat([X_numeric,X_categoty_encode],axis=1)
        
        
        max_depth=params['max_depth'] if 'max_depth' in params else 3
        learning_rate=params['learning_rate'] if 'learning_rate' in params else 0.01
        n_estimators=params['n_estimators'] if 'n_estimators' in params else 100
        n_jobs=params['n_jobs'] if 'n_jobs' in params else -1

        lgb=sLGBMClassifier(
                boosting_type='gbdt',
                objective = 'binary',
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=effective_n_jobs(n_jobs)
            ).fit(X_new,y,sample_weight=sample_weight)         
            
        return lgb
    
    
    def _fit_reg(self,X,y,max_iter,sample_weight):
        
        X_numeric,X_categoty_encode=self._get_X(X,y)
        
        X_new=pd.concat([X_numeric,X_categoty_encode],axis=1)
        
        self._scaler=StandardScaler().fit(X_new)

        X_new=pd.DataFrame(self._scaler.transform(X_new),
                       columns=X_new.columns,
                       index=X_numeric.index,dtype='float32') #downcast to save memory

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
                                            index=X_numeric.index,dtype='float32') #downcast to save memory
        
        if X_categoty.columns.size:
                
            self.encoder=WOEEncoder(regularization=1e-3).fit(X_categoty,y)
                
            X_categoty=self.encoder.transform(X_categoty)   
            
        else:
            
            self.encoder=None
            
            
        return X_numeric,X_categoty



class preSelector(Base,Specials,TransformerMixin):
    
    """ 
    线性预筛选,适用于二分类模型
    
    筛选过程(设定为None时代表跳过相应步骤,相应序号会被重置):
    Step 1.缺失值(所有):缺失率高于用户定义值的列将被筛除
    Step 2.唯一值(所有):唯一值占比高于用户定义值列将被筛除
    Step 3.方差(数值特征):方差低于用户定义值列的列将被筛除
    Step 4.卡方独立性检验p值(字符)/方差分析p值(数值):p值大于用户定义值的列将被剔除(不支持样本权重)
    Step 5.Lightgbm筛选(所有):split重要性低于用户定义值的列将被剔除
    Step 6.Iv值筛选(所有):等频30箱后iv值低于用户定义值的列将被剔除
    
    目前Step 4不支持sample weight(样本权重)
    
    Parameters:
    ----------
        na_pct:float or None,(0,1),默认0.99,缺失率高于na_pct的列将被筛除，设定为None将跳过此步骤
        unique_pct:float or None,(0,1),默认0.99,唯一值占比高于unique_pct的列将被筛除,unique_pct与variance需同时输入，任一设定为None将跳过此步骤
        variance:float or None,默认0,方差低于variance的列将被筛除,将忽略缺失值,unique_pct与variance需同时输入，任一设定为None将跳过此步骤
        chif_pvalue:float or None,(0,1),默认0.05,大于chif_pvalue的列将被剔除,该步骤不支持样本权重。为None将跳过此步骤
                    + 卡方计算中，缺失值将被视为单独一类,
                    + f值计算中，缺失值将被填补为接近+inf和-inf，计算两次，两次结果都不显著的列都将被剔除   
                    + 不支持样本权重
        tree_imps:int or None,lightgbm树的split_gan小于等于tree_imps的列将被剔除,默认1，设定为None将跳过此步骤
        tree_size:int,lightgbm树个数,若数据量较大可降低树个数，若tree_imps为None时该参数将被忽略
        iv_limit:float or None使用进行iv快速筛选的iv阈值(数值等频30箱，分类则按照类别分箱)
        n_jobs:int,默认-1,Lightgbm筛选过程中线程并行控制参数
        out_path:str or None,模型报告路径,将预筛选过程每一步的筛选过程输出到模型报告中
        missing_values:缺失值指代值
                + None
                + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换
                + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换且会被计入na_pct
        keep:list or None,需保留列的列名list
        
    Attribute:
    ----------
        features_info:dict,每一步筛选的特征进入记录
        preSelector_report:pd.DataFrame,outpath非None时产生的features_info数据框格式
    """    
    
    
    def __init__(self,na_pct=0.99,unique_pct=0.99,variance=0,chif_pvalue=0.05,tree_imps=1,random_state=123,
                 tree_size=250,iv_limit=0.02,out_path=None,missing_values=None,keep=None,n_jobs=-1
                 ):

        self.na_pct=na_pct
        self.unique_pct=unique_pct #
        self.variance=variance
        self.chif_pvalue=chif_pvalue #
        self.tree_imps=tree_imps #
        self.tree_size=tree_size
        self.iv_limit=iv_limit
        self.out_path=out_path
        self.missing_values=missing_values
        self.random_state=random_state
        self.keep=keep
        self.n_jobs=n_jobs
        
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
    
    def fit(self,X,y,cat_features=None,sample_weight=None):
        
        self._check_data(X,y)
        self._check_values()
        
        #cat_features
        cat_features=X.select_dtypes(['object','category']).columns.tolist() if cat_features is None else cat_features
        
        X=X.apply(lambda col:col.astype('str') if col.name in cat_features else col) if cat_features else X  

        #keep col
        X=X.drop(self.keep,axis=1) if self.keep else X
        
        #sp values
        X=self._sp_replace(X,self.missing_values,fill_num=np.nan,fill_str=np.nan) #_sp_replace(None) will return raw data
        
        self.features_info={}
    
        #开始筛选
        self.features_info['0.orgin']=X.columns.tolist()
        
        keep_col=self.features_info['0.orgin']
        
        print('Start'.center(100,'-'))
        
        step=0
      
        #fliter by nan
        if self.na_pct is not None and len(keep_col):
            
            keep_col=self._filterByNA(X[keep_col],self.na_pct,ws=sample_weight)
            
            step=step+1
            
            self.features_info[str(step)+'.filterbyNA']=keep_col
          
            print((str(step)+'.filterbyNA').rjust(35)+'Done.'.rjust(35,'_'))
            
        #fliter by unique_pct
        if self.unique_pct is not None and len(keep_col): 
            
            keep_col=self._filterByUnique(X[keep_col],self.unique_pct,ws=sample_weight)
            
            step=step+1
            
            self.features_info[str(step)+'.filterbyUnique']=keep_col
            
            print((str(step)+'.filterbyUniquepct').rjust(42)+'Done.'.rjust(28,'_'))
                      
        #fliter by variance
        if self.variance is not None and len(keep_col): 
            
            keep_col=self._fliterByVariance(X[keep_col],self.variance,ws=sample_weight)
            
            step=step+1
            
            self.features_info[str(step)+'.filterbyVariance']=keep_col
            
            print((str(step)+'.filterbyVariance').rjust(41)+'Done.'.rjust(29,'_'))
            
       
        #fliter by chi and f-value
        if self.chif_pvalue is not None and len(keep_col):
            
            keep_col=self._filterByChif(X[keep_col],y,self.chif_pvalue)  
            
            step=step+1
            
            self.features_info[str(step)+'.filterbyChi2Oneway']=keep_col
            
            print((str(step)+'.filterbyChi2Oneway').rjust(43)+'Done.'.rjust(27,'_'))
            
            
        # #fliter by pi
        # if self.pi_limit is not None and len(keep_col):
            
        #     keep_col=LgbmPISelector(threshold=self.pi_limit,early_stopping_rounds=None,random_state=123,
        #                             clf_params={'n_estimators':250,'max_depth':3,'learning_rate':0.05}).fit(X[keep_col],y,
        #                                                                                                     sample_weight=sample_weight).keep_col            
        #     step=step+1
            
        #     self.features_info[str(step)+'.filterbyPermutationImp']=keep_col

        #     print((str(step)+'.filterbyPermutationImp').rjust(47)+'Done.'.rjust(23,'_')) 
        
        # #fliter by LOFO_importannce
        # if self.lofoi_limit is not None and len(keep_col):
            
        #     keep_col=self._filterbyLofoimp(X[keep_col],y,lofo_imp=self.lofoi_limit,sample_weight=sample_weight)
            
        #     step=step+1
            
        #     self.features_info[str(step)+'.filterbyLOFOImp']=keep_col

        #     print((str(step)+'.filterbyLOFOImp').rjust(40)+'Done.'.rjust(30,'_'))              
             
        
        #fliter by lgbm-tree-imp  
        if self.tree_imps is not None and len(keep_col):

            keep_col=self._filterByTrees(X[keep_col],y,self.tree_size,self.tree_imps,sample_weight=sample_weight)
            
            step=step+1
            
            self.features_info[str(step)+'.filterbyTrees']=keep_col
            
            print((str(step)+'.filterbyTrees').rjust(38)+'Done.'.rjust(32,'_'))
            
        
        #fliter by iv 
        if self.iv_limit is not None and len(keep_col):   
            
            keep_col=self._filterbyIV(X[keep_col],y,self.iv_limit,sample_weight=sample_weight)
            
            step=step+1
            
            self.features_info[str(step)+'.filterbyIV']=keep_col
            
            print((str(step)+'.filterbyIV').rjust(35)+'Done.'.rjust(35,'_'))
   
        
        print('Done'.center(100,'-'))
        
        #summary
        for key in self.features_info:
            
            print('Step:{},features remain:{}'.format(key,len(self.features_info[key])))
        
        #report
        if self.out_path: 
            
            self.preSelector_report=pd.concat([pd.Series(self.features_info[key],name=key) for key in self.features_info.keys()],axis=1)
                
            self._writeExcel() 
        
        self.keep_col=keep_col
        
        self._is_fitted=True
                    
        return self
    
    
    def _filterByNA(self,X,na_pct,ws=None):
        """
        使用缺失值比例筛选数值
        + 缺失值指代为np.nan,即函数只会识别float类型的np.nan为缺失值
        + 对于分类特征
            + string型中np.nan为字符'nan'，因此其不会被当作np.nan处理
            + object型中np.nan可以为float,因此其可以被当作np.nan处理
            + category型中np.nan可以为float,因此其可以被当作np.nan处理

        Parameters:
        ----------  
        X:pandas.DataFrame,X特征
        unique_pct:float,唯一值比例阈值，大于等于阈值的列将被筛除
        ws:Series,样本权重
            
        Return:
        ---------- 
        list,筛选后的列名
        """
        
        if ws is None:
            
            nan_info=X.isnull().sum().div(len(X))
            
            keep_col=nan_info[nan_info<=na_pct].index.tolist()
            
        else:
            
            ws=np.array(ws)
    
            mask=np.transpose(ws*np.transpose(X.isnull().to_numpy())).sum(0)/ws.sum()<=na_pct
            
            keep_col=X.columns[mask].tolist()
        
        return keep_col
    
    
    def _filterByUnique(self,X,unique_pct,ws=None):
        """
        使用唯一值比例筛选数值、分类特征
        
        Parameters:
        ----------  
        X:pandas.DataFrame,X特征
        unique_pct:float,唯一值比例阈值，大于等于阈值的列将被筛除
        ws:Series,样本权重
            
        Return:
        ---------- 
        list,筛选后的列名
        """
  
        X=X.select_dtypes(include=['object','number'])
        X_oth=X.select_dtypes(exclude=['object','number'])
            
        
        if X.columns.size:
            
            if ws is None:
            
                X_unique_pct=X.apply(lambda x:x.value_counts(dropna=False,normalize=True).max())   
            
            else:
                
                ws=pd.Series(ws,index=X.index)
                
                X_unique_pct=X.apply(lambda x:ws.groupby(x,dropna=False).sum().max()/ws.sum()) 
                
            return X_unique_pct[X_unique_pct<unique_pct].index.tolist()+X_oth.columns.tolist()
        
        else:
            
            return X_oth.columns.tolist()

    
    def _fliterByVariance(self,X,variance,ws=None):
        """
        使用方差筛选数值特征，
        + 本函数只筛选数值特征，分类特征不进行处理直接输出
        
        Parameters:
        ----------  
        X:pandas.DataFrame,X特征
        variance:float,方差阈值，小于等于阈值的列将被筛除
        ws:Series,样本权重
            
        Return:
        ---------- 
        list,筛选后的列名
        """

        X_numeric=X.select_dtypes('number')
        X_oth=X.select_dtypes(exclude='number')
        
        def var_ws(vals, ws):

            na_mask=np.isnan(vals)
            vals=vals[~na_mask]
            ws=ws[~na_mask]

            if vals.size:

                avg = np.average(vals, weights=ws)    
                var = np.average((vals-avg)**2, weights=ws)  

            else:

                var=np.nan

            return (var)
        
        #drop constant columns
        X_numeric=X_numeric.loc[:,X_numeric.apply(lambda col: False if col.unique().size==1 else True)]
        
        if X_numeric.columns.size:
            
            if ws is None:
            
                X_var=X_numeric.var(ddof=0)
                
            else:
                
                ws=np.array(ws)
                
                X_var=X_numeric.apply(lambda x:var_ws(x,ws))
            
            return X_var[X_var>variance].index.tolist()+X_oth.columns.tolist()
        
        else:
            
            return X_oth.columns.tolist()
        

    def _filterByChif(self,X,y,alpha=0.05,method='fpr'):
        
        """
        分类模型:使用卡方与F检验的显著性进行特征初筛,参考:
        
            sklearn.feature_selection.chi2
            sklearn.feature_selection.f_classif
        
        从统计角度上，此两种方法都要求数据独立同分布，且f_classif默认样本的方差出自同一总体,若偏离要求太多则p-value意义不大。
        因此只使用这两种方法进行特征的初步筛选，目的在于筛选掉非常不重要的特征
        
        筛选依据为这些检验的显著性alpha值或其修正值,并非统计量的值(卡方值或F值),参考:
        
            sklearn.feature_selection.SelectFpr
            sklearn.feature_selection.SelectFdr
            sklearn.feature_selection.SelectFwe  
        
        注意:本过程不支持sample_weight
        
        Parameters:
        ----------  
        X:pandas.DataFrame,X特征
        y:pandas.Seires,目标特征
        alpha:float,显著性水平
        method:str,为减少第一类错误与第二类错误(误报，漏报)，可采用修正的p-value:
            'fpr':不修正,使用原始的p-value
            'fdr':基于估计的False Discovery rate修正p-value,该过程中p值越低的特征相应的显著性要求会比设定水平更高
            'fwe':基于Family-wise error rate修正p-value,该过程中所有特征的显著性要求都会比设定水平更高    
            筛选强度上,fpr<fdr<fwe,即fwe会筛选掉最多的特征，fpr会筛选掉最少的特征
            
        Return:
        ---------- 
        list,筛选后的列名
        """
        
        #drop constant columns
        X=X.loc[:,X.apply(lambda col: False if col.unique().size==1 else True)]        
        
        X_categoty=X.select_dtypes('object')
        
        X_numeric=X.select_dtypes('number')
        
        X_oth=X.select_dtypes(exclude=['object','number'])
        
        if method=='fpr':
            
            base_selector=SelectFpr
        
        elif method=='fdr':
            
            base_selector=SelectFdr
            
        elif method=='fwe':
            
            base_selector=SelectFwe
        
        #filter by chi2
        if X_categoty.columns.size:      
            
            X_categoty_encode=OrdinalEncoder().fit_transform(X_categoty.replace(np.nan,'missing'))
            
            selected=base_selector(chi2,alpha=alpha).fit(X_categoty_encode,y)
            
            cate_cols=X_categoty.columns[selected.get_support()].tolist()
            
        else:
            
            cate_cols=[]
            
        
        #filter by oneway
        if X_numeric.columns.size:
            
            cols_fill=X_numeric.median().to_dict()
            
            sp_mask=base_selector(f_classif,alpha=alpha).fit(X_numeric.fillna(np.finfo(np.float32).max),y).get_support() #fillna with inf
            
            sm_mask=base_selector(f_classif,alpha=alpha).fit(X_numeric.fillna(cols_fill),y).get_support() #fillna with median
            
            sn_mask=base_selector(f_classif,alpha=alpha).fit(X_numeric.fillna(-np.finfo(np.float32).max),y).get_support() #fillna with -inf

            num_cols=X_numeric.columns[sp_mask+sm_mask+sn_mask].tolist() #drop features with False among all data
            
        else:
                
            num_cols=[]
            
        
        return cate_cols+num_cols+X_oth.columns.tolist()
    

    def _filterByTrees(self,X,y,tree_size,tree_imps,sample_weight=None):
        
        """
        使用LightGbm的Split重要性筛选特征
        
        Parameters:
        ----------  
        X:pandas.DataFrame,X特征
        y:pandas.Seires,目标特征
        tree_size,int,数个数
        tree_imps,int,特征重要性阈值，小于等于阈值的列将被剔除
        sample_weight:Series,样本权重
            
        Return:
        ---------- 
        list,筛选后的列名
        """

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
                random_state=123,
                subsample=0.7,
                colsample_bytree=1,
                n_jobs=effective_n_jobs(self.n_jobs)
            ).fit(X_new,y,sample_weight=sample_weight,categorical_feature=X_category_encode.columns.tolist())         

            lgb_imps=lgb.booster_.feature_importance(importance_type='split')
        
            return X_new.columns[lgb_imps>tree_imps].tolist()+X_oth.columns.tolist()
        
        elif X_numeric.columns.size:
            
            lgb=sLGBMClassifier(
                boosting_type='gbdt',
                objective = 'binary',
                learning_rate=0.1,
                n_estimators=tree_size,
                random_state=123,
                subsample=0.7,
                colsample_bytree=1,
                n_jobs=effective_n_jobs(self.n_jobs)
            ).fit(X_numeric,y,sample_weight=sample_weight)         

            lgb_imps=lgb.booster_.feature_importance(importance_type='split')
        
            return X_numeric.columns[lgb_imps>tree_imps].tolist()+X_oth.columns.tolist()
        
        else:
            
            return X_oth.columns.tolist()
        
        
    # def _filterbyLofoimp(self,X,y,lofo_imp=0,sample_weight=None):
        
    #     """
    #     使用Leave one out重要性筛选特征:
    #         + 特征x的重要性=全量特征模型表在交叉验证集表现的均值-移除特征x后的模型在交叉验证集表现的均值
    #         + 基模型为lightgbm
    #         + 模型表现衡量标准为roc_auc
            
    #     该重要性计算量较大请慎用
        
    #     LOFOImportance:https://github.com/aerdem4/lofo-importance
        
    #     Parameters:
    #     ----------  
    #     X:pandas.DataFrame,X特征
    #     y:pandas.Seires,目标特征
    #     lofo_imp,int,特征重要性阈值，小于等于阈值的列将被剔除
    #     sample_weight:Series,样本权重
            
    #     Return:
    #     ---------- 
    #     list,筛选后的列名
    #     """
        
    #     X=X.apply(lambda x:x.astype('category') if x.dtypes=='object' else x)  
        
    #     dt=Dataset(X.join(y),y.name,features=X.columns.tolist())
           
    #     fimp=LOFOImportance(dt,'roc_auc',cv=5,n_jobs=-1,
    #                 fit_params={'sample_weight':sample_weight}).get_importance()
        
    #     return fimp[fimp['importance_mean']>lofo_imp]['feature'].tolist()
        
    
    def _filterbyIV(self,X,y,iv_limit,sample_weight=None): 
        """
        使用分箱IV重筛选特征:
            + 等频30箱
        
        Parameters:
        ----------  
        X:pandas.DataFrame,X特征
        y:pandas.Seires,目标特征
        iv_limit,int,IV重要性阈值，小于等于阈值的列将被剔除
        sample_weight:Series,样本权重
            
        Return:
        ---------- 
        list,筛选后的列名
        """
        
        _,vtabs=binFreq(X,y,ws=sample_weight,bin_num_limit=30)
        
        iv_t=pd.Series({key:vtabs[key]['total_iv'].max() for key in vtabs},name='iv_infos',dtype='float')

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