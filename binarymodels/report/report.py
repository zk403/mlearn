#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 21:10:13 2021

@author: zengke
"""
import pandas as pd
from sklearn.base import TransformerMixin,BaseEstimator
import numpy as np
#import time
from glob import glob
import os
import warnings



class EDAReport(BaseEstimator):
    
    def __init__(self,categorical_col=None,numeric_col=None,miss_value=[np.nan,'nan'],is_nacorr=False,out_path=None):
        """ 
        产生数据质量报告
        Params:
        ------
            categorical_col:list,类别特征列名
            numeric_col:list,连续特征列名
            miss_value:list,缺失值指代值
            is_nacorr:bool,是否输出缺失率相关性报告
            out_path:str or None,将数据质量报告输出到本地工作目录的str文件夹下，None代表不输出            
        
        Attributes:
        -------
            num_report:pd.DataFrame,连续特征质量报告
            char_report:pd.DataFrame,分类特征质量报告
            na_report:pd.DataFrame,单特征缺失率报告
            nacorr_report:pd.DataFrame,缺失率相关性报告
        """
        self.categorical_col = categorical_col
        self.numeric_col = numeric_col
        self.miss_value=miss_value
        self.is_nacorr=is_nacorr
        self.out_path=out_path
        
    def fit(self, X, y=None):
        
        if X.size:
            #填充缺失值
            self.X=X.copy().replace(self.miss_value,np.nan)   
            
            #产生报告
            self.num_report=self.num_info()
            self.char_report=self.char_info()
            self.na_report=self.nan_info()        
            if self.is_nacorr:
                self.nacorr_report=self.nan_corr()
            
            #输出报告    
            if self.out_path:            
                if not glob(self.out_path):
                    os.mkdir(self.out_path)
                
                #now=time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
                #writer = pd.ExcelWriter(self.out_path+'/data_report'+now+'.xlsx')
                writer = pd.ExcelWriter(self.out_path+'/EDAReport.xlsx')
    
                self.num_report.to_excel(writer,sheet_name='NUM')
                self.char_report.to_excel(writer,sheet_name='CHAR')        
                self.na_report.to_excel(writer,sheet_name='NAN')
                if self.is_nacorr:
                    self.nacorr_report.to_excel(writer,sheet_name='NAN_corr')
            
                writer.save()     
                print('to_excel done')   
                                    
        return self
    
    def transform(self, X):       
        
        if X.size:
            
            return X
        
        else:
            
            warnings.warn('0 rows in input X,return None')
            
            return pd.DataFrame(None)
        
    

    def num_info(self):
        
        """ 数据质量报告-数值特征
        """
        
        data =self.X
        
        if self.numeric_col is None:
            num_col=data.select_dtypes(include='number').columns
        else:
            num_col=self.num_col

        report=data[num_col].describe(percentiles=[0.2,0.4,0.6,0.8]).T.assign(
            MissingRate=data.apply(   
                lambda col:col.isnull().sum()/col.size    
               )
        ).reset_index().rename(columns={'index':'VarName'})
        
        return report
    

    def char_info(self):
        """ 数据质量报告-分类特征
        """
        
        data=self.X
        
        if self.categorical_col is None:
            category_col=data.select_dtypes(include=['object','category']).columns
        else:
            category_col=self.category_col
    
        report=pd.DataFrame()
        for Col in category_col:

            ColTable=data[Col].value_counts().sort_index().rename('Freq').reset_index() \
                .rename(columns={'index':'Levels'}).assign(VarName=Col)[['VarName','Levels','Freq']]

            ColTable['Percent']=ColTable.Freq/data[Col].size #占比
            ColTable['CumFreq']=ColTable.Freq.cumsum() #累计(分类特征类别有次序性时有参考价值)
            ColTable['CumPercent']=ColTable.CumFreq/data[Col].size #累计占比(分类特征类别有次序性时有参考价值)

            report=pd.concat([report,ColTable])
        
        return report
    
   
    def nan_info(self):
        """ 数据质量报告-缺失特征
        """        
        data=self.X
        
        report=pd.DataFrame(
        {'N':data.apply(   
            lambda col:col.size      
           ),
        'Missings':data.apply(   
            lambda col:col.isnull().sum()   
           ),
        'MissingRate':data.apply(   
            lambda col:col.isnull().sum()/col.size    
               ),
        'dtype':data.dtypes
            } ).reset_index().rename(columns={'index':'VarName'})
    
        return report
    
    #@property     
    def nan_corr(self):
        """ 数据质量报告-缺失特征相关性
        """        
        data=self.X        
        nan_info=data.isnull().sum()
        nan_corr_table=data[nan_info[nan_info>0].index].isnull().corr()
        return nan_corr_table



class businessReport(TransformerMixin):
    
    def __init__(self,target,index,columns,rename_columns=None,rename_index=None,out_path=None):
        """ 
        产生业务报告
        Params:
        ------
            target:str,目标变量名
            index:list,汇总到行的列名
            columns:list,汇总到列的列名
            rename_columns:list,重命名汇总到行的列名
            rename_index:list,重命名汇总到列的列名     
            out_path:将报告输出到本地工作目录的str文件夹下，None代表不输出 
        
        Attributes:
        -------
            ptable:pd.DataFrame,业务透视表
        """
        self.target = target
        self.index = index
        self.columns=columns
        self.rename_columns=rename_columns
        self.rename_index=rename_index
        self.out_path=out_path
        
    def fit(self, X, y=None):
        

        if X.size:
            
            values=self.target
            index=self.index
            columns=self.columns
            rename_columns=self.rename_columns
            rename_index=self.rename_index
            
            aggfunc=['count','sum','mean']
            rename_aggfunc=dict(zip(aggfunc,['#','event#','event%#']))

            self.ptable=pd.pivot_table(X,index=index,columns=columns,
                              values=values,aggfunc=aggfunc,
                              margins=True).rename(columns=rename_aggfunc,level=0)

            if rename_index:
                self.ptable.index.names=rename_index

            if rename_columns:
                self.ptable.columns.names=[None]+rename_columns

            
            #输出报告    
            if self.out_path:            
                
                if not glob(self.out_path):
                    
                    os.mkdir(self.out_path)
                
                #now=time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
                #writer = pd.ExcelWriter(self.out_path+'/BusinessReport'+now+'.xlsx')   
                writer = pd.ExcelWriter(self.out_path+'/BusinessReport.xlsx')                   
                self.ptable.to_excel(writer,sheet_name='BusinessReport')
            
                writer.save()     
                print('to_excel done')                       
                                    
        return self
    
    def transform(self, X):     
        
        if X.size:
            
            return X
        
        else:
            
            warnings.warn('0 rows in input X,return None')

            return pd.DataFrame(None)


