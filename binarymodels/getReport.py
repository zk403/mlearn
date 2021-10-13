#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:25:57 2020

@author: kezeng
"""


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import os
from sklearn.base import BaseEstimator,TransformerMixin


class getColmuns(TransformerMixin):

    
    def __init__(self,categorical_col=None,numeric_col=None,time_col=None,id_col=None,
                 drop_date=True,
                 num_as_float32=False,
                 category_as_str=True):
        """ 
        数据规范化
        Params:
        ------
            categorical_col:list,类别特征列名
            numeric_col:list,连续特征列名
            time_col:list,日期类型列名,规范化后被转换为datetime类型
            id_col:list,id列列名,若数据框自带id索引则可忽略此参数
            drop_date:bool,将日期特征剔除,默认True
            num_as_float32:bool,数值变量是否转换为float32类型,默认False
            category_as_str:bool,类别特征是否转换为object类型,默认True,若为False将转换为category类型
        Returns:
        ------
            pandas.dataframe
                已经规范化好的数据框
        
        """
        self.categorical_col = categorical_col
        self.numeric_col = numeric_col
        self.time_col = time_col
        self.id_col = id_col
        self.drop_date = drop_date
        self.num_as_float32 = num_as_float32
        self.category_as_str = category_as_str
        

    def transform(self, X):
        
        """ 自定义转换
        df:pandas.dataframe
        num_32:True时,数值列(float64)将转换为float32,可节省内存
        """
        
        X = X.copy()
    
        if self.id_col:
            id_data=X[self.id_col]
        else:
            id_data=None
        
        if self.numeric_col and self.num_as_float32==True:
            num_data= X[self.numeric_col].astype('float32')
        elif self.numeric_col and self.num_as_float32==False:
            num_data= X[self.numeric_col].astype('float64')
        else:
            num_data=None
            
        if self.time_col and self.drop_date==False:
            time_data=X[self.time_col].replace('[^0-9]','',regex=True).astype('datetime64')              
        else:
            time_data=None
            
        if self.categorical_col:
            if self.category_as_str==True:
                cate_data=X[self.categorical_col].astype('str')
            else:
                cate_data=X[self.categorical_col].astype('category') 
        else:
            cate_data=None        
            
        if self.id_col:
            full_data=pd.concat([pd.DataFrame(),time_data,cate_data,num_data,id_data],axis=1).set_index(self.id_col) #合并数据
        else:
            full_data=pd.concat([pd.DataFrame(),time_data,cate_data,num_data,id_data],axis=1) #合并数据
            
        return full_data

    def fit(self,X,y=None):
        return self



class getReport(BaseEstimator):
    
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
        
        #填充缺失值
        self.X=X.copy().replace(self.miss_value,np.nan)   
        
        #产生报告
        self.num_report=self.num_info()
        self.char_report=self.char_info()
        self.na_report=self.nan_info()        
        if self.is_nacorr:
            self.nacorr_report=self.nan_corr()
        
        #输出报告    
        if self.out_path != None:            
            if self.out_path not in os.listdir():
                os.mkdir(self.out_path)
            
            now=time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
            writer = pd.ExcelWriter(self.out_path+'/data_report'+now+'.xlsx')

            self.num_report.to_excel(writer,sheet_name='NUM')
            self.char_report.to_excel(writer,sheet_name='CHAR')        
            self.na_report.to_excel(writer,sheet_name='NAN')
            if self.is_nacorr:
                self.nacorr_report.to_excel(writer,sheet_name='NAN_corr')
        
            writer.save()     
            print('to_excel done')                
                                    
        return self
    
    #@property
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
    
    #@property    
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

            ColTable=data[Col].value_counts().rename('Freq').reset_index() \
                .rename(columns={'index':'Levels'}).assign(VarName=Col)[['VarName','Levels','Freq']]

            ColTable['Percent']=ColTable.Freq/data[Col].size #占比
            ColTable['CumFreq']=ColTable.Freq.cumsum() #累计(分类特征类别有次序性时有参考价值)
            ColTable['CumPercent']=ColTable.CumFreq/data[Col].size #累计占比(分类特征类别有次序性时有参考价值)

            report=pd.concat([report,ColTable])
        
        return report
    
    #@property    
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