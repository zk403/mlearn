#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:25:57 2020

@author: kezeng
"""


# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.impute import SimpleImputer,KNNImputer,MissingIndicator
import numpy as np
import time
import os


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
        
        #id列
        if self.id_col:
            id_data=X[self.id_col]
        else:
            id_data=None
            
        #数值列       
        if self.numeric_col and self.num_as_float32:
            num_data= X[self.numeric_col].astype('float32')
        elif self.numeric_col and not self.num_as_float3:
            num_data= X[self.numeric_col].astype('float64')
        elif not self.numeric_col and self.num_as_float32:
            num_data= X.select_dtypes(include='number').astype('float32')
        elif not self.numeric_col and not self.num_as_float32:
            num_data= X.select_dtypes(include='number').astype('float64')
        else:
            num_data=None
        
        #时间列            
        if self.time_col and not self.drop_date:
            time_data=X[self.time_col].replace('[^0-9]','',regex=True).astype('datetime64')              
        else:
            time_data=None
        
        #字符列                 
        if self.categorical_col and self.category_as_str:
            cate_data=X[self.categorical_col].astype('str')
        elif self.categorical_col and not self.category_as_str:
            cate_data=X[self.categorical_col].astype('category')
        elif not self.categorical_col and  self.category_as_str:
            cate_data=X.select_dtypes(include='object').astype('str')    
        elif not self.categorical_col and not self.category_as_str:
            cate_data=X.select_dtypes(include='object').astype('category')
        else:
            cate_data=None        
        
        #index    
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


class OutliersTransformer(TransformerMixin):
    
    def __init__(self,columns=None,quantile_range=(0.01,0.99),na_option='keep'):
        """ 
        分位数替代法处理异常值
        Params:
        ------
            columns:list,替代法的列名list,默认为全部数值列
            quantile_range:list,分位数上下限阈值
            na_option:str,{'keep'},缺失值处理方式,默认为keep即保留缺失值
        Returns
        ------
        pandas.dataframe
            已经处理好异常值的数据框
        Examples
        ------        
        """
        self.columns=columns
        self.quantile_range = quantile_range
        
    def fit(self,X, y=None):        
        quantile_range=self.quantile_range
        if self.columns:
            self.quantile_data=X[self.columns].quantile([min(quantile_range),max(quantile_range)])        
        else:
            self.quantile_data=X.select_dtypes('number').quantile([min(quantile_range),max(quantile_range)])        
        return self
    
    def transform(self,X):
        quantile_range=self.quantile_range        
        pd.options.mode.chained_assignment = None
        X=X.copy()
        for column in self.quantile_data.columns:
            X[column][X[column]<self.quantile_data.loc[min(quantile_range),column]]=self.quantile_data.loc[min(quantile_range),column]
            X[column][X[column]>self.quantile_data.loc[max(quantile_range),column]]=self.quantile_data.loc[max(quantile_range),column]        
        return X



class imputeNAN(TransformerMixin):
    
    def __init__(self,method=('constant','constant'),
                      missing_values=(np.nan,np.nan),
                      fill_value=(-9999,'missing'),  
                      n_neighbors=10,
                      weights_knn='uniform',
                      indicator=False):
        """ 
        缺失值填补，集成sklearn.impute        
        Params:
        ------
        method:(str,str)应对连续特征和分类特征的缺失值填补方法,连续可选constant,mean,median,knn,most_frequent,分类特征可选constant,most_frequent
            + 'constant':以fill_value的设定值填补
            + 'mean':以均值的设定值填补
            + 'median':以中位数填补
            + 'knn':KNN填补,注意本方法中事前将不对数据进行任何标准化
            + 'most_frequent':众数填补
        missing_values:str,int,float 缺失值指代值
        fill_value:str,int,float,method=constant时的填补设定值
        indicator:bool,是否生成缺失值指代特征
        n_neighbors:knn算法中的邻近个数k
        weights_knn:str,knn算法中的预测权重，可选‘uniform’, ‘distance’
        
        Attributes
        ------
        
        Examples
        ------        
        """
        self.method=method
        self.missing_values=missing_values
        self.fill_value=fill_value
        self.indicator=indicator
        self.weights_knn=weights_knn
        self.n_neighbors=n_neighbors

        
    def fit(self,X, y=None):  
        
        X=X.copy()
        X_num=X.select_dtypes(include='number')
        X_str=X.select_dtypes(include='object')
        
        if X_num.size and self.method[0]=='knn':
            
            self.imputer_num=KNNImputer(missing_values=self.missing_values[0],
                                       n_neighbors=self.n_neighbors,
                                       weights=self.weights_knn).fit(X_num)
            
        elif X_num.size and self.method[0]!='knn':
            self.imputer_num=SimpleImputer(missing_values=self.missing_values[0],
                                       strategy=self.method[0],
                                       fill_value=self.fill_value[0]).fit(X_num) 
            
        if X_str.size:
        
            self.imputer_str=SimpleImputer(missing_values=self.missing_values[1],
                                       strategy=self.method[1],
                                       fill_value=self.fill_value[1]).fit(X_str)        
        
        if self.indicator:
            na_s=X.isnull().sum().gt(0)
            self.na_cols=na_s[na_s>0].index
            
            self.indicator_na=MissingIndicator(missing_values=self.missing_values[1]).fit(X[self.na_cols])            
        
        return self
    
    def transform(self,X):
        
        X=X.copy()
        X_num=X.select_dtypes(include='number')
        X_str=X.select_dtypes(include='object')

        if X_num.size:        
            X_num_fill=pd.DataFrame(self.imputer_num.transform(X_num),
                                    columns=self.imputer_num.feature_names_in_,
                                    index=X.index
                                    )
        else:
            X_num_fill=None
        
        if X_str.size:
            X_str_fill=pd.DataFrame(self.imputer_str.transform(X_str),
                                   columns=self.imputer_str.feature_names_in_,
                                   index=X.index
                                   ) 
        else:
            X_str_fill=None


        if self.indicator:       

            X_na=pd.DataFrame(self.indicator_na.transform(X[self.na_cols]),
                                   columns=self.indicator_na.feature_names_in_,
                                   index=X.index
                                   )
        else:
            X_na=None
      
        return pd.concat([X_num_fill,X_str_fill,X_na],axis=1)
    
    
    
    
        