#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:25:57 2020

@author: kezeng
"""


# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer,KNNImputer,MissingIndicator
import numpy as np
from pandas.api.types import is_numeric_dtype
import warnings
#import time


class dtStandardization(TransformerMixin):
    
    """ 
    数据规范化：处理原始数据中实体重复,内存占用,索引等问题
    
    Params:
    ------
    id_col:id列list
    col_rm:需删除的列名list
    downcast:是否对数据中的numeric类型数据进行降级处理(float64->float32),降级后数据的内存占用将减少但会损失精度。存在id_col时,其将不进行降级处理
    set_index:是否将id_col设定为pandas索引
    drop_dup:是否执行去重处理,
        + 列:重复列名的列将被剔除并保留第一个出现的列,
        + 行:当id_col存在时,其将按照id_col进行行去重处理,此时重复id的行将被剔除并保留第一个出现的行,否则不做任何处理
        注意此模块假定行或列标识重复时相应行或列的数据也是重复的,若行列标示下存在相同标示但数据不同的情况时慎用此功能
        
    Returns:
    ------
        pandas.dataframe
    
    """ 
    
    def __init__(self,id_col=None,col_rm=None,downcast=True,set_index=True,drop_dup=True):       
        
        self.id_col=id_col
        self.col_rm=col_rm
        self.downcast=downcast
        self.drop_dup=drop_dup
        self.set_index=set_index
        
        
    def transform(self, X):
    
        
        if not X.index.is_unique:
            
            raise IOError('X.index is not unique,recommend set unique single index')
            
        X = X.copy()
        
        
        if X.size:
            
            #remove columns if exists
            X=X.drop(np.unique(self.col_rm),axis=1) if self.col_rm else X
            
            #downcast and drop dups
            if self.id_col and self.downcast:
                
                X=X.apply(lambda x:pd.to_numeric(x,'ignore','float') if x.name not in self.id_col and is_numeric_dtype(x) else x)
                    
                X=X.loc[~X[self.id_col].duplicated(),~X.columns.duplicated()] if self.drop_dup else X
            
            elif not self.id_col and self.downcast:
                
                X=X.apply(lambda x:pd.to_numeric(x,'ignore','float') if is_numeric_dtype(x) else x)                
                    
                X=X.loc[:,~X.columns.duplicated()] if self.drop_dup else X
                            
            elif self.id_col and not self.downcast:                
                    
                X=X.loc[~X[self.id_col].duplicated(),~X.columns.duplicated()] if self.drop_dup else X
                
            elif not self.id_col and not self.downcast:
                
                X=X.loc[:,~X.columns.duplicated()] if self.drop_dup else X
                
            #set index if id exists
            if self.id_col and self.set_index:
                
                X=X.set_index(self.id_col)
                
            return X
        
        else:
            
            warnings.warn('0 rows in input X,return None')  
            
            return pd.DataFrame(None)     
    

    def fit(self,X,y=None):       
        
        
        return self 


class dtypeAllocator(TransformerMixin):
    
    """ 
    列类型分配器：将原始数据中的列类型转换为适合进行数据分析与建模的数据类型，请注意
              + 本模块不支持对complex、bytes类型的列进行转换
              + 本模块将pandas的无序category类视为object类型，若原始数据存在有序category类型列时其将被转换为数值int8型 
              + 本模块暂不支持对pd.Interval类型的列进行任何转换
    Params:
    ------
        dtypes_dict={}
            + dict():是否自动处理输入数据并最终转换为object、number、date三种类型
                + 初始数据中的数值类型数据(float,int,bool)将被全部转换为float类型数据
                + 初始数据中的字符类型数据(str)将被全部转换为object类型数据
                + 初始数据中的无序分类类型数据(category-unordered)将被全部转换为object类型数据
                + 初始数据中的有序分类类型数据(category-ordered)将顺序被全部转换为int8类型数据(0,1,2,3...),其与原始数据的对应关系将被保存在self.order_info中
                + 初始数据中的时间类型数据(datetime,datetimetz)类型数据将保持默认,可通过参数选择是否剔除掉日期型数据
                + 初始数据中的时间差类型数据(timedelta)类型数据将被转换为float,时间单位需自行指定,且作用于全部的timedelta类型
                + 其他类型的列与col_rm列将不进行转换直接输出
            + dict={'num':colname_list,'str':colname_list,'date':colname_list}
                + colname_list是列名列表,可以为None,代表无此类特征,注意各个类的列名列表不能出现交集与重复,否则将报错终止
                + 若所有colname_list的特征只是数据所有列的一部分，则剩下部分的列将不做转换
                + colname_list不能含有col_rm中的列,否则会报错终止
        col_rm=None or list,不参与转换的列的列名列表，其不会参与任何转换且最终会保留在输出数据中        
        dtype_num='float64',数值类型列转换方式，默认为float64，可以选择float32/float16，注意其可以有效减少数据的内存占用，但会损失数据精度。请注意数据中的数值id列,建议不进行32或16转换
        t_unit=‘1 D’,timedelta类列处理为数值的时间单位，默认天
        drop_date=False,是否剔除原始数据中的日期列，默认False
        precision=5,数值类数据的精度,precision=5代表保留小数点后5位小数，设定好设定此值以获得数值的近似结果，否则后续分析比如分箱、woe编码等会在一些极端条件下产生错误结果。

    Returns:
    ------
        pandas.dataframe
            已经规范化好的数据框
    
    """    

    
    def __init__(self,dtypes_dict={},col_rm=None,t_unit='1 D',dtype_num='float64',drop_date=False,precision=6):

        self.dtypes_dict=dtypes_dict
        self.dtype_num = dtype_num
        self.col_rm = col_rm
        self.drop_date=drop_date
        self.t_unit=t_unit
        self.precision=precision
        

    def transform(self, X):
        
        """ 自定义转换
        df:pandas.dataframe
        num_32:True时,数值列(float64)将转换为float32,可节省内存
        """
        
        X = X.copy()
        
        if X.size:

            X_rm=X[self.col_rm] if self.col_rm else None
            
            X=X.drop(self.col_rm,axis=1) if self.col_rm else X        
            
            if isinstance(self.dtypes_dict,dict) and not self.dtypes_dict:
                
                X_out=self.getXAuto(X)
                
            elif isinstance(self.dtypes_dict,dict) and self.dtypes_dict:
     
                X_out=self.getX(X)
            
            else:
                
                raise IOError("dtypes_dict={'num':colname_list,'str':colname_list,'date':colname_list} or {}")
                           
            return(pd.concat([X_rm,X_out],axis=1))            
        
        else:
            
            warnings.warn('0 rows in input X,return None')  
            
            return pd.DataFrame(None)     
    

    def fit(self,X,y=None):       
        
        return self
    
    
    def getX(self,X):
        
        col_num=np.unique(self.dtypes_dict['num']).tolist() if 'num' in self.dtypes_dict.keys() else []
  
        col_obj=np.unique(self.dtypes_dict['str']).tolist() if 'str' in self.dtypes_dict.keys() else []
                     
        col_date=np.unique(self.dtypes_dict['date']).tolist() if 'date' in self.dtypes_dict.keys() else []
                    
        columns=pd.Series(col_num+col_obj+col_date,dtype='str')
            
        if columns.size and columns.is_unique and not np.sum(np.isin(self.col_rm,columns)):
                    
            X_keep=X.drop(columns,axis=1)
            
            X_num=X[col_num].astype(self.dtype_num).apply(np.round,args=(self.precision,))
            
            X_obj=X[col_obj].astype('str')
            
            X_date=None if self.drop_date else X[col_date].replace('[^0-9]','',regex=True).astype('datetime64')
            
            X_out=pd.concat([X_keep,X_num,X_obj,X_date],axis=1)
            
        elif not columns.size:
            
            X_out=self.getXAuto(X)
            
        else:
            
            raise IOError("duplicated colnames or col_rm in colname_list")
            
        return X_out
    
    
    def getXAuto(self,X):
      
        #数值
        X_num=X.select_dtypes(include=['number','bool']).astype(self.dtype_num).apply(np.round,args=(self.precision,))        
         
        #字符
        X_obj=X.select_dtypes(include=['object'])
        
        #类别
        X_cat=X.select_dtypes(include=['category'])
 
        col_ordered_d=X_cat.select_dtypes(include=['category']).apply(lambda col:col.cat.ordered).to_dict()
        
        col_ordered=[i for i in col_ordered_d if col_ordered_d[i]]
        col_unordered=[i for i in col_ordered_d if not col_ordered_d[i]]
        
        X_cat_ordered=X_cat[col_ordered]
        X_cat_unordered=X_cat[col_unordered].astype('str')
        
        self.order_info={col:dict(zip(X_cat_ordered[col].cat.categories,
                                      [i for i in range(X_cat_ordered[col].cat.categories.size)])) for col in X_cat_ordered.columns}
        
        X_cat_ordered=X_cat_ordered.apply(lambda col:col.cat.codes).astype(self.dtype_num)                
        
        #日期
        X_date=None if self.drop_date else X.select_dtypes(include=['datetime','datetimetz'])
        
        #时间差
        X_tdiff=X.select_dtypes(include=['timedelta']).div(pd.to_timedelta(self.t_unit)).astype(self.dtype_num).apply(np.round,args=(self.precision,))
        
        #其他
        X_oth=X.select_dtypes(exclude=['number','bool','object','category','timedelta','datetime','datetimetz'])
        
        #合并
        X_all=pd.concat([X_num,X_obj,X_cat_ordered,X_cat_unordered,X_date,X_tdiff,X_oth],axis=1)
        
        return  X_all



class outliersTransformer(TransformerMixin):
    
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

    def __init__(self,columns=None,quantile_range=(0.01,0.99),na_option='keep'):

        self.columns=columns
        self.quantile_range = quantile_range
        
    def fit(self,X, y=None):    
        
        X=X.copy()
        
        if X.size:
        
            quantile_range=self.quantile_range
            
            if self.columns:
                self.quantile_data=X[self.columns].quantile([min(quantile_range),max(quantile_range)])        
            else:
                self.quantile_data=X.select_dtypes('number').quantile([min(quantile_range),max(quantile_range)])        
        
        return self
    
    def transform(self,X):
        
        if X.size:
        
            quantile_range=self.quantile_range 
            
            pd.options.mode.chained_assignment = None
            
            X=X.copy()
            
            for column in self.quantile_data.columns:
                X[column][X[column]<self.quantile_data.loc[min(quantile_range),column]]=self.quantile_data.loc[min(quantile_range),column]
                X[column][X[column]>self.quantile_data.loc[max(quantile_range),column]]=self.quantile_data.loc[max(quantile_range),column]
            
            return X    
                
        else:
            
            warnings.warn('0 rows in input X,return None')  
        
            return pd.DataFrame(None) 


class nanTransformer(TransformerMixin):
    
    """ 
    缺失值填补，集成sklearn.impute        
    注意本模块不支持除字符、数值以外（时间、日期、时间差类）列的填充，这些列将直接返回原始值
    Params:
    ------
    method:(str,str)应对连续特征和分类特征的缺失值填补方法,连续可选constant,mean,median,knn,most_frequent,分类特征可选constant,most_frequent
        + 'constant':以fill_value的设定值填补
        + 'mean':以均值的设定值填补
        + 'median':以中位数填补
        + 'knn':KNN填补,注意本方法中事前将不对数据进行任何标准化
        + 'most_frequent':众数填补
    missing_values:list,缺失值指代值,数据中无论分类与连续都适用此指代值,若不同列的缺失指代值存在冲突，请先处理好这些冲突再使用本模块
    fill_value:str,int,float,method=constant时的填补设定值
    indicator:bool,是否生成缺失值指代特征
    n_neighbors:knn算法中的邻近个数k
    weights_knn:str,knn算法中的预测权重，可选‘uniform’, ‘distance’
    dtype_num:str,填补后数值类型列的dtype
    
    Attributes
    ------
    
    Examples
    ------        
    """    
    
    def __init__(self,method=('constant','constant'),
                      missing_values=[np.nan],
                      fill_value=(-9999,'missing'),  
                      n_neighbors=10,
                      weights_knn='uniform',
                      indicator=False,dtype_num='float32'):

        self.missing_values=missing_values
        self.method=method
        self.fill_value=fill_value
        self.indicator=indicator
        self.weights_knn=weights_knn
        self.n_neighbors=n_neighbors
        self.dtype_num=dtype_num

        
    def fit(self,X, y=None):            
        
        return self
    
    def transform(self,X, y=None):
        
        X=X.copy().replace(self.missing_values,np.nan)
        
        if X.size:
            
            X_num=X.select_dtypes(include='number')
            X_str=X.select_dtypes(include='object')
            X_oth=X.select_dtypes(exclude=['number','object'])
            
            if X_oth.columns.size:                
                warnings.warn("column which its dtype not in ('number','object') will not be imputed")      

    
            if X_num.size:
                
                if self.method[0]=='knn':
                    
                    imputer_num=KNNImputer(missing_values=np.nan,
                                               n_neighbors=self.n_neighbors,
                                               weights=self.weights_knn).fit(X_num)
                    
                elif self.method[0] in ('constant','mean','median','most_frequent'):
                    
                    imputer_num=SimpleImputer(missing_values=np.nan,
                                           strategy=self.method[0],
                                           fill_value=self.fill_value[0]).fit(X_num) 
                    
                else:
                    
                    raise IOError("method for numcol in ('knn','constant','mean','median','most_frequent')")
                
                
                X_num_fill=pd.DataFrame(imputer_num.transform(X_num),
                                        columns=imputer_num.feature_names_in_,
                                        index=X.index,dtype=self.dtype_num                                       
                                        ) 
            else:
                
                X_num_fill=None
            
            
            if X_str.size:
                
                if self.method[1] in ('constant','most_frequent'):
                
                    imputer_str=SimpleImputer(missing_values=np.nan,
                                               strategy=self.method[1],
                                               fill_value=self.fill_value[1]).fit(X_str)  
                    
                else:
                    
                    raise IOError("method for strcol in ('constant','most_frequent')")
                    
                
                X_str_fill=pd.DataFrame(imputer_str.transform(X_str),
                                       columns=imputer_str.feature_names_in_,
                                       index=X.index,dtype='str'
                                       ) 
            else:
                
                X_str_fill=None
    
    
            if self.indicator:      
                
                na_s=X.isnull().sum()
                
                na_cols=na_s[na_s>0].index
                
                indicator_na=MissingIndicator(missing_values=np.nan).fit(X[na_cols]) 
    
                X_na=pd.DataFrame(indicator_na.transform(X[na_cols]),
                                       columns=indicator_na.feature_names_in_,dtype='int8',
                                       index=X.index)
            else:
                
                X_na=None                      
            
            return pd.concat([X_oth,X_num_fill,X_str_fill,X_na],axis=1)
        
        else:
            
            warnings.warn('0 rows in input X,return None')  
            
            return pd.DataFrame(None)           