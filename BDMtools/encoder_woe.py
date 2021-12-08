# -*- coding: utf-8 -*-
from sklearn.base import TransformerMixin
#from category_encoders.ordinal import OrdinalEncoder
#import numpy as np
import pandas as pd
import scorecardpy as sc
from pandas.api.types import is_numeric_dtype,is_string_dtype
from joblib import Parallel,delayed
import numpy as np
from itertools import product


class woeTransformer(TransformerMixin):
    
    """ 
    对数据进行WOE编码
        
    Params:
    ------
        
    varbin:scorecardpy格式的特征分析报告字典结构,{var_name:bin_df,...},由woebin产生
    method:str,可选'old'和'new'
        + ‘old’:使用sc.woebin_ply进行woe编码
        + 'new':使用内置函数进行woe编码,其优化了效率与内存使用,注意,使用方法new时需对原始数据进行缺失值及特殊值进行填充处理，详见bm.nanTransformer
                - 连续特征填充为有序值如-999,分类特可填充为missing
                - 此时需使用sc.woebin对填充后的数据进行处理产生varbin作为本模块入参，且sc.woebin的special_values参数必须设定为None
    n_jobs,int,并行数量,默认1(所有core),在数据量非常大的前提下可极大提升效率，若数据量较少可设定为1
    verbose,int,并行信息输出等级        
    check_na:bool,为True时,在使用方法new时，若经woe编码后编码数据出现了缺失值，程序将报错终止   
            出现此类错误时多半是某箱样本量为1，或test或oot数据相应列的取值超出了train的范围，且该列是字符列的可能性极高     
            
    Attributes:
    -------   
    """        
    
    def __init__(self,varbin,method='new',n_jobs=1,verbose=0,check_na=True):
        
        self.varbin=varbin
        self.method=method
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.check_na=check_na
        
    def transform(self,X,y):
        """ 
        WOE转换
        """
        if self.method=='old':
        
            X_woe=pd.DataFrame(index=X.index).join(sc.woebin_ply(dt=X.join(y),bins=self.varbin,no_cores=None))
            
            X_woe=X_woe[X_woe.columns[X_woe.columns.str.contains('_woe')]]
          
            X_woe.columns=X_woe.columns.str[0:-4]
            
            return X_woe
        
        elif self.method=='new':
            
            p=Parallel(n_jobs=self.n_jobs,verbose=self.verbose)
            
            res=p(delayed(self.woe_map)(X[key],self.varbin[key],np.nan,self.check_na) 
                              for key in self.varbin)
            
            X_woe=pd.concat({col:col_woe for col,col_woe in res},axis=1)
            
            return X_woe  
        
        else:
            
            raise IOError('method in ("old","new")')
            
          
    def fit(self,X,y):
   
        return self      
    
    def woe_map(self,col,bin_df,special_values,check_na=True):
    
        if is_numeric_dtype(col):
            
            bin_df_drop= bin_df[~bin_df['breaks'].isin(["-inf",'missing',"inf"])]
            
            woe_nan= bin_df[~bin_df['breaks'].isin(["missing"])]['woe'][0]
            
            breaks=bin_df_drop['breaks'].astype('float64').tolist()
            
            woe=bin_df['woe'].tolist()
    
            col_woe=pd.cut(col,[-np.inf]+breaks+[np.inf],labels=woe,right=False,ordered=False).astype('float32')

            col_woe=col_woe.fillna(woe_nan)
            
        elif is_string_dtype(col):
            
            breaks=bin_df['bin'].tolist();woe=bin_df['woe'].tolist()
            
            raw_to_breaks=self.raw_to_bin_sc(col.unique().tolist(),breaks,special_values=special_values)
            
            breaks_to_woe=dict(zip(breaks,woe))
            
            col_woe=col.replace(special_values,'missing').map(raw_to_breaks).map(breaks_to_woe).astype('float32')            
            
        else:
            
            raise ValueError(col.name+"‘s dtype not in ('number' or 'str')")
            
        if check_na:
            
            if col_woe.isnull().sum()>0:
                
                raise ValueError(col.name+"_woe contains nans")
            
        return col.name,col_woe
        
    
    def raw_to_bin_sc(self,var_code_raw,breaklist_var,special_values):
        
        """ 
        分箱转换，将分类特征的值与breaks对应起来
        1.只适合分类bin转换
        2.此函数只能合并分类的类不能拆分分类的类        
        """ 
        
        breaklist_var_new=[i.replace(special_values,'missing').unique().tolist()
                                   for i in [pd.Series(i.split('%,%')) 
                                             for i in breaklist_var]]
        
        map_codes={}
        
        for raw,map_code in product(var_code_raw,breaklist_var_new):
            
            
            #多项组合情况
            if '%,%' in raw:
                
                raw_set=set(raw.split('%,%'))
                
                #原始code包含于combine_code中时
                if not raw_set-set(map_code):

                    map_codes[raw]='%,%'.join(map_code)
            
            #单项情况
            elif raw in map_code:
                
                map_codes[raw]='%,%'.join(map_code)
            
            #print(raw,map_code)
   
        return map_codes
    
    
