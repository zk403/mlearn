# -*- coding: utf-8 -*-
from sklearn.base import TransformerMixin
#from category_encoders.ordinal import OrdinalEncoder
#import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype,is_string_dtype
from joblib import Parallel,delayed
import numpy as np
from itertools import product


class woeTransformer(TransformerMixin):
    
    """ 
    对数据进行WOE编码
        
    Params:
    ------
        
    varbin:BDMtools.varReport(...).fit(...).var_report_dict,dict格式,woe编码参照此编码产生
    n_jobs,int,并行数量,默认1(所有core),在数据量非常大，列非常多的情况下可提升效率但会增加内存占用，若数据量较少可设定为1
    verbose,int,并行信息输出等级        
    special_values,list,缺失值指代值,注意special_values必须与varbin的缺失值指代值一致，否则缺失值的woe编码将出现错误结果
    check_na:bool,为True时,若经woe编码后编码数据出现了缺失值，程序将报错终止   
            出现此类错误时多半是某箱样本量为1，或test或oot数据相应列的取值超出了train的范围，且该列是字符列的可能性极高     
            
    Attributes:
    -------   
    """        
    
    def __init__(self,varbin,n_jobs=1,verbose=0,special_values=[np.nan],check_na=True):
        
        self.varbin=varbin
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.check_na=check_na
        self.special_values=special_values
        
    def transform(self,X,y):
        """ 
        WOE转换
        """
        X=X.copy().replace(self.special_values,np.nan)        
            
        p=Parallel(n_jobs=self.n_jobs,verbose=self.verbose)
            
        res=p(delayed(self.woe_map)(X[key],self.varbin[key],np.nan,self.check_na) 
                              for key in self.varbin)
            
        X_woe=pd.concat({col:col_woe for col,col_woe in res},axis=1)
            
        return X_woe  
            
          
    def fit(self,X,y):
   
        return self      
    
    def woe_map(self,col,bin_df,special_values,check_na=True):
    
        if is_numeric_dtype(col):
            
            bin_df_drop= bin_df[~bin_df['breaks'].isin([-np.inf,'missing',np.inf])]
            
            woe_nan= bin_df[bin_df['breaks'].eq("missing")]['woe'][0]
            
            breaks=bin_df_drop['breaks'].astype('float64').tolist()
            
            woe=bin_df[~bin_df['breaks'].eq('missing')]['woe'].tolist()
    
            col_woe=pd.cut(col,[-np.inf]+breaks+[np.inf],labels=woe,right=False,ordered=False).astype('float32')

            col_woe=col_woe.fillna(woe_nan)
            
        elif is_string_dtype(col):
            
            breaks=bin_df.index.tolist();woe=bin_df['woe'].tolist()
            
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
    
    
