# -*- coding: utf-8 -*-
from sklearn.base import TransformerMixin
#from category_encoders.ordinal import OrdinalEncoder
#import numpy as np
import pandas as pd
import copy
from pandas.api.types import is_numeric_dtype,is_string_dtype
from joblib import Parallel,delayed,effective_n_jobs
import numpy as np
from BDMLtools.fun import raw_to_bin_sc,Specials
from BDMLtools.base import Base

class woeTransformer(Base,Specials,TransformerMixin):
    
    """ 
    对数据进行WOE编码
        
    Params:
    ------
        
    varbin:BDMLtools.varReport(...).fit(...).var_report_dict,dict格式,woe编码参照此编码产生       
    special_values,特殊值指代值,若数据中某些值或某列某些值需特殊对待(这些值不是np.nan)时设定
            请特别注意:special_values必须与产生varbin的函数的special_values一致，否则special_values的woe编码将出现错误结果
            + None,保证数据默认
            + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
            + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
    woe_missing=None,float,缺失值的woe调整值，默认None即不调整.当missing箱样本量极少时，woe值可能不具备代表性，此时可调整varbin中的woe替换值至合理水平，例如设定为0
            经过替换后的varbin=保存在self.varbin中.本模块暂不支持对不同特征的woe调整值做区别处理，所有特征的woe调整值均为woe_missing            
    woe_special=None,float,特殊值的woe调整值,默认None即不调整.当special箱样本量极少时，woe值可能不具备代表性，此时可调整varbin中的woe替换值至合理水平，例如设定为0  
            经过替换后的varbin=保存在self.varbin中.本模块暂不支持对不同特征的woe调整值做区别处理，所有特征的woe调整值均为woe_special
    distr_limit=0.01,float,当woe_missing或woe_special不为None时,若missing或special箱占比低于distr_limit时才执行替换
    check_na:bool,为True时,若经woe编码后编码数据出现了缺失值，程序将报错终止，可能的错误原因:   
            + 某箱样本量太少，且该列是字符列的可能性极高    
            + test或oot数据相应列的取值超出了train的范围，且该列是字符列的可能性极高  
            + special_value设定前后不一致(产生varbin的speical value与本模块的speical value要一致)
    dtype,可选'float32'与'float64',转换woe数据为np.float32/np.float64格式，breaks也会以np.float32/np.float64格式分段数据
        + 模块会使用varbin中的breaks分段数据，其本身为np.float64，因此fit中的数据的number列也必须为float64,否则会因为格式不一致产生精度问题
        + 若fit中的数据的number列为float32型，则请设定为float32以保证不因格式不一致而产生精度问题
        + 请不要在原始数据中共用不同的数值精度格式，例如float32与float64共用，int32与int64共用...，请使用bm.dtypeAllocator统一建模数据的格式
    n_jobs,int,并行数量,默认1(所有core),在数据量非常大，列非常多的情况下可提升效率但会增加内存占用，若数据量较少可设定为1
    verbose,int,并行信息输出等级 
            
    Attributes:
    -------   
    
    """        
    
    def __init__(self,varbin,n_jobs=1,verbose=0,special_values=None,woe_special=None,check_na=True,woe_missing=None,distr_limit=0.01,dtype='float64'):
        
        self.varbin=varbin
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.check_na=check_na
        self.special_values=special_values
        self.woe_missing=woe_missing
        self.woe_special=woe_special
        self.distr_limit=distr_limit
        self.dtype=dtype
        
    def transform(self,X,y=None):
        """ 
        WOE转换
        """
        self._check_param_dtype(self.dtype)
        
        self._check_X(X)

        self.varbin=copy.deepcopy(self.varbin)
        
        if isinstance(self.woe_missing,(int,float)):        
            
            
            for key in self.varbin:
            
                if 'missing' in self.varbin[key].index.tolist() and self.varbin[key].loc['missing','count_distr']<self.distr_limit:
                    
                    self.varbin[key].loc['missing','woe'] = self.woe_missing

        elif self.woe_missing is None:
            
            pass
        
        else:
            
            raise ValueError("woe_missing in (None,int,float).")
            
            
        if isinstance(self.woe_special,(int,float)):        
            
            
            for key in self.varbin:
            
                if 'missing' in self.varbin[key].index.tolist() and self.varbin[key].loc['missing','count_distr']<self.distr_limit:
                    
                    self.varbin[key].loc['special','woe'] = self.woe_special

        elif self.woe_special is None:
            
            pass
        
        else:
            
            raise ValueError("woe_special in (None,int,float).")            
            
        n_jobs=effective_n_jobs(self.n_jobs)   
                              
        p=Parallel(n_jobs=n_jobs,verbose=self.verbose)
        
        res=p(delayed(self._woe_map)(X[key],self.varbin[key],self.check_na,self.special_values,self.dtype) 
                              for key in self.varbin)
            
        X_woe=pd.concat({col:col_woe for col,col_woe in res},axis=1)
            
        return X_woe  
            
          
    def fit(self,X=None,y=None):
   
        return self      
    
    def _woe_map(self,col,bin_df,check_na=True,special_values=None,dtype='float64'):
        
        col=self._sp_replace_single(col,self._check_spvalues(col.name,special_values),fill_num=2**63,fill_str='special')
            
        if is_numeric_dtype(col):
            
            bin_df_drop= bin_df[~bin_df['breaks'].isin([-np.inf,'missing','special',np.inf])]
            
            woe_nan= bin_df[bin_df['breaks'].eq("missing")]['woe'][0]
            
            woe_sp= bin_df[bin_df['breaks'].eq("special")]['woe'][0]
            
            breaks=bin_df_drop['breaks'].astype('float64').tolist()
            
            woe=bin_df[~bin_df['breaks'].isin(['missing','special'])]['woe'].tolist()

            if special_values:
                
                breaks_cut=breaks+[2**63] if dtype=='float64' else np.float32(breaks+[2**63]).tolist()
    
                col_woe=pd.cut(col,[-np.inf]+breaks_cut+[np.inf],labels=woe+[woe_sp],right=False,ordered=False).astype(dtype)
      
                col_woe=col_woe.fillna(woe_nan)
                
            else:
                
                breaks_cut=breaks if dtype=='float64' else np.float32(breaks).tolist()
                
                col_woe=pd.cut(col,[-np.inf]+breaks_cut+[np.inf],labels=woe,right=False,ordered=False).astype(dtype)

                col_woe=col_woe.fillna(woe_nan)
                
            
        elif is_string_dtype(col):
            
            woe_nan= bin_df[bin_df['breaks'].eq("missing")]['woe'][0]
        
            woe_sp= bin_df[bin_df['breaks'].eq("special")]['woe'][0]
        
            breaks=bin_df[~bin_df['breaks'].isin(['missing','special'])].index.tolist()
        
            woe=bin_df[~bin_df['breaks'].isin(['missing','special'])]['woe'].tolist()
            
            if all(list(map(self._is_no_sp,[i.split('%,%') for i in breaks]))):
                
                breaks.append('special')
                woe.append(woe_sp)
                
            if all(list(map(self._is_no_na,[i.split('%,%') for i in breaks]))):
                
                breaks.append('missing')
                woe.append(woe_nan)    
                
            raw_to_breaks=raw_to_bin_sc(col.unique().tolist(),breaks)
            
            breaks_to_woe=dict(zip(breaks,woe))
            
            col_woe=col.map(raw_to_breaks).map(breaks_to_woe).astype('float32')            
            
        else:
            
            raise ValueError(col.name+"‘s dtype not in ('number' or 'str')")
            
        if check_na:
            
            if col_woe.isnull().sum()>0:
                
                raise ValueError(col.name+"_woe contains nans,bins in each variables in varbin should include all the possible values among with all the split data")
            
        return col.name,col_woe
    
    
    def _is_no_sp(self,strings):    
    
        if 'special' in strings:
            
            return False
        
        else:
            
            return True
    
    def _is_no_na(self,strings):
        
        if 'missing' in strings:
            
            return False
        
        else:
            
            return True
    
