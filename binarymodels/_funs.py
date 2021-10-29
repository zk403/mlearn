#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 23:08:38 2021

@author: zengke
"""

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype,is_string_dtype
from sklearn.base import TransformerMixin
from joblib import Parallel,delayed
import warnings


class binAdjuster(TransformerMixin):
    
    def __init__(self,breaks_list,combine_ratio,special_values=[np.nan],n_jobs=-1,verbose=0):
        """ 
        自动分箱调整器
        Params:
        ------
            breaks_list:list,类别特征列名 
            combine_ratio,
            special_values,
            n_jobs,
            verbose,
        
        Attributes:
        -------
        """
        self.combine_ratio = combine_ratio        
        self.breaks_list=breaks_list
        self.special_values=special_values
        self.n_jobs=n_jobs
        self.verbose=verbose
        
    
    def fit(self, X, y):
        
        if X.size:
            
            parallel=Parallel(n_jobs=self.n_jobs,verbose=self.verbose)
            
            col_break=parallel(delayed(self.combine_badprob)(X,y,
                                                             self.combine_ratio,col,self.breaks_list[col],
                                                             10,
                                                             self.special_values
                                                                       )
                         for col in list(self.breaks_list.keys()))            
            
            self.breaks_list_adj={col:breaks for col,breaks in col_break}
                                    
        return self
    
    
    def transform(self, X):       
        
        if X.size:
            
            return X
        
        else:
            
            warnings.warn('0 rows in input X,return None')
            
            return pd.DataFrame(None)



    def interval_num(self,bin_list,method='adj'):
        
        #数值特征合并分段bin_list like [‘[-inf,1)’，‘[1,2)’，‘[2,inf]’]
        
        if bin_list:
        
            count=0
            l=[]
            while count<len(bin_list):
                #
                lower,upper=bin_list[count].split(',')
                lower_f=float(lower[1:])
                upper_f=float(upper[:-1])
                l.extend([lower_f,upper_f])
                count=count+1
    
            l_s=pd.Series(l,dtype='float64').replace([-np.inf,np.inf],np.nan).dropna()
    
            #调整的分箱上下区间一致时则剔除该区间
            if method=='adj':
                l_s=l_s.drop_duplicates(keep=False).tolist()
            #正常的分箱上下区间一致则需保留该区间
            else:
                l_s=l_s.drop_duplicates().tolist()
    
            return l_s
        
        else:
            
            return []


    def combine_badprob(self,X,y,combine_ratio,col,breaks_list_sc_var,iter_times=10,special_values=[np.nan]):    
        
        
        var_raw=X[col].replace(special_values,np.nan)
        
        
        #数值型列
        if is_numeric_dtype(var_raw):
            
            #数据处理
            var_cut=pd.cut(var_raw,[-np.inf]+breaks_list_sc_var+[np.inf],duplicates='drop',right=False)        
            var_cut=pd.Series(np.where(var_cut.isnull(),'missing',var_cut),
                              index=var_cut.index,
                              name=col)
            adj_rows=True
                
            iters=0
    
            #当需合并的行为0或迭代次数超过阈值时停止循环
            while adj_rows and iters<iter_times:                            
                
                #处理数据并分箱
                var_bin=X.join(y).groupby(var_cut)[y.name].mean().reset_index().rename(
                        columns={col:'bin',y.name:'badprob'}
                    )
                var_bin['bin']=var_bin['bin'].astype('str')
    
                var_bin_dropna=var_bin[var_bin['bin']!='missing']
    
                bin_num=var_bin_dropna.index.size
        
                print(var_bin)
                
                #当分箱数少于等于3时停止迭代
                    
                if bin_num<=3:
                    
                    warnings.warn('bin_num<=3,stop combining')
                    
                    if 'breaks_out' in locals():
                        
                        return (col,breaks_out)
                        
                    else:
                        
                        return (col,breaks_list_sc_var)
                        
                    break
                
                #计算badrate_Ratio,combine_ratio在这里定义
                var_bin_ratio=var_bin_dropna['badprob'].diff(1).abs().div(
                            var_bin_dropna.badprob
                    )
    
                index_adj=var_bin_dropna[var_bin_ratio.lt(combine_ratio)].index
    
                index_adj=set(index_adj.append(index_adj-1))
    
                index_not_adj=set(var_bin_dropna.index)-index_adj
                
                #需调整的bin长度
                adj_rows=len(index_adj)
                not_adj_rows=len(index_not_adj)
                
                #print(adj_rows,not_adj_rows)
                
                bin_list_adj=var_bin_dropna.loc[index_adj,'bin'].sort_index().tolist()
                bin_list_not_adj=var_bin_dropna.loc[index_not_adj,'bin'].sort_index().tolist()
                
                #print(breaks)
                
                breaks=self.interval_num(bin_list_not_adj,method='other')+self.interval_num(bin_list_adj,method='adj')
                
                #若breaks非空则返回调整结果
                if breaks:                
    
                    breaks=np.unique(
                                np.sort(breaks)
                                ).tolist()
        
                    breaks_out=breaks

                    var_cut=pd.cut(var_raw,np.unique([-np.inf]+breaks+[np.inf]).tolist(),duplicates='drop',right=False)
    
                    var_bin=X.join(y).groupby(var_cut)[y.name].mean().reset_index().rename(
                               columns={col:'bin',y.name:'badprob'}
                            )   
    
                    var_bin['bin']=var_bin['bin'].astype('str')
                        
                iters=iters+1
            
            #若迭代停止后breaks为空则返回之前结果或原始的breaks
            if breaks:
                
                return (col,breaks_out)
                
            #若迭代停止后breaks为空则返回之前结果或原始的breaks
            else:
                    
                if 'breaks_out' in locals():
                        
                    return (col,breaks_out)
                        
                else:    
                    
                    return (col,breaks_list_sc_var)

        #字符型列                
        elif is_string_dtype(var_raw):
            
            return (col,breaks_list_sc_var)
        
        else:
            
            return (col,breaks_list_sc_var)
            