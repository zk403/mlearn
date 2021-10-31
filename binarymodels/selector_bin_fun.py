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
from itertools import product,groupby
from sklearn.cluster import KMeans


class binAdjusterKmeans(TransformerMixin):
    
    def __init__(self,breaks_list,combine_ratio=0.1,bin_limit=5,seed=123,special_values=[np.nan],n_jobs=-1,verbose=0):
        """ 
        基于Kmeans的分箱调整算法:一种自动非优化分箱算法        
        一般通过细分箱后各个分箱的BadRate近似时需要合并,本算法可自动实现这一过程。
        
        Params:
        ------
        
            breaks_list:分箱字典结构,{var_name:[bin],...},支持scorecardpy与toad的breaks_list结构 
            combine_ratio,float,分箱合并阈值,在bin_limit=1的情况下,阈值越大合并的箱数越多,建议范围0.01-0.1
            bin_limit:int,分箱合并阈值,最终分箱的最低分箱数限制,越低则合并的箱数越多,建议4-6,同时若特征初始分箱小于bin_limit则不执行合并算法
            seed:int,kmeans随机种子,
                + 本算法在合并差距较大的barprob箱时,kmeans的随机性会被放大导致合并结果不可复现,设定seed值以复现合并结果
                + 设定合理的combine_ratio与bin_limit可极大的降低kmeans的随机性
            special_values,特殊值指代值,需与breaks_list一致            
            n_jobs,int,并行数量,默认-1(所有core),在数据量较大的前提下可极大提升效率
            verbose,int,并行信息输出等级        
            
        Attributes:
        -------
        """
        self.combine_ratio = combine_ratio       
        self.bin_limit=bin_limit
        self.breaks_list=breaks_list
        self.special_values=special_values
        self.seed=seed
        self.n_jobs=n_jobs
        self.verbose=verbose
        
    
    def fit(self, X, y):
        
        if X.size:
            
            breaks_list=self.get_Breaklist_sc(self.breaks_list,X,y)
            
            parallel=Parallel(n_jobs=self.n_jobs,verbose=self.verbose)
            
            col_break=parallel(delayed(self.combine_badprob_kmeans)(X,y,
                                                                    self.combine_ratio,
                                                                    self.bin_limit,col,
                                                                    breaks_list[col],
                                                                    self.seed,
                                                                    self.special_values)
                               for col in list(breaks_list.keys()))     
            
            self.breaks_list_adj={col:breaks for col,breaks in col_break}
                                    
        return self
    
    
    def transform(self, X):       
        
        if X.size:
            
            return X
        
        else:
            
            warnings.warn('0 rows in input X,return None')
            
            return pd.DataFrame(None)
        
    
    def combine_badprob_kmeans(self,X,y,combine_ratio,bin_limit,col,breaks_list_sc_var,random_state=123,special_values=['nan',np.nan]):    
    
        global var_bin,res_km_s
        var_raw=X[col].replace(special_values,np.nan)
    
        if is_string_dtype(var_raw):
            
            #填充缺失值
            var_cut_fillna=pd.Series(np.where(var_raw.isnull(),'missing',var_raw),
                                      index=var_raw.index,
                                      name='bin')
            
            #分类特征进行原始数据到其breaklist的映射
            var_code_raw=var_cut_fillna.unique().tolist()
                                       
            map_codes=self.raw_to_bin_sc(var_code_raw,breaks_list_sc_var,special_values)
                 
            var_map=var_cut_fillna.map(map_codes)
    
            #初始化停止条件
            combine_ratio_count=True
            n_clusters=len(breaks_list_sc_var)
            iter_times=len(breaks_list_sc_var)
            iters=0
            breaks=breaks_list_sc_var
            
            #若初始分箱数小于阈值则不进行合并
            if n_clusters<bin_limit:
                
                return col,breaks
            
            else:
            
                #当combine_ratio_count为0 或 k值(最终分箱数)小于指定值 或 迭代次数达到限定时停止循环时停止迭代
                while combine_ratio_count and n_clusters>bin_limit and iters<iter_times:
    
                    #print(combine_ratio_count,bin_num,breaks,n_clusters)
    
                    #计算badrate
                    var_bin=pd.concat([var_map,y],axis=1).groupby(var_map)[y.name].mean().reset_index().rename(
                                        columns={y.name:'badprob'}
                                    )            
                    var_bin['bin']=var_bin['bin'].astype('str')
    
                    #kmeans合并分箱
                    n_clusters_unique=var_bin['badprob'].unique().size
    
                    #combine_ratio_count>0时调低n_clusters
                    if combine_ratio_count:
                        #防止n_clusters数值大于样本数
                        if n_clusters_unique<n_clusters:
    
                            n_clusters=n_clusters_unique-1
    
                        else:
                            n_clusters=n_clusters-1
    
                    #combine_ratio_count=0时n_clusters不变        
                    else:
                        n_clusters=n_clusters_unique
    
                    #防止n_clusters为0的情况
                    if n_clusters<=1:
                        n_clusters=1
    
                    n_clusters=var_bin['badprob'].unique().size-1#起始K值为badrate箱数-1，即至少合并一个分箱
                    res_km=KMeans(n_clusters=n_clusters,random_state=random_state).fit_predict(var_bin[['badprob']])
                    res_km_s=pd.Series(res_km,var_bin.index,name='cluster')
    
                    #产生新的合并breaks
                    breaks=var_bin.groupby(res_km_s)['bin'].apply(lambda x : '%,%'.join(x)).tolist() #产生新的breaks
    
                    #计算combine_ratio
                    var_bin_ratio=var_bin['badprob'].diff(1).abs().div(var_bin.badprob+1e-10)
                    combine_ratio_count=var_bin_ratio.lt(combine_ratio).sum()
    
                    #映射新breaks到原数据  
                    var_code_raw=var_map.unique().tolist()
                    mapcode=self.raw_to_bin_sc(var_code_raw,breaks,special_values=['0.0%,%2.0%,%missing'])  
                    var_map=var_map.map(mapcode)
    
                    #重新分箱
        #             var_bin=pd.concat([var_map,y],axis=1).groupby(var_map)[y.name].mean().reset_index().rename(
        #                                 columns={y.name:'badprob'}
        #                             )            
        #             var_bin['bin']=var_bin['bin'].astype('str')
    
        #             bin_num=var_bin['bin'].size
    
                    iters=iters+1
    
                return col,breaks
        
        elif is_numeric_dtype(var_raw):
            
            #数值特征进行分箱并填补缺失值
            var_cut=pd.cut(var_raw,[-np.inf]+breaks_list_sc_var+[np.inf],duplicates='drop',right=False)   
            
            var_cut_fillna=pd.Series(np.where(var_cut.isnull(),'missing',var_cut),
                                      index=var_cut.index,
                                      name='bin') 
            
            #定义初始值
            combine_ratio_count=True
            n_clusters=len(breaks_list_sc_var)+1
            bin_num=999
            iter_times=len(breaks_list_sc_var)+1
            iters=0
            breaks=breaks_list_sc_var
            
            #若初始分箱数小于阈值则不进行合并
            if n_clusters<bin_limit:
                
                return col,breaks
            
            else:
            
                while combine_ratio_count and bin_num>bin_limit and iters<iter_times:
    
                    #计算分箱badprob
                    var_bin=pd.concat([var_cut_fillna,y],axis=1).groupby(var_cut_fillna)[y.name].mean().reset_index().rename(
                                    columns={y.name:'badprob'}
                                )
                    var_bin=var_bin[var_bin['bin']!='missing'] #移除缺失值
                    var_bin['bin']=var_bin['bin'].astype('str')            
    
                    #kmeans合并分箱
    
                    n_clusters_unique=var_bin['badprob'].unique().size
    
                    #combine_ratio_count>0时调低n_clusters
                    if combine_ratio_count:
                        #防止n_clusters数大于聚类样本数
                        if n_clusters_unique<n_clusters:
    
                            n_clusters=n_clusters_unique-1
    
                        else:
                            n_clusters=n_clusters-1
    
                    #combine_ratio_count=0时n_clusters不变      
    
                    else:
                        n_clusters=n_clusters_unique    
    
                    #防止n_clusters为0的情况
                    if n_clusters<=1:
                        n_clusters=1
    
                    res_km=KMeans(n_clusters=n_clusters,random_state=random_state).fit_predict(var_bin[['badprob']])
                    res_km_s=pd.Series(res_km,var_bin.index,name='cluster')
    
                    #生成需合并分箱的各个分组的index
                    g_index_list=var_bin.groupby(res_km_s)['bin'].apply(lambda x : x.index.sort_values().tolist()).tolist()
    
                    #计算combine_ratio_count
                    var_bin_ratio=var_bin['badprob'].diff(1).abs().div(var_bin.badprob+1e-10)
                    combine_ratio_count=var_bin_ratio.lt(combine_ratio).sum()
    
                    #从原始的breaklist中drop掉需合并的分箱点
    
                    index_drop=self.getindex(g_index_list)
                    #print(breaks,g_index_list,index_drop,var_bin)
                    breaks=self.drop_index_l(breaks,index_drop)
    
                    #映射新breaks到原数据
                    var_cut=pd.cut(var_raw,[-np.inf]+breaks+[np.inf],duplicates='drop',right=False)   
                    var_cut_fillna=pd.Series(np.where(var_cut.isnull(),'missing',var_cut),
                                          index=var_cut.index,
                                          name='bin')      
                    var_bin=pd.concat([var_cut_fillna,y],axis=1).groupby(var_cut_fillna)[y.name].mean().reset_index().rename(
                                    columns={y.name:'badprob'}
                                )
                    var_bin=var_bin[var_bin['bin']!='missing']
                    var_bin['bin']=var_bin['bin'].astype('str')
                    bin_num=var_bin['bin'].size
    
                    iters=iters+1
    
                return col,breaks
            
        #列类型为其他特殊情况    
        else:
            
            return col,breaks

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
    

    def getindex(self,g_index_list):
        
        """ 
        寻找list中值连续索引的位置并标记
        """ 
        
        ll=[]
        for lst in g_index_list:
            fun = lambda x: x[1]-x[0]
            for k, g in groupby(enumerate(lst), fun):
                l1 = [j for i, j in g]  # 连续数字的列表
                if len(l1) > 1:
                    ll.append(min(l1)) 
        return ll
    

    def drop_index_l(self,l,index_drop):
        """ 
        从list中剔除指定位置的元素
        """ 
  
        l=l.copy()
        
        for counter, index in enumerate(index_drop):
            index = index - counter
            l.pop(index)
            
        return l
    
    def get_Breaklist_sc(self,break_list,X,y):
        
        """
        将toad的breaklist结构转化为scorecardpy可用的结构
        """      
        
        
        #判断break_list是sc格式还是toad格式
        count=0
        for var_list in list(break_list.values()):
            
            for value in var_list:
                if isinstance(value,list):
                    count=count+1           
                break
            
        columns=list(break_list.keys())
        
        #toad格式时转换为sc格式
        if count>0:
        
            cate_colname=X[columns].select_dtypes(exclude='number')
            num_colname=X[columns].select_dtypes(include='number')

            break_list_sc=dict()

            #将toad的breaklist转化为scorecardpy的breaklist
            for key in break_list.keys():
                
                #分类列需调整格式
                if key in cate_colname and break_list[key]: 

                    bin_value_list=[]
                    
                    for value in break_list[key]:
                        #if 'nan' in value:
                        #    value=pd.Series(value).replace('nan','missing').tolist()
                        bin_value_list.append('%,%'.join(value))

                    break_list_sc[key]=bin_value_list
                
                #数值列默认
                elif key in num_colname and break_list[key]:
                    
                    break_list_sc[key]=break_list[key]
                
                #空breaklist调整格式
                else:

                    break_list_sc[key]=[-np.inf,np.inf]
        #sc格式
        else:   
            break_list_sc=break_list
                
        return break_list_sc   
  