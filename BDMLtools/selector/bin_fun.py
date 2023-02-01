#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 23:08:38 2021

@author: zengke
"""

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype,is_string_dtype,is_array_like
from joblib import Parallel,delayed,effective_n_jobs
import warnings
from itertools import groupby
from sklearn.cluster import KMeans
from scipy.stats import chi2,chi2_contingency
from warnings import warn
from BDMLtools.fun import raw_to_bin_sc,Specials
from BDMLtools.base import Base
from BDMLtools.report.report import varReportSinge


def R_pretty(low, high, n):
    '''
    pretty breakpoints, the same as pretty function in R
    
    Params
    ------
    low: minimal value 
    high: maximal value 
    n: number of intervals
    
    Returns
    ------
    numpy.ndarray
        returns a breakpoints array
    '''
    # nicenumber
    def nicenumber(x):
        exp = np.floor(np.log10(abs(x)))
        f   = abs(x) / 10**exp
        if f < 1.5:
            nf = 1.
        elif f < 3.:
            nf = 2.
        elif f < 7.:
            nf = 5.
        else:
            nf = 10.
        return np.sign(x) * nf * 10.**exp
    
    # pretty breakpoints
    d     = abs(nicenumber((high-low)/(n-1)))
    miny  = np.floor(low  / d) * d
    maxy  = np.ceil (high / d) * d
    return np.arange(miny, maxy+0.5*d, d)


def remove_outlier(col):
    '''
    remove outlier using iqr

    Params
    ------
    col: numpy.ndarray

    Returns
    ------
    numpy.ndarray
        returns col with outliers removed
    '''        

    iq=np.nanpercentile(col,[1, 25, 75, 99])

    iqr = iq[2] - iq[1]

    if iqr == 0:

        col_rm_outlier=col[(col >= iq[0]) & (col<=iq[3])]

    else:

        col_rm_outlier=col[(col >= iq[1]-3*iqr) & (col<=iq[2]+3*iqr)]  

    if np.unique(col_rm_outlier).size==1:

        col_rm_outlier=col

    return col_rm_outlier


def is_monotonic(col):    
    
    return np.all(np.diff(col) > 0) or np.all(np.diff(col) <0)



def check_monotonirc(col,y,cuts,ws=None):
    
    if is_array_like(ws):

        if ws.size!=y.size:

            raise ValueError('length of weight not equal to y')

        y = y * ws

        count = ws

    else:

        count=np.ones(col.size)  
    
    cuts_remain=[]
    
    for point in cuts:
        
        col_group=np.digitize(col,sorted([-np.inf]+[point]+cuts_remain+[np.inf]))
        
        g=np.zeros(0)
        b=np.zeros(0)
        count_g=np.zeros(0)

        for i in np.unique(col_group):

            y_g=y[col_group==i]
            unit_g=count[col_group==i]

            count_g=np.append(count_g,np.sum(unit_g))

            b=np.append(b,np.sum(y_g))
            g=np.append(g,np.sum(unit_g) - np.sum(y_g))   


        bad_prob=b/count_g    
        
        if is_monotonic(bad_prob):
            
            cuts_remain.append(point)            
      
    return np.array(cuts_remain)



def rm_edge_point(col,cut_off):

    cut_off=cut_off[(cut_off>col.min()) & (cut_off<col.max())]
    
    cut_remain=[]

    for point in cut_off:

        cut_g=np.digitize(col,sorted([-np.inf]+[point]+cut_remain+[np.inf]),right=False)

        if sum(~np.isin(np.arange(start=1,stop=cut_g.max()+1),np.unique(cut_g)))==0:

            cut_remain.append(point)

    cut_remain=np.array(cut_remain)
    
    return cut_remain



def binFreq(X,y,bin_num_limit=10,special_values=None,ws=None,coerce_monotonic=False):

    """
    等频分箱
    Parameters:
    --
        X
        y
        bin_num_limit:
        special_values
    """
  
    def get_breaks(col,y,bin_num_limit=bin_num_limit,ws=ws,special_values=special_values,coerce_monotonic=coerce_monotonic):       
        
        col=Specials()._sp_replace_single(col,Specials()._check_spvalues(col.name,special_values),fill_num=np.nan,fill_str='special')

        if col.isnull().all():
            
            warn('nan column:{},return blank breaks'.format(col.name))
            
            breaks=[]     
  
            
        elif is_numeric_dtype(col):
            
            if np.max(col) == np.min(col):
                
                warn('constant column:{},return blank breaks'.format(col.name))
                
                breaks=[]   
                
            else:
            
                y=y[~np.isnan(col)]    
    
                if is_array_like(ws) and coerce_monotonic:                
                    
                    ws=ws[~np.isnan(col)]
                    
                    if ws.size!=y.size:
                    
                        raise ValueError('length of weight not equal to y')
                        
                else:
                    
                    ws=np.ones(y.size)
                
                col=col[~np.isnan(col)]
                
                if np.unique(col).size<bin_num_limit:
                    
                    n_bins_adj=np.unique(col).size
                
                else:
                    
                    n_bins_adj=bin_num_limit
                    
                breaks=np.percentile(col,np.arange(n_bins_adj+1)/n_bins_adj*100)[1:-1]
                
                #adjust bin for extreamly unbalanced count distr
                if ws[col<np.min(breaks)].sum()/ws.sum()<=min(1/n_bins_adj,0.01):
                    
                    breaks=breaks[breaks!=np.min(breaks)]                
                
                breaks=np.unique(breaks).tolist()    
                
                if coerce_monotonic:
                    
                    breaks=check_monotonirc(col,y,breaks,ws=ws).tolist()
               
        elif is_string_dtype(col):
            
            breaks=col.unique().tolist()
            
        else:
            
            raise ValueError('dtype in only number and object')
            
        return breaks

    breaks_list={name_value[0]:get_breaks(name_value[1],y) 
                 for name_value in X.iteritems()}
    
    breaks_list=Base()._check_breaks(breaks_list)
    
    bins={col:varReportSinge().report(X[col],y,breaks_list[col],ws,special_values) for col in X.columns}
   
    return breaks_list,bins


class binKmeans(Base,Specials):
    
    """ 
    基于Kmeans的分箱调整算法:一种自动非优化分箱算法        
    一般通过细分箱后各个分箱的BadRate近似时需要合并,本算法可自动实现这一过程。注意‘missing’值与'special'值将不进行合并
    
    Params:
    ------
    
        breaks_list:分箱字典结构,{var_name:[bin],...},支持scorecardpy与toad的breaks_list结构 
        combine_ratio,float,分箱合并阈值,在bin_limit=1的情况下,阈值越大合并的箱数越多
            + 数值特征建议范围0.01-0.1
            + 分类特征中若类别水平非常多则建议加大combine_ratio,建议范围0.5
        bin_limit:int,分箱合并阈值,最终分箱的最低分箱数限制,越低则合并的箱数越多,建议4-6,同时若特征初始分箱小于bin_limit则不执行合并算法
        seed:int,kmeans随机种子,
            + 本算法在合并差距较大的barprob箱时,kmeans的随机性会被放大导致合并结果不可复现,设定seed值以复现合并结果
            + 设定合理的combine_ratio与bin_limit可极大的降低kmeans的随机性            
        sample_weight:array,样本权重
        special_values,list,dict,特殊值指代值,若数据中某些值或某列某些值需特殊对待(这些值不是np.nan)时设定
            + None,无特殊值
            + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
            + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan 
        n_jobs,int,并行数量,默认-1,在数据量较大、列较多的前提下可极大提升效率但会增加内存占用
        verbose,int,并行信息输出等级        
        
    Attributes:
    -------
        breaks_list:dict,产生的分箱dict
        bins:dict,当前breaks_list下的特征分析报告
    
    
    """    
    
    def __init__(self,breaks_list,combine_ratio=0.1,bin_limit=5,seed=123,sample_weight=None,special_values=None,n_jobs=-1,verbose=0):

        self.combine_ratio = combine_ratio       
        self.bin_limit=bin_limit
        self.breaks_list=breaks_list
        self.special_values=special_values
        self.sample_weight=sample_weight
        self.seed=seed
        self.n_jobs=n_jobs
        self.verbose=verbose
    
    def fit(self, X, y):
        
        self._check_data(X, y)
        
        #breaks_list=self.get_Breaklist_sc(self.breaks_list,X,y)
        breaks_list=self.breaks_list
        
        n_jobs=effective_n_jobs(self.n_jobs)  
        
        parallel=Parallel(n_jobs=n_jobs,verbose=self.verbose)
        
        col_break=parallel(delayed(self._combine_badprob_kmeans)(X[col],y,
                                                                self.combine_ratio,
                                                                self.bin_limit,
                                                                self.breaks_list[col],
                                                                self.sample_weight,
                                                                self.special_values,
                                                                self.seed)
                           for col in list(breaks_list.keys()))     
        
        self.breaks_list=self._check_breaks({col:breaks for col,breaks,_ in col_break})
        
        self.bins={col:vtab for col,_,vtab in col_break}
                                    
        return self
    
    
    def transform(self, X,y=None):       
            
        return X
        
        
    def _combine_badprob_kmeans(self,col,y,combine_ratio,bin_limit,breaks,ws=None,special_values=None,random_state=123):    
     
         #global var_bin,res_km_s
            
         var_raw=self._sp_replace_single(col,self._check_spvalues(col.name,special_values),fill_num=np.nan,fill_str='special')
         
         if is_array_like(ws):                
        
             ws=ws
        
             if ws.size!=y.size:
        
                 raise ValueError('length of weight not equal to y')
                 
         else:
             
             ws=pd.Series(np.ones(col.size),index=y.index)
         
            
         if is_string_dtype(var_raw):
             
             #fillna
             var_cut_fillna=pd.Series(np.where(var_raw.isnull(),'missing',var_raw),
                                       index=var_raw.index,
                                       name='bin')
             
             #map raw code to new code
             var_code_raw=var_cut_fillna.unique().tolist()
                                        
             map_codes=raw_to_bin_sc(var_code_raw,breaks)
                  
             var_map=var_cut_fillna.map(map_codes)
         
             #initialize params
             combine_ratio_count=True
             n_clusters=len(breaks)
             iter_times=len(breaks)
             iters=0
             breaks=breaks
             
             #no merge if n_clusters<=bin_limit(user-defined)
             if n_clusters<bin_limit:
                 
                 vtab=varReportSinge().report(col,y,breaks=breaks,sample_weight=ws,special_values=special_values)
                 
                 return col.name,breaks,vtab
             
             else:
             
                 
                 while True:
                     
                     #get badprob                             
                     gp=pd.concat([var_map,y.mul(ws).rename(y.name),ws.rename('ws')],axis=1).groupby(var_map)
        
                     var_bin=gp[y.name].sum().div(gp['ws'].sum()).rename('badprob').reset_index()
         
                     var_bin['bin']=var_bin['bin'].astype('str')
         
                     #unique values for n_cluster adjustion 
                     n_clusters_unique=var_bin['badprob'].unique().size
         
                     #combine_ratio_count>0: lower n_clusters
                     if combine_ratio_count:
        
                         #n_clusters should not greater than unique samples in data
                         if n_clusters_unique<n_clusters:
         
                             n_clusters=n_clusters_unique-1
         
                         else:
                             
                             n_clusters=n_clusters-1
         
                     #update combine_ratio when combine_ratio_count=0 to make futher merge
                     else:
                         
                         if combine_ratio==1:
                             
                             combine_ratio=combine_ratio
                         
                         else:
                             
                             combine_ratio=combine_ratio+0.02
         
                     #n_clusters not 0
                     if n_clusters<=1:
                         
                         n_clusters=1
         
         
                     n_clusters=var_bin['badprob'].unique().size-1                
                     
                     res_km=KMeans(n_clusters=n_clusters,random_state=random_state).fit_predict(var_bin[['badprob']])
                     res_km_s=pd.Series(res_km,var_bin.index,name='cluster')
         
                     #update string breaks
                     breaks=var_bin.groupby(res_km_s)['bin'].apply(lambda x : '%,%'.join(x)).tolist() 
         
                     #combine_ratio calculation
                     var_bin_ratio=var_bin['badprob'].diff(1).abs().div(var_bin.badprob+1e-10)
                     combine_ratio_count=var_bin_ratio.lt(combine_ratio).sum()                    
         
                     #map old string codes to new 
                     var_code_raw=var_map.unique().tolist()
                     mapcode=raw_to_bin_sc(var_code_raw,breaks)  
                     var_map=var_map.map(mapcode)
        
                     iters=iters+1
         
                     #stop condition 1/2: n_clusters<=bin_limit
                     if len(breaks)<=bin_limit:
                         
                         break  
                     #stop condition 2/2:iters gt max iters
                     if iters>=iter_times:
                         
                         break
                         
                 vtab=varReportSinge().report(col,y,breaks=breaks,sample_weight=ws,special_values=special_values)
                 return col.name,breaks,vtab
     
         elif is_numeric_dtype(var_raw):
             
             #initialize params
             combine_ratio_count=True
             n_clusters=len(breaks)+1
             iter_times=col.unique().size
             iters=0
             breaks=breaks
             
             #no merge if n_clusters<=bin_limit(user-defined)
             if n_clusters<bin_limit:
                 
                 vtab=varReportSinge().report(col,y,breaks=breaks,sample_weight=ws,special_values=special_values)
                 
                 return col.name,breaks,vtab
             
             else:
             
                 while True:
        
                     var_cut=pd.cut(var_raw,[-np.inf]+breaks+[np.inf],duplicates='drop',right=False)   
             
                     #get badprob
                     var_cut_fillna=pd.Series(np.where(var_cut.isnull(),'missing',var_cut),
                                       index=var_cut.index,
                                       name='bin') 
         
                     #get badprob
                     gp=pd.concat([var_cut_fillna,y.mul(ws).rename(y.name),ws.rename('ws')],axis=1).groupby(var_cut_fillna)
        
                     var_bin=gp[y.name].sum().div(gp['ws'].sum()).rename('badprob').reset_index()
                     
                     #numeric missings excluded
                     var_bin=var_bin[var_bin['bin']!='missing'] 
                     var_bin['bin']=var_bin['bin'].astype('str')                      
         
                     #unique values for n_cluster adjustion  
                     n_clusters_unique=var_bin['badprob'].unique().size
         
                     #combine_ratio_count>0: lower n_clusters
                     if combine_ratio_count:
                         
                         #n_clusters should not greater than unique samples in data
                         if n_clusters_unique<n_clusters:
         
                             n_clusters=n_clusters_unique-1
         
                         else:
                             
                             n_clusters=n_clusters-1
          
                     #stop condition 1/3:combine_ratio_count=0 then break
                     else:
                     
                         if combine_ratio==1:
                             
                             combine_ratio=combine_ratio
                         
                         else:
                             
                             combine_ratio=combine_ratio+0.02
                     
         
                     #n_clusters not 0
                     if n_clusters<=1:
                         
                         n_clusters=1
         
                     res_km=KMeans(n_clusters=n_clusters,random_state=random_state).fit_predict(var_bin[['badprob']])
                     res_km_s=pd.Series(res_km,var_bin.index,name='cluster')
        
                     #get index of bins to be merged
                     g_index_list=var_bin.groupby(res_km_s)['bin'].apply(lambda x : x.index.sort_values().tolist()).tolist()
         
                     #combine_ratio_count calculation
                     var_bin_ratio=var_bin['badprob'].diff(1).abs().div(var_bin.badprob+1e-10)
                     combine_ratio_count=var_bin_ratio.lt(combine_ratio).sum()
         
                     #remove points from orginal breaks
                     index_drop=self._getindex(g_index_list)
        
                     breaks=np.delete(breaks,index_drop).tolist()
                     iters=iters+1
                     
                     #stop condition 1/2:bin_num<=bin_limit(user-defined)
                     if len(breaks)+1<=bin_limit:
                         
                         #print('len(breaks)<=bin_limit')
        
                         break
                     
                     #stop condition 2/2:iters gt max iters
                     if iters>=iter_times:
                         
                         #print('iter_times')
                         
                         break
                     
                 vtab=varReportSinge().report(col,y,breaks=breaks,sample_weight=ws,special_values=special_values)
                 
                 return col.name,breaks,vtab
             
         #列类型为其他特殊情况    
         else:
             
             raise ValueError("col's dtype in (number,object).")


    def _getindex(self,g_index_list):
        
        """ 
        寻找list中值连续索引的位置并标记
        """ 
        
        ll=[]
        for lst in g_index_list:
            fun = lambda x: x[1]-x[0]
            for k, g in groupby(enumerate(lst), fun):
                l1 = [j for i, j in g] 
                if len(l1) > 1:
                    ll.append(min(l1)) 
        return ll

    
class binTree(Base,Specials):
    
    """ 
    决策树递归最优分箱
    分类特征处理方式:按照badrate对类进行排序并进行ordinal编码再进行分箱(与scorecardpy一致)
    分类中不要出现字符空('' or "")类
    Params:
    ------
    max_bin=50,初始分箱数
        + 使用Pretty Breakpoints获取预分箱点,详见R的pretty函数
        + 移除异常值,移除边界点
        + 越多的初始分箱数能够得到越好的最优分箱点，但会增加计算量。max_bin=50时与scorecardpy一致
    criteria='iv',决策树进行分割的指标，
        + 目前支持iv_gain与ks_gain,gain表示分割前后的指标增益,参数tol可控制分割停止的增益限制
        + 当适用样本权重ws时，算法会改为计算加权的iv_gain与ks_gain
    max_iters=100,决策树递归次数
    tol=1e-4,决策树进行分割的指标的增益小于tol时停止分割
        + tol越小分割越容易,越大分割越难
    distr_limit=0.05,每一箱的样本占比限制
    bin_num_limit=8,分箱总数限制
    coerce_monotonic=False,是否强制bad_prob单调，默认否
        + 为True时,本算法中会在最优分割过程中加入单调性限制,强制每一个新的分割点都必须先使bad_rate单调
        + 若x与y本身有单调趋势则强制单调能够取得理想的结果,若x与y的关系是非线性关系则强制单调结果会不理想
    ws=None,None or pandas.core.series.Series,样本权重
        + 样本权重影响分割的指标iv_gain与ks_gain的计算结果，进而影响最优分割点的选择
            + 若仅对好坏样本加权(权重非0)则iv_gain与ks_gain无变化，也不会影响最优分割点的选择
        + 若coerce_monotonic=True,样本权重影响bad_rate的计算进而会影响单调最优分箱的分割点的选择
            + 若仅对好坏样本加权(权重非0)则bad_rate的计算的排序性不变，也不会影响最优分割点的选择        
    special_values:特殊值指代值,若数据中某些值或某列某些值需特殊对待(这些值不是np.nan)时设定
        + None,保证数据默认
        + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
        + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被算法认定为'special'
    n_jobs=-1,int,并行数量,默认-1,在数据量较大、列较多的前提下可极大提升效率但会增加内存占用
    verbose=0,并行信息输出等级    
        
    Attributes:
    -------
    breaks_list:dict,产生的分箱dict
    bins:dict,当前breaks_list下的特征分析报告
    """    
    
    def __init__(self,max_bin=50,criteria='iv',max_iters=100,
                 tol=1e-4,distr_limit=0.05,bin_num_limit=8,coerce_monotonic=False,
                 ws=None,special_values=None,n_jobs=-1,verbose=0):

        self.max_bin=max_bin
        self.criteria=criteria
        self.max_iters=max_iters
        self.tol=tol
        self.distr_limit=distr_limit
        self.bin_num_limit=bin_num_limit
        self.ws=ws
        self.coerce_monotonic=coerce_monotonic
        self.special_values=special_values
        self.n_jobs=n_jobs
        self.verbose=verbose
        
    
    def fit(self, X, y):
        
        self._check_data(X, y)
        
        n_jobs=effective_n_jobs(self.n_jobs)  
        
        p=Parallel(n_jobs=n_jobs,verbose=self.verbose)
        
        res=p(delayed(self._get_treecut)(col[1],y,self.max_bin,
                                        self.criteria,self.max_iters,
                                        self.tol,self.distr_limit,
                                        self.bin_num_limit,
                                        self.ws,
                                        self.coerce_monotonic,
                                        self.special_values) for col in X.iteritems())
        
        self.breaks_list=self._check_breaks({col_name:breaks for col_name,breaks,_ in res})        
        self.bins={col_name:vtab for col_name,_,vtab in res}
                                    
        return self
    
    
    def transform(self, X,y=None):       
        
        return X

        
    def _get_treecut(self,col,y,max_bin,criteria,max_iters,tol,distr_limit,bin_num_limit,ws,coerce_monotonic,special_values):
        
        col_raw=col.copy()
        
        col=self._sp_replace_single(col,self._check_spvalues(col.name,special_values),fill_num=np.nan,fill_str='special')
        
        #sample_wieght
        if is_array_like(ws):
            
            ws=ws.values
        
        else:
            
            ws=None
        
        
        #numeric column
        if is_numeric_dtype(col):
                      
            
            #no cut applied when col's unique value pop too high               
            if col.value_counts(dropna=False).div(col.size).max()>0.95:
                
                breaks=[]       
                
                vtab=varReportSinge().report(col_raw,y,breaks,sample_weight=ws,special_values=special_values)
                
            elif np.unique(col[~np.isnan(col)]).size==1:
                
                breaks=[]
                
                vtab=varReportSinge().report(col_raw,y,breaks,sample_weight=ws,special_values=special_values)
            
            #tree cut
            else:
                
                breaks=self._get_bestsplit(col.values,y.values,max_bin=max_bin,
                                     criteria=criteria,
                                     max_iters=100,
                                     tol=0.0001,
                                     ws=ws,
                                     distr_limit=distr_limit,
                                     is_str_dtype=False,
                                     coerce_monotonic=coerce_monotonic,
                                     bin_num_limit=bin_num_limit)
                
                vtab=varReportSinge().report(col_raw,y,breaks,sample_weight=ws,special_values=special_values)
        
        #string columns         
        elif is_string_dtype(col):
            
            
            if np.unique(col).size==1:
            
                breaks=[]
                
            else:
                #sort levels by bad_rate(no-wieght)
                codes=y.groupby(col).mean().sort_values().index.tolist()
                
                #ordinal encode data start with 0
                map_code=dict(zip(codes,range(len(codes))))
    
                #tree cut
                breaks_raw=self._get_bestsplit(col.map(map_code).values,
                                         y.values,
                                         criteria=criteria,
                                         max_iters=100,
                                         tol=1e-4,
                                         ws=ws,
                                         distr_limit=distr_limit,
                                         is_str_dtype=True,
                                         bin_num_limit=bin_num_limit)
                
                #restore string breaks
                breaks=['%,%'.join(i) for i in np.split(codes,np.int32(breaks_raw)) if i.tolist()]    
            
            vtab=varReportSinge().report(col_raw,y,breaks,sample_weight=ws,special_values=special_values)
            
        else:
            
            raise ValueError("col's dtype in ('number','object')")
  

        return col.name,breaks,vtab


    def _get_bestsplit(self,col,y,max_bin=50,ws=None,criteria='iv',tol=1e-4,
                      max_iters=100,distr_limit=0.05,bin_num_limit=8,
                      is_str_dtype=False,coerce_monotonic=False): 
        
        #get sample_weight
        if is_array_like(ws):
            
            if ws.size!=y.size:

                raise ValueError('length of weight not equal to y')
            
            y = y * ws
            
            count=ws
            
        else:
            
            count=np.ones(col.size)    
        
        
        nan_sum=pd.isnull(col).any()
            
        #string dtype variable
        if is_str_dtype:
            
            cuts_remain=np.unique(col)            
        
        #number dtype variable
        else:
                      
            #remove outliers
            col_rm_outlier=remove_outlier(col)
            
            #adjust max_bin for improving performance
            if pd.unique(col_rm_outlier).size<max_bin:

                max_bin_adj=pd.unique(col_rm_outlier).size

            else:

                max_bin_adj=max_bin  

            #R pretty bins:cut points looks better but will lose iv or ks gain
            cuts_remain=R_pretty(np.nanmin(col_rm_outlier),np.nanmax(col_rm_outlier),max_bin_adj) 

            #equal freqs
            #cuts_remain = np.unique(np.nanpercentile(col_rm_outlier,np.linspace(0, 1, max_bin + 1)[1:-1] * 100, interpolation='lower'))

            #equal width 
            #_,cuts_remain = np.histogram(col_rm_outlier,max_bin,weights=ws)#histogram bins                
       
    
        cuts_tree=[];inds=[];best_criteria=[0]
        
        iters=0
    
        # tree cut
        while True:
            
            iters=iters+1
            
            inds=[];c_dis_d=[]
            
            for point in cuts_remain.copy():            
                
                interval=sorted(np.unique([-np.inf]+[point]+cuts_tree+[np.inf]))
                         
                col_group=np.digitize(col,interval,right=False)
          
                #if point that could not split data(edge point),then pass the iteration
                if sum(~np.isin(np.arange(start=1,stop=col_group.max()+1),np.unique(col_group)))>0:
                    
                    cuts_remain=cuts_remain[cuts_remain!=point]
            
                else:
    
                    #split data and count the good/bad
                                
                    g=np.zeros(0)
                    b=np.zeros(0)
                    count_g=np.zeros(0)
                    
                    for i in np.unique(col_group):
                        
                        y_g=y[col_group==i]
                        unit_g=count[col_group==i]
    
                        count_g=np.append(count_g,np.sum(unit_g))
    
                        b=np.append(b,np.sum(y_g))
                        g=np.append(g,np.sum(unit_g) - np.sum(y_g))   
    
                    
                    bad_prob=b/count_g    
                    # count_g=pd.Series(count).groupby(col_group).sum().ravel()
                    # b=pd.Series(y).groupby(col_group).sum().ravel()
                    # g=count_g-b
    
                    # interim=pd.DataFrame({'count':count,'y':y}).groupby(col_group).sum()
                    # count_g=interim['count'].ravel()
                    # b=interim['y'].ravel()
                    # g=count_g-b
        
                    #if no good or bad sample in cut of the point,then pass the iteration 
                    if 0 in b.tolist() or 0 in g.tolist():
                        
                        cuts_remain=cuts_remain[cuts_remain!=point]    
                    
                    #if cuts_tree with current point shows no monotonic trend,then pass the iteration(no nans in col) 
                    elif coerce_monotonic and (not is_str_dtype) and (not is_monotonic(bad_prob)) and (not nan_sum):                    

                        cuts_remain=cuts_remain[cuts_remain!=point]
                
                    #if cuts_tree with current point shows no monotonic trend,then pass the iteration(nans in col)  
                    elif coerce_monotonic and (not is_str_dtype) and (not is_monotonic(bad_prob[:-1])) and (nan_sum):                 

                        cuts_remain=cuts_remain[cuts_remain!=point]
                    
                    #else get criteria calculated
                    else:
    
                        g_dis=g/g.sum()
                        b_dis=b/b.sum()
                        g_dis_cs=np.cumsum(g_dis)
                        b_dis_cs=np.cumsum(b_dis)   
    
                        # nan distr will not be calculated
                        if nan_sum>0:
                            
                            c_dis=(count_g/count_g.sum())[:-1]
    
                        else:
    
                            c_dis=count_g/count_g.sum()
    
                        c_dis_d.append(c_dis)
    
    
                        # iv&ks supported,more criterias will be added in future
                        if criteria=='iv':
    
                            ind=((b_dis-g_dis)*np.log((b_dis+1e-10)/(g_dis+1e-10))).sum()
                            #print(cuts_tree,ind)
    
                        elif criteria=='ks':
    
                            ind=np.max(np.abs(g_dis_cs-b_dis_cs))
    
                        else:
    
                            raise ValueError("criteria in 'iv' or 'ks'")
    
                        inds.append(ind)
                        
            # stop condition(or): 
            #.   len(cuts_remain)==0,
            #.   beyond max_iters,
            #.   criteria_gain stop growing,
            #.   count_distr below limit
            #.   bin num beyond limit
            
            if not cuts_remain.size:
                
                #print("len(cuts_remain)==0")
                
                break
                        
            else:
                #get best point and update params 
                best_point=cuts_remain[np.argmax(inds)] #best split point
    
                best_point_dist_min=c_dis_d[np.argmax(inds)].min() 
    
                best_criteria.append(np.max(inds)) #socre at best split point
    
                cuts_remain=cuts_remain[cuts_remain!=best_point] # pop best_point out of orgin cut         
    
                #calculate best_criteria_gain
                best_criteria_gain=(best_criteria[-1]-best_criteria[-2])/best_criteria[-2] if best_criteria[-2] else 1
    
                cuts_tree.append(best_point)
    
                #remove cut point with lower count_distr
                if best_point_dist_min<distr_limit:
    
                    cuts_tree.remove(best_point)
    
                    best_criteria=best_criteria[:-1]
    
                bin_num=len(cuts_tree)+1 #bin num 
            
                if iters>max_iters: 
    
                    #print("max_iters reach")
    
                    break
    
                if best_criteria_gain<tol:
    
                    #print("best_criteria_gain reach")
    
                    break
    
                if bin_num>=bin_num_limit:
    
                    #print("bin_num_limit reach")
    
                    break
            
        return (sorted(cuts_tree))

            
class binChi2(Base,Specials):            
            
    """ 
    卡方自动分箱,合并卡方值较低的分箱并调整分箱样本量，分箱数至用户定义水平    
    [ChiMerge:Discretization of numeric attributs](http://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf)
    分类特征处理方式:按照badrate对类进行排序并进行ordinal编码再进行卡方分箱(与scorecardpy一致)
    分类中不要出现字符空('' or "")类
    Params:
    ------
    max_bin=50,初始分箱数，
        + 使用Pretty Breakpoints获取预分箱点,详见R的pretty函数
        + 移除异常值,移除边界点
        + 越多的初始分箱数越有可能得到越好的卡方分箱点，但会增加计算量。max_bin=50时与scorecardpy一致
    tol=0.1,卡方分箱合并分箱的卡方显著性阈值，tol=0.1下自由度为1的卡方分布的阈值约为2.70554345409542
         + 越高的tol合并的箱越少,反之则则合并的箱越多
         + 理论上推荐的阈值为0.1，0.05，0.01,实践中可依据需要进行设定         
    distr_limit=0.05,每一箱的样本占比限制
    bin_num_limit=8,分箱总数限制，实际分箱数小于等于bin_num_limit
    coerce_monotonic=False,是否强制bad_prob单调，默认否
        + 本算法中会先在预分箱中强制单调，再进行卡方分箱以保证卡方分箱单调
        + 若x与y本身有单调趋势则强制单调能够取得理想的结果,若x与y的关系是非线性关系则强制单调结果会不理想
    ws=None,None or pandas.core.series.Series,样本权重
    special_values:特殊值指代值,若数据中某些值或某列某些值需特殊对待(这些值不是np.nan)时设定
        + None,保证数据默认
        + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
        + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan  
    n_jobs=-1,int,并行数量,默认-1,在数据量较大、列较多的前提下可极大提升效率但会增加内存占用
    verbose=0,并行信息输出等级    
        
    Attributes:
    -------
    breaks_list:dict,产生的分箱dict
    bins:dict,当前breaks_list下的特征分析报告
    """    
    
    def __init__(self,max_bin=50,tol=0.1,distr_limit=0.05,bin_num_limit=8,
                 coerce_monotonic=False,ws=None,special_values=None,n_jobs=-1,verbose=0):

        self.max_bin=max_bin
        self.tol=tol
        self.distr_limit=distr_limit
        self.bin_num_limit=bin_num_limit
        self.ws=ws
        self.coerce_monotonic=coerce_monotonic
        self.special_values=special_values
        self.n_jobs=n_jobs
        self.verbose=verbose
        
    
    def fit(self, X, y):
        
        self._check_data(X, y)
        
        n_jobs=effective_n_jobs(self.n_jobs)  
            
        p=Parallel(n_jobs=n_jobs,verbose=self.verbose)
        
        res=p(delayed(self._get_chi2merge)(col[1],y,
                                        self.max_bin,
                                        self.tol,
                                        self.distr_limit,
                                        self.bin_num_limit,
                                        self.ws,
                                        self.coerce_monotonic,
                                        self.special_values) for col in X.iteritems())
        
        self.breaks_list=self._check_breaks({col_name:breaks for col_name,breaks,_ in res})
        self.bins={col_name:vtab for col_name,_,vtab in res}            
                                
        return self   
    

    def transform(self, X):       
        
        return X
           
    
    def _get_chi2merge(self,col,y,max_bin=50,tol=0.1,distr_limit=0.05,bin_num_limit=8,ws=None,coerce_monotonic=False,special_values=None):
    
        col_raw=col.copy()
        
        col=self._sp_replace_single(col,self._check_spvalues(col.name,special_values),fill_num=np.nan,fill_str='special')        
    
        #sample_wieght
        if is_array_like(ws):
    
            ws=ws.values
            
            if ws.size!=y.size:
                
                raise ValueError('length of weight not equal to y')
    
        else:
    
            ws=np.ones(y.size)
            
        #num columns
        if is_numeric_dtype(col):
            
            #print(col.name)
            
            #no merge applied when only one unique value existed in col
            if np.isnan(col).all():
   
                 breaks=[]
            
            elif np.unique(col[~np.isnan(col)]).size==1:
                
                breaks=[]
                
            else:
                #chi2_merge
                breaks=self._chi2_merge(col.values,y.values,ws=ws,max_bin=max_bin,
                              distr_limit=distr_limit,stop_limit=tol,
                              bin_num_limit=bin_num_limit,is_str_dtype=False,
                              coerce_monotonic=coerce_monotonic)
            
            #get vtab using chi2-breaks
            vtab=varReportSinge().report(col_raw,y,breaks,sample_weight=ws,special_values=special_values)
            
        elif is_string_dtype(col):
            
            #print(col.name)
            
            #no merge applied when only one unique value existed in col
            if np.unique(col).size==1:
                
                breaks=[]
                
            else:
                #sort levels by bad_rate(no-wieght)
                codes=y.groupby(col).mean().sort_values().index.tolist()
    
                #ordinal encode data start with 0
                map_code=dict(zip(codes,list(range(len(codes)))))
    
                #chi2_merge
                breaks_raw=self._chi2_merge(col.map(map_code).values,y.values,ws=ws,
                                      distr_limit=distr_limit,stop_limit=tol,
                                      bin_num_limit=bin_num_limit,
                                      is_str_dtype=True)
    
                #restore string breaks
                breaks=['%,%'.join(i) for i in np.split(codes,np.int32(breaks_raw)) if i.tolist()] 
            
            #get vtab using chi2-breaks
            vtab=varReportSinge().report(col_raw,y,breaks,sample_weight=ws,special_values=special_values)
    
        else:
                
            raise ValueError("col's dtype in ('number','object')")    
            
            
        return col.name,breaks,vtab     


    def _chi2_merge(self,col,y,ws,max_bin=50,distr_limit=0.05,bin_num_limit=8,stop_limit=0.1,
                   is_str_dtype=False,coerce_monotonic=False):   
    
        if max_bin<2:
            
            raise ValueError('max_bin should greater than 1')
            
        #get count limit per bin
            
        count_limit=distr_limit*ws.sum()
        
        #get chi2 threshold at stop_limit
        threshold = chi2.isf(stop_limit, df=1)
       
        #drop nans  
        if is_str_dtype:
                
            cuts=np.sort(np.unique(col))[1:]
            
        else:
            #get initial-binning        
            y=y[~np.isnan(col)]
   
            ws=ws[~np.isnan(col)]
            
            if ws.size!=y.size:
            
                raise ValueError('length of weight not equal to y')
                    
            col=col[~np.isnan(col)]
        
            #drop outtliers
            col_rm_outtlier=remove_outlier(col)
    
            #adjust max_bin
            if np.unique(col_rm_outtlier).size<max_bin:
    
                max_bin_adj=np.unique(col_rm_outtlier).size
    
            else:
    
                max_bin_adj=max_bin    
         
            #get pretty cuts
            cuts=R_pretty(np.nanmin(col_rm_outtlier),np.nanmax(col_rm_outtlier),max_bin_adj)
    
            #remove edge point in initial-binning
            cuts=rm_edge_point(col,cuts)    
            
            if coerce_monotonic:
                
                #remove points to make monotonic bad_prob if coerce_monotonic==True
                cuts=check_monotonirc(col,y,cuts,ws=ws)
        
    
        #calculate chi2 value using initial-binning    
        _,chi2_d,count_list,cuts_bin=self._chi2_bin(col,y,cuts,ws)
        
        cuts=np.array([i[0] if i[0]!= -np.inf else i[1] for i in cuts_bin])
      
        
        #pop points out of initial-binning
        while True:    
    
            # #for debug:check length of idx,cuts,count,chi2 
            # if len(idx) and len(cuts) and len(count_list) and len(chi2_d) and not ((len(idx)+1)==len(cuts)==len(count_list)==len(chi2_d)):
                
            #     print("len(idx)=={}".format(str(len(idx))))
            #     print("len(cuts)=={}".format(str(len(cuts))))
            #     print("len(count_list)=={}".format(str(len(count_list))))
            #     print("len(chi2_d)=={}".format(str(len(chi2_d))))
                
            #     raise ValueError('not (len(idx)==len(cuts)==len(count_list)==len(chi2_d))')
            
            if len(cuts)==0 or len(count_list)==0 or len(chi2_d)==0:
    
                cuts=np.array([])
                
                break
            
            #remove cut point with lowest chi2 value
            elif chi2_d.min()<threshold:
                
                #if string col‘s cuts can not make all bins' chi2 lower than threshold,then merge all bins
                if is_str_dtype and np.unique(cuts).size==1:
                    
                    cuts=np.array([])
                 
                else:
                    #print('point {} out due to chi2'.format(str(cuts[np.argmin(chi2_d)])))
                    
                    cuts=cuts[cuts!=cuts[np.argmin(chi2_d)]]
    
                    _,chi2_d,count_list,cuts_bin=self._chi2_bin(col,y,cuts,ws)
                    
                    cuts=np.array([i[0] if i[0]!= -np.inf else i[1] for i in cuts_bin])
     
            #remove cut point with bin count less than user-defined   
            elif count_list.min()<count_limit:
            
                #if string col‘s cuts can not make all bins' distr lower than count_limit,then merge all bins
                if is_str_dtype and np.unique(cuts).size==1:
                    
                    cuts=np.array([])
                    
                else:
                    
                    #print('point {} out due to count_limit'.format(str(cuts[np.argmin(count_list)])))                               
                    cuts=cuts[cuts!=cuts[np.argmin(count_list)]]
                    _,chi2_d,count_list,cuts_bin=self._chi2_bin(col,y,cuts,ws)               
                    cuts=np.array([i[0] if i[0]!= -np.inf else i[1] for i in cuts_bin])
    
            #remove cut point with lowest chi2 value when bin_num higher than user-defined   
            elif len(cuts)>bin_num_limit:
                 
                #print('point {} out due to bin_num_limit'.format(str(cuts[np.argmin(chi2_d)])))
    
                cuts=cuts[cuts!=cuts[np.argmin(chi2_d)]]
    
                _,chi2_d,count_list,cuts_bin=self._chi2_bin(col,y,cuts,ws)   
    
                cuts=np.array([i[0] if i[0]!= -np.inf else i[1] for i in cuts_bin])
                
            #else break the loop 
            else:
    
                break
                
        return np.unique(cuts).tolist()
    

    def _chi2_bin(self,col,y,cut_off,ws=None):        
    
        if not np.equal(np.unique(y),np.array([0.,1.])).all():
            
            raise ValueError('y values only in (0.,1.)')
        
        # update cut_group    
        cut_off=np.sort(np.unique(cut_off))
            
        cut_off_all=[-np.inf]+cut_off.tolist()+[np.inf]
    
        cut_g=np.digitize(col,cut_off_all,right=False)
        
        cut_bin=np.array([cut_off_all[i:i + 2] for i in range(len(cut_off_all) - 1)])
    
        #sample weights
        if is_array_like(ws):
            
            ws=ws
            
            if ws.size!=y.size:
                
                raise ValueError('length of weight not equal to y')
                
        else:
            
            ws=np.ones(y.size)
        
        gid=np.unique(cut_g)
        chi2_d=[1e3]
        count_list=[]
        idx=[]
    
        for i in range(len(gid)-1):
            
            idx.append(i)
            
            y_g_1=y[cut_g==gid[i]]
            ws_g_1=ws[cut_g==gid[i]]        
    
            y_g_2=y[cut_g==gid[i+1]]
            ws_g_2=ws[cut_g==gid[i+1]]
            
            
            # if all vals in y groupby col equal to 0 or 1 then chi2==0
            if (np.append(y_g_1,y_g_2)==0).all() or (np.append(y_g_1,y_g_2)==1).all():
                
                xtab=np.array([[ws_g_1[y_g_1==0].sum(),ws_g_2[y_g_2==0].sum()],
                               [ws_g_1[y_g_1==1].sum(),ws_g_2[y_g_2==1].sum()]])
                
                #merge 1st point in bin when distr too low
                if i==0:
                    
                    count_list.append(xtab[:,0].sum())
                    
                count_list.append(xtab[:,1].sum())
                    
                #all vals in y groupby col equal to 0 or 1 then chi2==0
                chi2_d.append(0.) 
                
            else:
    
                #y-values only in (0,1)
                xtab=np.array([[ws_g_1[y_g_1==0].sum(),ws_g_2[y_g_2==0].sum()],
                               [ws_g_1[y_g_1==1].sum(),ws_g_2[y_g_2==1].sum()]])
                
                #merge 1st point in bin when distr too low
                if i==0:
                    
                    count_list.append(xtab[:,0].sum())
                
                count_list.append(xtab[:,1].sum())
            
                #calculate chi2 using scipy
                ccsq,_,_,_,=chi2_contingency(xtab,correction=False) 
    
                chi2_d.append(ccsq) 
        
        return np.array(idx),np.array(chi2_d),np.array(count_list),np.array(cut_bin)


class binPretty(Base,Specials):            
            
    """ 
    pretty分箱,使用pretty cuts作为预分箱再调整分箱至用户定义水平    
    分类特征处理方式:按照badrate对类进行排序并进行ordinal编码再进行分箱
    分类中不要出现字符空('' or "")类
    Params:
    ------
    max_bin=50,初始分箱数，
        + 使用Pretty Breakpoints获取预分箱点,详见R的pretty函数
        + 移除异常值,移除边界点
        + 越多的初始分箱数越有可能得到越好的卡方分箱点，但会增加计算量。max_bin=50时与scorecardpy一致    
    distr_limit=0.05,每一箱的样本占比限制
    bin_num_limit=8,分箱总数限制，实际分箱数小于等于bin_num_limit
    coerce_monotonic=False,是否强制bad_prob单调，默认否
        + 本算法中会先在预分箱中强制单调，再进行合并分箱以保证分箱单调
        + 若x与y本身有单调趋势则强制单调能够取得理想的结果,若x与y的关系是非线性关系则强制单调结果会不理想
    ws=None,None or pandas.core.series.Series,样本权重
    special_values:特殊值指代值,若数据中某些值或某列某些值需特殊对待(这些值不是np.nan)时设定
        + None,保证数据默认
        + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
        + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan  
    n_jobs=-1,int,并行数量,默认-1,在数据量较大、列较多的前提下可极大提升效率但会增加内存占用
    verbose=0,并行信息输出等级    
        
    Attributes:
    -------
    breaks_list:dict,产生的分箱dict
    bins:dict,当前breaks_list下的特征分析报告
    """    
    
    def __init__(self,max_bin=50,distr_limit=0.05,bin_num_limit=8,
                 coerce_monotonic=False,ws=None,special_values=None,n_jobs=-1,verbose=0):

        self.max_bin=max_bin
        self.distr_limit=distr_limit
        self.bin_num_limit=bin_num_limit
        self.ws=ws
        self.coerce_monotonic=coerce_monotonic
        self.special_values=special_values
        self.n_jobs=n_jobs
        self.verbose=verbose
        
    
    def fit(self, X, y):
        
        self._check_data(X,y)

        n_jobs=effective_n_jobs(self.n_jobs)  
        
        p=Parallel(n_jobs=n_jobs,verbose=self.verbose)
        
        res=p(delayed(self._get_prettymerge)(col[1],y,
                                        self.max_bin,
                                        self.distr_limit,
                                        self.bin_num_limit,
                                        self.ws,
                                        self.coerce_monotonic,
                                        self.special_values) for col in X.iteritems())
        
        self.breaks_list=self._check_breaks({col_name:breaks for col_name,breaks,_ in res})
        self.bins={col_name:vtab for col_name,_,vtab in res}            
                                    
        return self   
    

    def transform(self, X):       
        
        if X.size:
            
            return X
        
        else:
            
            warnings.warn('0 rows in input X,return None')
            
            return pd.DataFrame(None)            
    
    
    def _get_prettymerge(self,col,y,max_bin=50,distr_limit=0.05,bin_num_limit=8,ws=None,coerce_monotonic=False,special_values=None):
    
        col_raw=col.copy()
        
        col=self._sp_replace_single(col,self._check_spvalues(col.name,special_values),fill_num=np.nan,fill_str='special')                
    
        #sample_wieght
        if is_array_like(ws):
    
            ws=ws.values
            
            if ws.size!=y.size:
                
                raise ValueError('length of weight not equal to y')
    
        else:
    
            ws=np.ones(y.size)
            
        #num columns
        if is_numeric_dtype(col):
            
            #no merge applied when only one unique value existed in col
            if np.isnan(col).all():
   
                 breaks=[]
            
            elif np.unique(col[~np.isnan(col)]).size==1:
                
                breaks=[]
                
            else:
                #merge
                breaks=self._pretty_merge(col.values,y.values,ws=ws,max_bin=max_bin,
                              distr_limit=distr_limit,
                              bin_num_limit=bin_num_limit,is_str_dtype=False,
                              coerce_monotonic=coerce_monotonic)
            
            #get vtab using chi2-breaks
            vtab=varReportSinge().report(col_raw,y,breaks,sample_weight=ws,special_values=special_values)
            
        elif is_string_dtype(col):
            
            #no merge applied when only one unique value existed in col
            if np.unique(col).size==1:
                
                breaks=[]
                
            else:
                #sort levels by bad_rate(no-wieght)
                codes=y.groupby(col).mean().sort_values().index.tolist()
    
                #ordinal encode data start with 0
                map_code=dict(zip(codes,list(range(len(codes)))))
    
                #merge
                breaks_raw=self._pretty_merge(col.map(map_code).values,y.values,ws=ws,
                                      distr_limit=distr_limit,
                                      bin_num_limit=bin_num_limit,
                                      is_str_dtype=True)
    
                #restore string breaks
                breaks=['%,%'.join(i) for i in np.split(codes,np.int32(breaks_raw)) if i.tolist()] 
            
            #get vtab using chi2-breaks
            vtab=varReportSinge().report(col_raw,y,breaks,sample_weight=ws,special_values=special_values)
    
        else:
                
            raise ValueError("col's dtype in ('number','object')")    
            
            
        return col.name,breaks,vtab     


    def _pretty_merge(self,col,y,ws,max_bin=50,distr_limit=0.05,bin_num_limit=8,
                   is_str_dtype=False,coerce_monotonic=False):   
    
        if max_bin<2:
            
            raise ValueError('max_bin should greater than 1')
            
        #get count limit per bin
        count_limit=distr_limit*ws.sum()
       
        #drop nans  
        if is_str_dtype:
                
            cuts=np.sort(np.unique(col))[1:]
            
        else:
            #get initial-binning        
            y=y[~np.isnan(col)]                         
                
            ws=ws[~np.isnan(col)]
            
            if ws.size!=y.size:
            
                raise ValueError('length of weight not equal to y')
                    
            col=col[~np.isnan(col)]
        
            #drop outtliers
            col_rm_outtlier=remove_outlier(col)
    
            #adjust max_bin
            if np.unique(col_rm_outtlier).size<max_bin:
    
                max_bin_adj=np.unique(col_rm_outtlier).size
    
            else:
    
                max_bin_adj=max_bin    
         
            #get pretty cuts
            cuts=R_pretty(np.nanmin(col_rm_outtlier),np.nanmax(col_rm_outtlier),max_bin_adj)
    
            #remove edge point in initial-binning
            cuts=rm_edge_point(col,cuts)    
            
            if coerce_monotonic:
                
                #remove points to make monotonic bad_prob if coerce_monotonic==True
                cuts=check_monotonirc(col,y,cuts,ws=ws)
        
    
        #calculate chi2 value using initial-binning    
        count_list,cuts_bin=self._pretty_bin(col,y,cuts,ws)
        
        cuts=np.array([i[0] if i[0]!= -np.inf else i[1] for i in cuts_bin])
      
        
        #pop points out of initial-binning
        while True:    
            
            #print("len(cuts)=={},len(count_list)=={}".format(str(len(cuts)),str(len(count_list))))

    
            #bin_num==1 then stop 
    
            if len(cuts)==0 or len(count_list)==0:
    
                cuts=np.array([])
                
                break
     
            #remove cut point with bin count less than user-defined   
            elif count_list.min()<count_limit:
            
                #if string col‘s cuts can not make all bins' distr lower than count_limit,then merge all bins
                if is_str_dtype and np.unique(cuts).size==1:
                    
                    cuts=np.array([])
                    
                else:
                    
                    #print('point {} out due to count_limit'.format(str(cuts[np.argmin(count_list)])))                               
                    cuts=cuts[cuts!=cuts[np.argmin(count_list)]]
                    
                    count_list,cuts_bin=self._pretty_bin(col,y,cuts,ws)    
                    
                    cuts=np.array([i[0] if i[0]!= -np.inf else i[1] for i in cuts_bin])
    
            #remove cut point with lowest bin count when bin_num higher than user-defined   
            elif len(cuts)>bin_num_limit:
                 
                #print('point {} out due to bin_num_limit'.format(str(cuts[np.argmin(chi2_d)])))
    
                cuts=cuts[cuts!=cuts[np.argmin(count_list)]]
    
                count_list,cuts_bin=self._pretty_bin(col,y,cuts,ws)   
    
                cuts=np.array([i[0] if i[0]!= -np.inf else i[1] for i in cuts_bin])
                
            #else break the loop 
            else:
    
                break
                
        return np.unique(cuts).tolist()
    

    def _pretty_bin(self,col,y,cut_off,ws=None):        
    
        if not np.equal(np.unique(y),np.array([0.,1.])).all():
            
            raise ValueError('y values only in (0.,1.)')
        
        # update cut_group    
        cut_off=np.sort(np.unique(cut_off))
            
        cut_off_all=[-np.inf]+cut_off.tolist()+[np.inf]
    
        cut_g=np.digitize(col,cut_off_all,right=False)
        
        cut_bin=np.array([cut_off_all[i:i + 2] for i in range(len(cut_off_all) - 1)])
    
        #sample weights
        if is_array_like(ws):
            
            ws=ws
            
            if ws.size!=y.size:
                
                raise ValueError('length of weight not equal to y')
                
        else:
            
            ws=np.ones(y.size)
        
        gid=np.unique(cut_g)
        count_list=[]
    
        for i in range(len(gid)-1):
            
            y_g_1=y[cut_g==gid[i]]
            ws_g_1=ws[cut_g==gid[i]]        
    
            y_g_2=y[cut_g==gid[i+1]]
            ws_g_2=ws[cut_g==gid[i+1]]
            
            
            # if all vals in y groupby col equal to 0 or 1 then chi2==0
            if (np.append(y_g_1,y_g_2)==0).all() or (np.append(y_g_1,y_g_2)==1).all():
                
                xtab=np.array([[ws_g_1[y_g_1==0].sum(),ws_g_2[y_g_2==0].sum()],
                               [ws_g_1[y_g_1==1].sum(),ws_g_2[y_g_2==1].sum()]])
                
                #merge 1st point in bin when distr too low
                if i==0:
                    
                    count_list.append(xtab[:,0].sum())
                    
                count_list.append(xtab[:,1].sum())

                
            else:
    
                #y-values only in (0,1)
                xtab=np.array([[ws_g_1[y_g_1==0].sum(),ws_g_2[y_g_2==0].sum()],
                               [ws_g_1[y_g_1==1].sum(),ws_g_2[y_g_2==1].sum()]])
                
                #merge 1st point in bin when distr too low
                if i==0:
                    
                    count_list.append(xtab[:,0].sum())
                
                count_list.append(xtab[:,1].sum())
        
        return np.array(count_list),np.array(cut_bin) 
    