#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 23:08:38 2021

@author: zengke
"""

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype,is_string_dtype,is_array_like
from sklearn.base import TransformerMixin
from joblib import Parallel,delayed
import warnings
from itertools import product,groupby
from sklearn.cluster import KMeans
from scipy.stats import chi2


class varReportSinge:
    
    def report(self, X, y,breaks,special_values=[np.nan,'nan'],sample_weight=None):
        
        report_var=self.getReport_Single(X,y,breaks,special_values,sample_weight)
        
        return report_var
        
    def getReport_Single(self,X,y,breaklist_var,special_values,sample_weight):
         

         col=X.name

         #处理缺失值
         var_fillna=X.replace(special_values,np.nan)
         
         #breaklist_var=list(breaks_list_dict[col])
         
         #判断数据类型
         if is_numeric_dtype(var_fillna):
           
             #按照分箱sc的breaklist的区间进行分箱
             var_cut=pd.cut(var_fillna,[-np.inf]+breaklist_var+[np.inf],duplicates='drop',right=False).cat.add_categories('missing')
             
             var_bin=var_cut.fillna('missing')
         
         elif is_string_dtype(var_fillna):    
             
             var_cut=pd.Series(np.where(var_fillna.isnull(),'missing',var_fillna),
                       index=var_fillna.index,
                       name=var_fillna.name)

             #转换字原始符映射到分箱sc的breaklist的字符映射
             var_code_raw=var_cut.unique().tolist()
                          
             map_codes=self.raw_to_bin_sc(var_code_raw,breaklist_var,special_values)
                        
             var_bin=pd.Series(pd.Categorical(var_cut.map(map_codes)),index=var_fillna.index,name=col)
             
             if 'missing' in var_bin.cat.categories:
                 
                 var_bin= var_bin.fillna('missing') 
            
             else:
                 
                 var_bin= var_bin.cat.add_categories('missing').fillna('missing')
             
         else:
             
             raise ValueError('dtypes in X in (number,object),others not support')
         
         #若只计算全量数据则只输出全量的特征分析报告
             
         if is_array_like(sample_weight):         
             var_bin=pd.concat([var_bin,y.mul(sample_weight).rename(y.name)],axis=1) 
             var_bin['sample_weight']=sample_weight
         else:
             var_bin=pd.concat([var_bin,y],axis=1)
             var_bin['sample_weight']=1
         
         #print var_bin
         rename_aggfunc=dict(zip(['sample_weight','target'],['count','bad']))
         result=pd.pivot_table(var_bin,index=col,values=[y.name,'sample_weight'],
                           margins=False,
                           aggfunc='sum').rename(columns=rename_aggfunc,level=0)#.droplevel(level=1,axis=1) 

         var_tab=self.getVarReport_ks(result,col) 
         
         if is_string_dtype(var_fillna):
             
             var_tab.index=var_tab.index.astype('category') 

         return  var_tab.assign(
             
             breaks=var_tab.index.categories.map(lambda x:x if isinstance(x,str) else x.right))
            
    def getVarReport_ks(self,var_ptable,col):
        
        var_ptable['badprob']=var_ptable['bad'].div(var_ptable['count'])
        var_ptable['count_distr']=var_ptable['count'].div(var_ptable['count'].sum())
        var_ptable['good']=var_ptable['count'].sub(var_ptable['bad'])
        var_ptable['good_dis']=var_ptable['good'].div(var_ptable['good'].sum())
        var_ptable['bad_dis']=var_ptable['bad'].div(var_ptable['bad'].sum())
        var_ptable['bin_iv']=var_ptable['bad_dis'].sub(var_ptable['good_dis']).mul(
            (var_ptable["bad_dis"]+1e-10).div((var_ptable["good_dis"]+1e-10)).apply(np.log)
        )
        var_ptable['total_iv']=var_ptable['bin_iv'].sum()
        var_ptable['woe']=(var_ptable["bad_dis"]+1e-10).div((var_ptable["good_dis"]+1e-10)).apply(np.log)
        var_ptable['ks']=var_ptable['good_dis'].cumsum().sub(var_ptable['bad_dis'].cumsum()).abs()
        var_ptable['ks_max']=var_ptable['ks'].max()
        var_ptable['variable']=col
        var_ptable.index.name='bin'
        #var_ptable=var_ptable.reset_index()
        #var_ptable['bin']=var_ptable['bin'].astype('str')
        var_ptable=var_ptable[['variable', 'count', 'count_distr', 'good', 'bad', 'badprob','woe', 'bin_iv', 'total_iv','ks','ks_max']]
        
        return var_ptable
    
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
    
    

class binAdjusterKmeans(TransformerMixin):
    
    """ 
    基于Kmeans的分箱调整算法:一种自动非优化分箱算法        
    一般通过细分箱后各个分箱的BadRate近似时需要合并,本算法可自动实现这一过程。注意‘missing’值将不进行合并
    建议场景:水平非常多的分类特征进行分箱合并时，可先使用此算法进行自动合并
    
    Params:
    ------
    
        breaks_list:分箱字典结构,{var_name:[bin],...},支持scorecardpy与toad的breaks_list结构 
        combine_ratio,float,分箱合并阈值,在bin_limit=1的情况下,阈值越大合并的箱数越多,建议范围0.01-0.1
        bin_limit:int,分箱合并阈值,最终分箱的最低分箱数限制,越低则合并的箱数越多,建议4-6,同时若特征初始分箱小于bin_limit则不执行合并算法
        seed:int,kmeans随机种子,
            + 本算法在合并差距较大的barprob箱时,kmeans的随机性会被放大导致合并结果不可复现,设定seed值以复现合并结果
            + 设定合理的combine_ratio与bin_limit可极大的降低kmeans的随机性
        special_values,特殊值指代值,需与breaks_list一致            
        n_jobs,int,并行数量,默认-1,在数据量较大、列较多的前提下可极大提升效率但会增加内存占用
        verbose,int,并行信息输出等级        
        
    Attributes:
    -------
    """    
    
    def __init__(self,breaks_list,combine_ratio=0.1,bin_limit=5,seed=123,special_values=[np.nan],n_jobs=-1,verbose=0):

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
    
        #global var_bin,res_km_s
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
        
            cate_colname=X[columns].select_dtypes(include='object').columns.tolist()
            num_colname=X[columns].select_dtypes(include='number').columns.tolist()
            oth_colname=X[columns].select_dtypes(exclude=['number','object']).columns.tolist()
            if oth_colname:
                raise ValueError('supported X.dtypes only in (number,object),use bm.dtypeAllocator to format X')

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
    


class binAdjusterChi(TransformerMixin):
    
    def __init__(self,bin_num=10,chi2_p=0.1,special_values=[np.nan],n_jobs=-1,verbose=0):
        """ 
        卡方单调分箱:先等频分箱,再合并低于卡方值(交叉表卡方检验的差异不显著)的分箱或不单调(badrate)的分箱
        + 只针对连续特征,分类特征将被忽略
        + 结果只提供参考，需与其他分箱方法一起使用
        
        Params:
        ------
            bin_num:int,预分箱(等频)箱数,越大的值会使卡方分箱计算量越大,同时会增加分箱精度
            chi2_p:float,卡方分箱的p值,一般p值越小合并的箱越少,越大则合并的箱越多
            special_values:list,特殊值,分箱中将被替换为np.nan
            n_jobs,int,并行数量,默认-1(所有core),在数据量较大特征较多的前提下可极大提升效率
            verbose,int,并行信息输出等级   
            
        Attributes:
        -------
        """    
        self.bin_num=bin_num
        self.chi2_p=chi2_p
        self.special_values=special_values
        self.n_jobs=n_jobs
        self.verbose=verbose

    
    def fit(self, X, y):
        
        if X.size:
            
            X=X.select_dtypes(include='number')
            #print(X.shape)
            
            if X.size:
            
                parallel=Parallel(n_jobs=self.n_jobs,verbose=self.verbose)
                col_break=parallel(delayed(self.chi2_bin)(X[col],y,self.bin_num,self.chi2_p,self.special_values) 
                                   for col in X.columns)
                
                self.col_break=col_break
                self.breaks_list_chi2m = {col:breaks for col,breaks in col_break}
                
            else:
                
                raise ValueError('no numeric columns find in X')                                 
                                    
        return self
    
    
    def transform(self, X):       
        
        if X.size:
            
            return X
        
        else:
            
            warnings.warn('0 rows in input X,return None')
            
            return pd.DataFrame(None)
        
        
    def chi2_bin(self,X,y,max_bin=10,chi2_p=0.5,special_values=[np.nan]): 
    
    
        #global count
        target = y.values
        var_single = X.replace(special_values,np.nan)
        threshold = chi2.isf(chi2_p, df=1)

        if max_bin < 2 or var_single.dropna().unique().size<=1:

            cuts = [0]

            return X.name,cuts

        else:

            #1.eq-freq cut
            cuts = sorted(list(
                set(np.nanpercentile(var_single,np.linspace(0, 1, max_bin + 1)[1:-1] * 100, interpolation='lower')))
                         )
            cutoff = [-np.inf] + cuts + [np.inf]


            #2.produce cutoff pair between each breaks
            seg = [cutoff[i:i + 2] for i in range(len(cutoff) - 1)]


            tb = []
            cutoffs = []        

            #calculate good and bad count
            for index, (p1, p2) in enumerate(seg):

                #fliter y in the X range of cutoffs
                mask = (var_single > p1) & (var_single <= p2)
                yy = target[mask]

                #
                cntr = len(yy) #freq
                cntb = int(yy.sum()) #bad count
                cntg = cntr - cntb #good count
                if cntr > 0:
                    tb.append([cntb, cntg])
                    if p2 < np.inf:
                        cutoffs.append(p2)

            freq_tb = np.array(tb)#二维数组
            cutoffs = sorted(cutoffs)
            monot_ok = False

            count=0
            while (len(freq_tb) > 1):

                count=count+1
                minidx = 0
                minvalue = np.inf
                for i in range(len(freq_tb) - 1):

                    a, b = freq_tb[i]
                    c, d = freq_tb[i + 1]
                    N = (a + b + c + d)

                    #calculate chi2

                    #ccsq,_,_,_,=chi2_contingency(freq_tb[i:i+2],correction=False) #calculate chi2 using scipy
                    ccsq = N*(a*d-b*c)**2/((a+b)*(a+c)*(c+d)*(b+d))

                    chiv = ccsq
                    chiv = 0 if (a + c) == 0 or (b + d) == 0 else chiv

                    #chiv should not be infinite
                    if minvalue > chiv:
                        minvalue = chiv
                        minidx = i

                #chi2<user defined threshold or not monot_ok then combine
                if (minvalue < threshold) or (not monot_ok):
                    #print(cutoffs,cutoffs[minidx])
                    cutoffs = np.delete(cutoffs, minidx)

                    tmp = freq_tb[minidx] + freq_tb[minidx + 1]

                    freq_tb[minidx] = tmp
                    freq_tb = np.delete(freq_tb, minidx + 1, 0)
                    bad_rate = [i[0]/(i[1]+i[0]) for i in freq_tb]
                    bad_rate_monotone = [(bad_rate[i-1]<bad_rate[i]<bad_rate[i+1])or
                                     (bad_rate[i-1]>bad_rate[i]>bad_rate[i+1]) 
                                     for i in range(1,len(bad_rate)-1)]

                    monot_ok = np.array(bad_rate_monotone).all()

                else:

                    break

            if cutoffs.tolist():
                
                if len(cutoffs)==1:
                
                    return X.name,[0]
                
                elif var_single.max()==max(cutoffs):
                    
                    return X.name,cutoffs[:-1].tolist()
                
                else:
                    
                    return X.name,[0]

            else:

                return X.name,[0]
            


class binTree(TransformerMixin):
    
    """ 
    决策树递归最优分箱
    
    Params:
    ------
    max_bin=50,初始分箱数，越多的初始分箱数能够得到越好的最优分箱点，但会增加计算量。max_bin=50时scorecardpy一致
    criteria='iv',决策树进行分割的指标，目前支持iv与ks
    max_iters=100,决策树递归次数
    tol=1e-4,决策树进行分割的指标的增益小于tol时停止分割
    distr_limit=0.05,每一箱的样本占比限制
    bin_num_limit=8,分箱总数限制
    coerce_monotonic=False,是否强制bad_prob单调，默认否
    ws=None,None or pandas.core.series.Series,样本权重
    special_values=[np.nan,'nan'],特殊值指代值列表，其将被转换为np.nan(数值列)或missing(字符列),若不同列的特殊指代值存在冲突，请先处理好这些冲突
    n_jobs=-1,int,并行数量,默认-1,在数据量较大、列较多的前提下可极大提升效率但会增加内存占用
    verbose=0,并行信息输出等级    
        
    Attributes:
    -------
    """    
    
    def __init__(self,max_bin=50,criteria='iv',max_iters=100,
                 tol=1e-4,distr_limit=0.05,bin_num_limit=8,coerce_monotonic=False,
                 ws=None,special_values=[np.nan,'nan'],n_jobs=-1,verbose=0):

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
        
        if X.size:
            
            p=Parallel(n_jobs=self.n_jobs,verbose=self.verbose)
            res=p(delayed(self.get_treecut)(col[1],y,self.max_bin,
                                            self.criteria,self.max_iters,
                                            self.tol,self.distr_limit,
                                            self.bin_num_limit,
                                            self.ws,self.special_values,
                                            self.coerce_monotonic) for col in X.iteritems())
            
            self.breaks_list={col_name:breaks for col_name,breaks,_ in res}
            self.bin={col_name:vtab for col_name,_,vtab in res}
                                    
        return self
    
    
    def transform(self, X):       
        
        if X.size:
            
            return X
        
        else:
            
            warnings.warn('0 rows in input X,return None')
            
            return pd.DataFrame(None)
        
        
    def get_treecut(self,col,y,max_bin,criteria,max_iters,tol,distr_limit,bin_num_limit,ws,special_values,coerce_monotonic):
        
        #sample_wieght
        if ws is not None:
            
            ws=ws.values()
        
        else:
            
            ws=None
        
        
        #numeric column
        if is_numeric_dtype(col):
                      
            col=col.replace(special_values,np.nan)
            
            #no cut applied when col's unique value pop too high               
            if col.value_counts(dropna=False).div(col.size).max()>0.95:
                
                breaks=[]       
                
                vtab=varReportSinge().report(col,y,breaks,special_values=special_values,sample_weight=ws)
                
            elif np.unique(col[~np.isnan(col)]).size==1:
                
                breaks=[]
                
                vtab=varReportSinge().report(col,y,breaks,special_values=special_values,sample_weight=ws)
            
            #tree cut
            else:
                
                breaks=self.get_bestsplit(col.values,y.values,max_bin=50,
                                     criteria=criteria,
                                     max_iters=100,
                                     tol=0.0001,
                                     ws=ws,
                                     distr_limit=distr_limit,
                                     is_str_dtype=False,
                                     coerce_monotonic=coerce_monotonic,
                                     bin_num_limit=bin_num_limit)
                
                vtab=varReportSinge().report(col,y,breaks,special_values=special_values,sample_weight=ws)
        
        #string columns         
        elif is_string_dtype(col):
            
            col=col.replace(special_values,'missing')
            
            #sort levels by bad_rate
            codes=y.groupby(col).mean().sort_values().index.tolist()
            
            #ordinal encode data start with 0
            map_code=dict(zip(codes,list(range(len(codes)))))

            #tree cut
            breaks_raw=self.get_bestsplit(col.map(map_code).values,
                                     y.values,
                                     criteria=criteria,
                                     max_iters=100,
                                     tol=1e-4,
                                     ws=ws,
                                     distr_limit=distr_limit,
                                     is_str_dtype=True,
                                     bin_num_limit=bin_num_limit)
        
            breaks=['%,%'.join(i) for i in np.split(codes,breaks_raw)]    
            
            vtab=varReportSinge().report(col,y,breaks,special_values=special_values,sample_weight=ws)
            
        else:
            
            raise ValueError("col's dtype in ('number','object')")
  

        return col.name,breaks,vtab


    def get_bestsplit(self,col,y,max_bin=50,ws=None,criteria='iv',tol=1e-4,
                      max_iters=100,distr_limit=0.05,bin_num_limit=8,
                      is_str_dtype=False,coerce_monotonic=False): 
        
        #get sample_weight
        if is_array_like(ws):
            
            y = y * ws
            
            count=ws
            
        else:
            
            count=np.ones(col.size)    
        
        
        nan_sum=pd.isnull(col).sum()
            
        #string dtype variable
        if is_str_dtype:
            
            cuts_remain=np.unique(col)            
        
        #number dtype variable
        else:
                    
            #adjust max_bin for improving performance
            if np.unique(col).size<max_bin:

                max_bin_adj=np.unique(col).size

            else:

                max_bin_adj=max_bin    

            #remove outliers
            col_rm_outlier=self.remove_outlier(col)

            #R pretty bins:cut points looks better but will lose iv or ks gain
            cuts_remain=self.R_pretty(col_rm_outlier.min(),col_rm_outlier.max(),max_bin_adj) 

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
            
            for point in cuts_remain:            
                
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
                        
                    elif coerce_monotonic and not is_str_dtype and not self.is_monotonic(bad_prob) and nan_sum==0:                    

                        cuts_remain=cuts_remain[cuts_remain!=point]
                
                    elif coerce_monotonic and not is_str_dtype and not self.is_monotonic(bad_prob[:-1]) and nan_sum>0:                 

                        cuts_remain=cuts_remain[cuts_remain!=point]
    
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
                
                print("len(cuts_remain)==0")
                
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
                if best_point_dist_min<=distr_limit:
    
                    cuts_tree.remove(best_point)
    
                    best_criteria=best_criteria[:-1]
    
                bin_num=len(cuts_tree)+1 #bin num 
            
                if iters>max_iters: 
    
                    print("max_iters reach")
    
                    break
    
                if best_criteria_gain<tol:
    
                    print("best_criteria_gain reach")
    
                    break
    
                if bin_num>=bin_num_limit:
    
                    print("bin_num_limit reach")
    
                    break
            
        return (sorted(cuts_tree))


    def R_pretty(self,low, high, n):
        '''
        pretty breakpoints, the same as pretty function in R,
        
        Params
        ------
        low: minimal value 
        low: maximal value 
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
    
    
    def remove_outlier(self,col):
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
    
    
    def is_monotonic(self,col):    
    
        return np.all(np.diff(col) > 0) or np.all(np.diff(col) <0)



            
            
            
            
            
            