# -*- coding: utf-8 -*-
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
import warnings
from BDMtools.report_report import varReport
from BDMtools.selector_bin_fun import binFreq,binPretty,binTree,binChi2,binKmeans
#from joblib import Parallel,delayed
#from pandas.api.types import is_numeric_dtype


class binSelector(TransformerMixin):    
    
    """ 
    最优分箱
    Parameters:
    ----------
        method:str,分箱方法
            + ‘freq’:数值等频分箱，分类特征按其类别分箱
            + ‘freq-kmeans’:基于Kmeans，对freq-cut结果进行自动调整，以将badrate近似的箱进行合并
            + 'pretty':使用Pretty Breakpoints获取数值特征分箱点
                + pretty分箱点更加美观，适合报告、绘图
                + 详见R的pretty函数           
            + 'tree':决策树,递归分裂iv/ks增益最高的切分点形成新分箱直到达到终止条件
            + 'chi2':卡方,先等频预分箱,再递归合并低于卡方值(交叉表卡方检验的差异不显著)的分箱
        max_bin:int,预分箱数,越多的预分箱数越有可能得到越好的分箱点，但会增加计算量,不适用于method=‘freq’
            + method=‘pretty’时代表pretty预分箱数 
            + method=‘freq-kmeans’时代表freq预分箱数               
            + method='tree'时,代表pretty预分箱数        
            + method='chi2'时,代表pretty预分箱数        
        distr_limit,最终箱样本占比限制,不适用于method=‘freq’
            + method='pretty'时，箱最终箱样本占比限制,
            + method='freq-kmeans'时，箱最终箱样本占比限制
            + method='tree'时,箱最终箱样本占比限制
            + method='chi2':时,箱最终箱样本占比限制
        bin_num_limit,
            + method=‘freq’时代表等频分箱数
            + method='freq-kmeans'时，合并分箱最低限制,bin_num_limit<max_bin时才有效果
            + method='pretty'时，代表分箱数限制
            + method='tree'时,代表分箱数限制,实际分箱数将小于等于改值
            + method='chi2':卡方,代表分箱数限制,实际分箱数将小于等于改值
        coerce_monotonic=False,是否强制数值特征的bad_prob单调，默认否
            强制bad_prob单调适用于所有本模块所支持的分箱算法
            若分箱后的x与y本身有单调关系,则强制单调能够取得理想的结果,若分箱后x的woe与y无关系非单调相关,则强制单调效果将不佳
            +  method='freq'时，将先强制freq cut单调，此时分箱结果将可能低于bin_num_limit,分箱占比也将发生变化  
            +  method='freq-kmeans'时，将先强制freq cut单调，在适用keamns算法进行合并            
            +  method='pretty'时，将强制pretty cut的预分箱单调，再根据条件合并分箱
            +  method='tree'时,最优分割过程中加入单调性限制,强制每一个新的加入的分割点都必须先使bad_rate单调
            +  method='chi2':先在预分箱中强制单调，再进行卡方分箱以保证卡方分箱单调
        sample_weight=None,样本权重,非0
            + 若数据进行过抽样，则可设定sample_weight
            + 其将会影响最终vtable的每一箱的count,bad,good,bad_prob,iv,ks等,
            + 若只对好坏样本进行加权则只会影响bad_prob
            + 当method in ('tree','chi2')时若sample_weight非空，则算法会计算加权后的iv_gain,ks_gain或卡方值
        special_values,缺失值指代值
            + None,保证数据默认
            + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
            + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan  
        iv_limit=0.02:float,IV阈值,IV低于该阈值特征将被剔除
        keep=None,list or None,保留列的列名list             
        n_jobs:int,列并行计算job数,默认-1,并行在数据量较大，特征较多时能够提升效率，但会增加内存消耗
        verbose:int,并行计算信息输出等级
        
    Attribute:
    ----------

    """
    
    def __init__(self,method='freq',max_bin=50,distr_limit=0.05,bin_num_limit=8,special_values=None,
                 iv_limit=0.02,keep=None,sample_weight=None,coerce_monotonic=False,
                 breaks_list_adj=None,n_jobs=-1,verbose=0):
        
        self.method=method
        self.max_bin=max_bin
        self.distr_limit=distr_limit
        self.bin_num_limit=bin_num_limit
        self.iv_limit=iv_limit
        self.keep=keep
        self.special_values=special_values
        self.breaks_list_adj=breaks_list_adj
        self.coerce_monotonic=coerce_monotonic
        self.sample_weight=sample_weight
        self.n_jobs=n_jobs
        self.verbose=verbose
        
    def transform(self,X,y=None):
        
        return X[self.keep_col]
              
    def fit(self,X,y):
        """ 
        
        """    
                  
        if y.name:
            
            self.target=y.name
            
        else:
            raise ValueError('name y using pd.Series(y,name=yname)')        
        
        #get bins using given breaks_list_adj
        if self.breaks_list_adj:                     

            self.keep_col=list(self.breaks_list_adj.keys())
            
            self.adjbin=varReport(breaks_list_dict=self.breaks_list_adj,
                          special_values=self.special_values,                       
                          n_jobs=self.n_jobs,
                          verbose=self.verbose).fit(X[self.keep_col],y).var_report_dict
            
                    
        #get bins using algorithms                                
        else:
            
            if self.method == 'freq':
                
                #using freq cut
                self.breaks_list,bin_res=binFreq(X,y,
                                         bin_num_limit=self.bin_num_limit,
                                         special_values=self.special_values,
                                         ws=self.sample_weight,
                                         coerce_monotonic=self.coerce_monotonic
                                         )

            elif self.method == 'freq-kmeans':
                
                #using freq-kmeans to combine bins with similar badprob after freq cut
                breaks_list_freq,_=binFreq(X,y,
                                         bin_num_limit=self.max_bin,
                                         special_values=self.special_values,
                                         ws=self.sample_weight,
                                         coerce_monotonic=self.coerce_monotonic
                                         )
                
                res_Kmeans=binKmeans(breaks_list=breaks_list_freq,
                                     combine_ratio=0.1,
                                     bin_limit=self.bin_num_limit,
                                     seed=123,
                                     sample_weight=self.sample_weight,
                                     special_values=self.special_values,
                                     n_jobs=self.n_jobs,
                                     verbose=self.verbose).fit(X,y)
                
                self.breaks_list=res_Kmeans.breaks_list                
                
                bin_res=res_Kmeans.bins
                
            elif self.method == 'pretty':
                
                #using pretty-cuts
                res_pretty=binPretty(max_bin=self.max_bin,distr_limit=self.distr_limit,bin_num_limit=self.bin_num_limit,
                                 coerce_monotonic=self.coerce_monotonic,ws=self.sample_weight,
                                 special_values=self.special_values,n_jobs=self.n_jobs,verbose=self.verbose).fit(X,y)
                    
                self.breaks_list=res_pretty.breaks_list
                
                bin_res=res_pretty.bins         
            

            elif self.method == 'tree':  
                
                #using treecut
                res_tree=binTree(max_bin=self.max_bin,criteria='iv',distr_limit=self.distr_limit,
                                 bin_num_limit=self.bin_num_limit,ws=self.sample_weight,
                                 coerce_monotonic=self.coerce_monotonic,
                                 special_values=self.special_values,n_jobs=self.n_jobs,
                                 verbose=self.verbose).fit(X,y)
                    
                self.breaks_list=res_tree.breaks_list

                bin_res=res_tree.bins     
                
            elif self.method == 'chi2':  

                #using chi2merge
                res_chi2=binChi2(max_bin=self.max_bin,tol=0.1,distr_limit=self.distr_limit,bin_num_limit=self.bin_num_limit,
                                 coerce_monotonic=self.coerce_monotonic,ws=self.sample_weight,
                                 special_values=self.special_values,n_jobs=self.n_jobs,verbose=self.verbose).fit(X,y)
                    
                self.breaks_list=res_chi2.breaks_list
                
                bin_res=res_chi2.bins                                     
                                                
            else:
                
                raise ValueError("method in ('freq','pretty','pretty-kmeans','chi2','tree')")                              
            
            
            #get iv and ks 
            optbindf_ks=pd.concat(bin_res.values())           

            self.iv_info=optbindf_ks.groupby('variable')['bin_iv'].sum().rename('total_iv')
            self.ks_info=optbindf_ks.groupby('variable')['ks'].max().rename('ks_max')
            
            #fliter by iv
            self.keep_col=self.iv_info[self.iv_info>=self.iv_limit].index.tolist()  
            
            if not self.keep_col:
                
                warnings.warn('iv_limit too high to keep any variables,reset iv_limit')  
            
            #keep user-defined columns    
            if self.keep:
                
                if not np.isin(self.keep,X.columns.tolist()).all():
                    
                    raise ValueError("keep columns not in X")     
                    
                self.keep_col=list(set(self.keep_col+self.keep))  
            
            #keep bin info and breaks info for checking and rebinning
            self.bins={column:bin_res.get(column) for column in self.keep_col}
            self.breaks_list={column:self.breaks_list.get(column) for column in self.keep_col}          
       
        return self

    def getBreaklist_sc(self,break_list,X,y):
        
        """
        将toad的breaklist结构转化为scorecardpy可用的结构
        """      
        
        
        #breaks_list in format of sc or toad
        count=0
        for var_list in list(break_list.values()):
            
            for value in var_list:
                if isinstance(value,list):
                    count=count+1           
                break
            
        columns=list(break_list.keys())
        
        #toad breaks_list transform to sc breaks_list
        #if string col in data
        if count>0:
        
            cate_colname=X[columns].select_dtypes(include='object').columns.tolist()
            num_colname=X[columns].select_dtypes(include='number').columns.tolist()
            oth_colname=X[columns].select_dtypes(exclude=['number','object']).columns.tolist()
            if oth_colname:
                raise ValueError('supported X.dtypes only in (number,object),use bm.dtypeAllocator to format X')

            break_list_sc=dict()

            for key in break_list.keys():
                
                #for string columns
                if key in cate_colname and break_list[key]: 

                    bin_value_list=[]
                    
                    for value in break_list[key]:

                        bin_value_list.append('%,%'.join(value))

                    break_list_sc[key]=bin_value_list
                
                #for number columns
                elif key in num_colname and break_list[key]:
                    
                    break_list_sc[key]=break_list[key]
                
                #for breaks=[]
                else:

                    break_list_sc[key]=[-np.inf,np.inf]
                    
        #if no string col in data
        else:   
            
            break_list_sc=dict()

            for key in break_list.keys():
                
                if not break_list[key]:
                    
                    break_list_sc[key]=[-np.inf,np.inf]
                    
                else:
                    
                    break_list_sc[key]=break_list[key]
                
        return break_list_sc                
    
    
    def fit_adjustBin(self):
        """
        developing...        
        """          
        return self   
