# -*- coding: utf-8 -*-
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
import scorecardpy as sc
import warnings
from BDMtools.report_report import varReport
from BDMtools.selector_bin_fun import binAdjusterKmeans,binAdjusterChi,binTree
#from joblib import Parallel,delayed
#from pandas.api.types import is_numeric_dtype


class binSelector(TransformerMixin):    
    
    """ 
    最优分箱与交互分箱
    Parameters:
    ----------
        method:str,分箱方法
            + ‘freq’:等频分箱
            + ‘freq-kmeans’:等频kmeans分箱，基于Kmeans，对等频分箱结果进行自动调整，以将badrate近似的箱进行合并
            + 'tree':决策树,递归分裂iv/ks增益最高的切分点形成新分箱直到达到终止条件
            + 'chi':卡方,先等频预分箱,再递归合并低于卡方值(交叉表卡方检验的差异不显著)的分箱，开发中
            + 'chi_m':卡方单调,先等频预分箱,再合并低于卡方值(交叉表卡方检验的差异不显著)的分箱或不单调(badrate)的分箱    
                      注意chi_m只适用于数值列,字符列将在breaks_list中被剔除
        max_bin:int,预分箱数,越多的预分箱数能够得到越好的分箱点，但会增加计算量
            + method=‘freq-kmeans’时代表等频预分箱数              
            + method='chi_m'时代表卡方单调分箱的等频预分箱数   
            + method='tree'时,代表pretty预分箱数              
        distr_limit,method为'tree'和'chi'时代表分箱最终箱样本占比限制
        bin_num_limit,
            + method=‘freq’时代表等频分箱数
            + method='freq-kmeans'时，合并分箱最低限制,bin_num_limit<n_bins时才有效果
            + method='tree'时,代表分箱数限制
            + method='chi':卡方,代表分箱数限制,开发中 
        coerce_monotonic=False,是否强制bad_prob单调，默认否
            强制bad_prob单调目前仅作用于method='tree'，后续将支持更多分箱算法
        sample_weight=None,样本权重，主要用于调整分箱后的坏样本率
        special_values,list,缺失值、特殊值指代值,数值特征被替换为np.nan，分类特征将被替换为'missing'
        iv_limit=0.02:float,IV阈值,IV低于该阈值特征将被剔除
        keep=None,list or None,保留列的列名list             
        n_jobs:int,列并行计算job数,默认-1,并行在数据量较大，特征较多时能够提升效率，但会增加内存消耗
        verbose:int,并行计算信息输出等级
    Attribute:
    ----------
        features_info:dict,每一步筛选的特征进入记录
    """
    
    def __init__(self,method='freq',max_bin=50,distr_limit=0.05,bin_num_limit=8,special_values=[np.nan,'nan'],
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
        最优分箱
        """              
        if y.name:
            self.target=y.name
        else:
            raise ValueError('name y using pd.Series(y,name=yname)')        
        
        #若给定分箱breaklist已知,可直接调用并进行分箱与后续woe编码
        if self.breaks_list_adj:
                     
            
            #只选择需要调整分箱的特征进行分箱
            self.keep_col=list(self.breaks_list_adj.keys())
            
            # self.adjbin_woe=sc.woebin(X[self.keep_col].join(y).replace(self.special_values,np.nan),
            #                           y=y.name,
            #                           breaks_list=self.break_list_adj,check_cate_num=False)
            
            self.adjbin_woe=varReport(breaks_list_dict=self.breaks_list_adj,
                          special_values=self.special_values,                       
                          n_jobs=self.n_jobs,
                          verbose=self.verbose).fit(X[self.keep_col],y).var_report_dict
            
                    
        #若不给定breaklist，则进行最优分箱                                   
        else:
            
            if self.method=='freq-kmeans':
            
                self.breaks_list=self.getBreakslistFinbin(X,y,self.max_bin,self.special_values)
                self.breaks_list=binAdjusterKmeans(breaks_list=self.breaks_list,
                                                   bin_limit=self.bin_num_limit,
                                                   special_values=self.special_values,
                                                   n_jobs=self.n_jobs,
                                                   verbose=self.verbose,
                                                   ).fit(X,y).breaks_list_adj
                keep_col=list(self.breaks_list.keys())
                
                bin_res=varReport(breaks_list_dict=self.breaks_list,
                              special_values=self.special_values,                       
                              n_jobs=self.n_jobs,
                              verbose=self.verbose).fit(X[keep_col],y).var_report_dict
             
            elif self.method=='freq':
                
                self.breaks_list=self.getBreakslistFinbin(X,y,self.bin_num_limit,self.special_values)
                
                keep_col=list(self.breaks_list.keys())
                #最优分箱的特征iv统计值
                
                bin_res=varReport(breaks_list_dict=self.breaks_list,
                              special_values=self.special_values,                       
                              n_jobs=self.n_jobs,
                              verbose=self.verbose).fit(X[keep_col],y).var_report_dict
            
            #卡方单调分箱
            elif self.method == 'chi_m':
            
                #注意chi_m只适用于数值列,字符列将在breaks_list中被剔除
                self.breaks_list=binAdjusterChi(bin_num=self.max_bin,
                                                   chi2_p=0.1,
                                                   special_values=self.special_values,
                                                   n_jobs=self.n_jobs,
                                                   verbose=self.verbose).fit(X, y).breaks_list_chi2m
                
                self.breaks_list=self.getBreakslistFinbin(X,y,self.max_bin,self.special_values)
                
                keep_col=list(self.breaks_list.keys())
                #最优分箱的特征iv统计值
                
                bin_res=varReport(breaks_list_dict=self.breaks_list,
                              special_values=self.special_values,                       
                              n_jobs=self.n_jobs,
                              verbose=self.verbose).fit(X[keep_col],y).var_report_dict
                                
                
            elif self.method in('chi','tree'):  
                
                #使用决策树最优分箱，卡方分箱仍在开发中后续会进行更新
                res_tree=binTree(max_bin=self.max_bin,criteria='iv',distr_limit=self.distr_limit,
                                 bin_num_limit=self.bin_num_limit,ws=self.sample_weight,
                                 coerce_monotonic=self.coerce_monotonic,
                                 special_values=self.special_values,n_jobs=self.n_jobs,
                                 verbose=self.verbose).fit(X,y)
                    
                self.breaks_list=res_tree.breaks_list
                keep_col=list(self.breaks_list.keys())
                
                bin_res=res_tree.bin     
                                                
            else:
                
                raise ValueError("method in ('freq','freq-kmeans','chi_m','chi','tree')")                              
            
            
            optbindf_ks=pd.concat(bin_res.values())           

            self.iv_info=optbindf_ks.groupby('variable')['bin_iv'].sum().rename('total_iv')
            self.ks_info=optbindf_ks.groupby('variable')['ks'].max().rename('ks_max')
            
            #IV筛选特征
            self.keep_col=self.iv_info[self.iv_info>=self.iv_limit].index.tolist()  
            
            if not self.keep_col:
                
                warnings.warn('iv_limit too high to keep any variables,reset iv_limit')  
                
            if self.keep:
                
                self.keep_col=list(set(self.keep_col+self.keep))  
            
            self.bin={column:bin_res.get(column) for column in self.keep_col}
            self.breaks_list={column:self.breaks_list.get(column) for column in self.keep_col}          
       
        return self

    def getBreaklist_sc(self,break_list,X,y):
        
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

                        bin_value_list.append('%,%'.join(value))

                    break_list_sc[key]=bin_value_list
                
                #数值列默认
                elif key in num_colname and break_list[key]:
                    
                    break_list_sc[key]=break_list[key]
                
                #空breaklist调整格式
                else:

                    break_list_sc[key]=[-np.inf,np.inf]
                    
        #sc格式/toad全数值格式
        else:   
            
            break_list_sc=dict()

            for key in break_list.keys():
                
                if not break_list[key]:
                    
                    break_list_sc[key]=[-np.inf,np.inf]
                    
                else:
                    
                    break_list_sc[key]=break_list[key]
                
        return break_list_sc 
    

    def getBreakslistFinbin(self,X,y,bin_num,special_values):

        """
        等频分箱产生sc.woebin可用的breaklist,用于细分箱
        Parameters:
        --
            X:特征数据,pd.DataFrame
            y:目标变量列,pd.Series,必须与X索引一致
        """
        
        CatCol=X.select_dtypes(include='object').columns.tolist() #分类列
        NumCol=X.select_dtypes(include='number').columns.tolist() #数值列
        OthCol=X.select_dtypes(exclude=['number','object']).columns.tolist()
        if OthCol:
            raise ValueError('supported X.dtypes only in (number,object),use bm.dtypeAllocator to format X')

        breaklist={}
        
        X=X

        #bin_num下若特征分布过于集中,等频分箱将合并集中的分箱区间为最终分箱区间
        for numcol in NumCol:
            
            numcol_s=X[numcol].replace(special_values,np.nan)
            
            if numcol_s.dropna().size:
            
                _,breaks=pd.qcut(numcol_s.dropna(),bin_num,duplicates='drop',retbins=True,precision=3)
                
            else:
                
                breaks=[0]
            
            if len(breaks)==2:
                
                breaklist[numcol]=breaks.tolist()[:-1]
                
            elif len(breaks)==1:
                
                breaklist[numcol]=breaks.tolist()
                
            elif len(breaks)==0:
                
                breaklist[numcol]=numcol_s.dropna().unique().tolist()
                
            else:
                
                breaklist[numcol]=breaks.tolist()[1:-1]
            
        #分类变量则根据类别数直接分箱    
        for catcol in CatCol:
            
            catcol_s=X[catcol].replace(special_values,'missing')
            
            breaklist[catcol]=catcol_s.unique().tolist()
                
        return breaklist                  
    
    
    def fit_adjustBin(self,X,y,br_to_adjusted,special_values):
        """
        根据最优分箱结果调整分箱，需先运行fit
        plt.rcParams["figure.figsize"] = [10, 5]
        Parameters:
        --       
        br_to_adjusted:
        
        Return:
        --
        self.break_list_adj
        self.adjbin_woe
        
        """          
            
        keep_col=list(br_to_adjusted.keys())
        
        
        bin_sc=sc.woebin(X[keep_col].join(y),
                         y=y.name,
                         breaks_list=br_to_adjusted)
        
        breaks_str=sc.woebin_adj(
            dt=X[keep_col].join(y),
            y=y.name,
            adj_all_var=True,
            count_distr_limit=0.05,
            bins=bin_sc,
            method='chimerge'
        ) 
        
        exec('self.breaks_dict_raw='+breaks_str)

        self.break_list_adj={key:np.sort(self.breaks_dict_raw[key]).tolist() for key in list(self.breaks_dict_raw.keys())}
        
        del self.breaks_dict_raw
        
        self.adjbin_woe=sc.woebin(X[keep_col].join(y).replace(special_values,np.nan),
                                  y=y.name,breaks_list=self.break_list_adj)
                    
        return self   
