#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 23:08:38 2021

@author: zengke
"""
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
import warnings
import re
from BDMLtools.base import Base
from BDMLtools.selector.bin_fun import binFreq,binPretty,binTree,binChi2,binKmeans
from BDMLtools.report.report import varGroupsReport,varReportSinge
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype,is_numeric_dtype
from BDMLtools.plotter.base import BaseWoePlotter



class binSelector(Base,TransformerMixin):    
    
    """ 
    自动分箱
    本模块提供自动分箱方法包括等频、kmeans，pretty，决策树、卡方等
    此外字符类数据的levels不要为'','missing','special',None等特殊字符

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
            若分箱后x与y本身有理想的单调关系,则强制单调能够取得理想的结果,若分箱后x的woe与y无关或非单调相关(例如U型),则强制单调后的分箱效果不佳
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
        levels=10,int,分类变量若水平大于levels将被剔除不进行分箱
        special_values,特殊值指代值,若数据中某些值或某列某些值需特殊对待(这些值不是np.nan)时设定
            + None,保证数据默认
            + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换为special箱
            + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换为special箱
        iv_limit=0.02:float,IV阈值,IV低于该阈值特征将被剔除
        keep=None,list or None,保留列的列名list,其将保留于self.keep_col中但不会产生特征分析报告，通过transform筛选后的数据将保留这些特征 
        n_jobs:int,列并行计算job数,默认-1,并行在数据量较大，特征较多时能够提升效率，但会增加内存消耗
        verbose:int,并行计算信息输出等级
        
    Attribute:
    ----------
        keep_col:list经分箱、iv筛选后还存在的列的列名list，
        breaks_list:dict,经指定方法分箱后产生的分箱点list
        bins:dict,经指定方法分箱后产生的特征分析报告
        iv_info:pd.Series,分箱后各个特征的iv
        ks_info:pd.Series,分箱后各个特征的ks
    """
    
    def __init__(self,method='freq',max_bin=50,distr_limit=0.05,bin_num_limit=8,special_values=None,
                 iv_limit=0.02,keep=None,sample_weight=None,coerce_monotonic=False,levels=10,n_jobs=-1,verbose=0):
        
        self.method=method
        self.max_bin=max_bin
        self.distr_limit=distr_limit
        self.bin_num_limit=bin_num_limit
        self.iv_limit=iv_limit
        self.keep=keep
        self.special_values=special_values
        self.coerce_monotonic=coerce_monotonic
        self.levels=levels
        self.sample_weight=sample_weight
        self.n_jobs=n_jobs
        self.verbose=verbose
        
        self._is_fitted=False
        
    def transform(self,X,y=None):
        
        self._check_is_fitted()
        self._check_X(X)
        
        return X[self.keep_col]

    def fit(self,X,y):
        """ 
        
        """    
        
        self._check_data(X, y)        
        self._check_colname(X)
       
        if y.name:
            
            self.target=y.name
            
        else:
            
            raise ValueError('name y using pd.Series(y,name=yname)')        
            
        X=X.drop(self._check_levels(X,self.levels),axis=1)
        
        if not X.size:
            
            raise ValueError('no variable to get binning!')
        
            
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
            
            raise ValueError("method in ('freq','pretty','freq-kmeans','chi2','tree')")                              
        
        
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
            
        self._is_fitted=True
       
        return self     
    
    def _check_levels(self,X,level_lim=10):
    
        X_c=X.select_dtypes('object')
        
        if X_c.columns.size:
            
            c_l=X_c.apply(lambda x:x.nunique())
            
            return c_l[c_l>level_lim].index.tolist()
        
        else:
            
            return []

    
class binAdjuster(Base,BaseWoePlotter):  
    
    """ 
    交互式分箱,支持单特征、组特征的交互式分箱及分箱调整
    
    Parameters:
    ----------
    breaks_list_dict:dict,需要调整的特征分箱字典
    column:str,来自X的组变量,用于分组调整,且只支持单个组特征
    sort_column:list,排序组变量水平,必须涵盖组的所有的水平
    psi_base:str,若column不为None时,进行组分箱调整时的psi基准,需符合X.query(str)语法
        + 'all':以特征在全量数据的分布为基准
        + user-define:用户输入支持X.query的表达式以确定base        
    special_values:特殊值指代值
        + None,保持数据默认
        + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
        + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
    sample_weight:numpy.array or pd.Series(...,index=X.index) or None,样本权重，若数据是经过抽样获取的，则可加入样本权重以计算加权的badrate,woe,iv,ks等指标以还原抽样对分析影响
    figure_size:tuple,特征分析图的图形大小
    
    Attribute:
    ----------
        breaks_list_adj:dict,经调整后的分箱结果
        vtabs_dict_adj:dict,经调整后的特征分析报告
        
    Method:
    ----------     
        fit(X,y):给定X,y并开始分箱调整
        transform(X):给定X并根据调整结果进行特征选择
    
    binAdjuster的交互内容    
    
        1: next:当前特征分箱完毕,跳转到下个特征
        2: yes:调整当前特征分箱:
            输入需调整的分箱:
                + 连续:输入[数值1,数值2,...]调整分段继续，
                    - 分段中不用写最大/最小值
                    - 若输入空白则会在全数据上进行最优分箱
                + 分类:输入[字符1,字符2,...]调整分段继续，
                    - 其中若合并分类特征写成“字符3%,%字符4”
                    - 其中字符必须涵盖该分类特征的所有水平,若有遗漏则将被转换为missing            
        3: back :返回前一个特征并进行调整
        4: remove :当前特征分箱无法调整至合理水平,在调整最终结果中剔除该特征信息
                + 只要某特征被选择为remove,那么该特征无论调整了多少次分箱都会被最终从结果中剔除
        0: exit:终止分箱程序
            + 输入"y"终止,其他则继续
            
    """         
        
    
    
    def __init__(self,breaks_list_dict,column=None,sort_column=None,psi_base='all',
                 special_values=None,sample_weight=None,figure_size=None):
        
        self.breaks_list_dict=breaks_list_dict
        self.column=column
        self.sort_column=sort_column
        self.psi_base=psi_base
        self.special_values=special_values
        self.sample_weight=sample_weight  
        self.figure_size=figure_size

        self._is_fitted=False
        
    def fit(self,X,y):
        
        self._check_data(X,y)                
        self._check_colname(X) 
        self.breaks_list_dict=self._check_breaks(self.breaks_list_dict)
            
        if not np.all(np.isin(list(self.breaks_list_dict),X.columns)):
            
            raise ValueError("breaks_list_dict contains colname not in X")
            
            
        if self.column is None:
            
            breaks_list_adj,vtabs_dict_adj=self._get_breaks_adj(self.breaks_list_dict,
                                                                X,y,
                                                                sample_weight=self.sample_weight,
                                                                special_values=self.special_values,
                                                                figure_size=self.figure_size)
            
        else:
            
            breaks_list_adj,vtabs_dict_adj=self._get_breaks_adj_g(self.breaks_list_dict,
                                                                X,y,
                                                                column=self.column,
                                                                sort_column=self.sort_column,
                                                                psi_base=self.psi_base,
                                                                sample_weight=self.sample_weight,
                                                                special_values=self.special_values,
                                                                figure_size=self.figure_size)           
            
            
        self.breaks_list_adj=self._check_breaks(breaks_list_adj)
            
        self.vtabs_dict_adj=vtabs_dict_adj

        self._is_fitted=True
        
        return self
              

    def transform(self,X,y=None):
        
        self._check_is_fitted()
        self._check_X(X)
        
        return X[self.breaks_list_dict.keys()]
        
    
    def _split_by_re(self,string,pattern):
    
        indices=[i.start() for i in re.finditer(pattern,string)]
        
        indices=[0]+indices+[len(string)] if 0 not in indices else indices+[len(string)]
        
        ind_range=[indices[i:i + 2] for i in range(len(indices) - 1) if i]
        
        res=[string[:indices[1]]]+[string[i[0]+1:i[1]] for i in ind_range]
        
        return res
    
    def _menu(self,i, xs_len, x_i):
    
        print('>>> Adjust breaks for ({}/{}) {}?'.format(i, xs_len, x_i))
        print('1: next \n2: yes \n3: back \n4: remove \n0: exit')
        
        adj_brk = input("Selection: ")
        
        while isinstance(adj_brk,str):
            
            if str(adj_brk).isdigit():
                
                adj_brk = int(adj_brk)
                
                if adj_brk not in [0,1,2,3,4]:
                    
                    warnings.warn('Enter an item from the menu, or 0 to exit.')         
                    
                    adj_brk = input("Selection: ")  
            else: 
                
                warnings.warn('input 1,2,3,4,0')
                
                adj_brk = input("Selection: ") 
            
        return adj_brk
    
    
    def _is_numeric(self,strung):
        
        try:
            
            float(strung)
            
            return True
        
        except:
            
            return False
    
    def _get_breaks_adj(self,br_adj,X,y,
                        sample_weight=None,special_values=None,
                        figure_size=None):
    
        global breaks_list_adj,vtabs_dict_adj
        
        # set param
        adj_count=0
        var_sum=len(br_adj)
        var_dict=dict(zip(range(len(br_adj)),br_adj.keys()))
        adj_status=False
    
        # set output
        breaks_list_adj={}
        vtabs_dict_adj={}
        
        # colname_del
        
        colname_del=set()

        while True:
    
            # default binning and plotting using given breaks 
            if not adj_status:
    
                colname=var_dict[adj_count]
    
                breaks=br_adj[colname] 
    
            print('----Adjusting {}...----'.format(colname))
            print('Current breaks: {}...'.format(breaks))
            
            binx=varReportSinge().report(X[colname],y,breaks,sample_weight=sample_weight,
                                            special_values=special_values) 
    
            fig,_=self._get_plot_single(binx,figure_size=None,show_plot=True)
    
            fig
            plt.show()
    
            # interactive options
            option=self._menu(adj_count+1,var_sum,colname)
    
            #opt==1:no adjustion,go next variable
            if option==1:
    
                adj_count+=1
    
                breaks_list_adj[colname]=breaks
                vtabs_dict_adj[colname]=binx
    
                adj_status=False
                
                print('----Adjusting {} finish.----'.format(colname))
                
            #opt==2:adjusting breaks and re-binning variable 
            elif option==2:   
    
                if is_numeric_dtype(X[colname]):
                    
                    breaks = input(">>> Enter modified breaks: ")
                
                    breaks = re.sub("^[,\.]+|[,\.]+$|\s", "", breaks).split(',')  
                    
    
                    while True:
    
                        if breaks==['']:
                            
                            breaks=binTree(n_jobs=1,coerce_monotonic=True).fit(X[[colname]],y).breaks_list[colname]
    
                            break
    
                        elif all([self._is_numeric(i) for i in breaks]):
                            
                            breaks=sorted(np.float64(breaks).tolist())
    
                            break
    
                        else:
    
                            warnings.warn('Breaks could not be converted to number.')
    
                            breaks = input(">>> Enter modified breaks: ")
    
                            breaks = re.sub("^[,\.]+|[,\.]+$|\s", "", breaks).split(',')  
                                
    
                elif is_string_dtype(X[colname]):
                    
                    breaks = input(">>> Enter modified breaks: ")
                
                    breaks = re.sub("^[,\.]+|[,\.]+$|\s", "", breaks)
    
                    
                    if not breaks:
                        
                        breaks=binTree(n_jobs=1,coerce_monotonic=True).fit(X[[colname]],y).breaks_list[colname]
                        
                    else:
    
                        breaks = self._split_by_re(breaks,'[,][^%]')
    
                else:
    
                    raise ValueError("{}'s dtype in ('number' or 'object')".format(colname))
    
    
                adj_status=True
    
            #opt==3:roll back to previous variable     
            elif option==3:
    
                adj_count+=-1 if adj_count else adj_count
                
                print('Roll back to previous variable.')
    
                adj_status=False     
                
            #opt==4:remove current variable and go next  
            elif option==4:
    
                adj_count+=1
                
                colname_del.add(colname)
                
                print('variable {} will be removed and go next.'.format(colname))
    
                adj_status=False          
            
            
            #opt==0:stop adjustion by user     
            elif option==0:
                
                print('Adjustion has not been completed yet,are you sure?')
                
                adj_status=False
                
                if_exit = input("Input 'y' to exit or other to continue :")
                
                # stop condition (1/2):user defined
                if if_exit=='y':
                    
                    print('Stop adjusting...,result store in global variables "breaks_list_adj" and "vtabs_dict_adj"')
                    
                    break
                    
            else:
                    
                raise ValueError('option not in (0,1,2,3,4)')
                     
            # stop condition (2/2):all variables done
            if adj_count==var_sum:
    
                print('Adjustion complete...')
    
                break 
            
        if colname_del:
            
            for key in colname_del:   
                
                if key in breaks_list_adj:             
                    
                    del breaks_list_adj[key] 
                    
                if key in vtabs_dict_adj:            
                    
                    del vtabs_dict_adj[key] 
                
        return breaks_list_adj,vtabs_dict_adj


    def _get_breaks_adj_g(self,br_adj,X,y,column,sort_column=None,psi_base='all',
                    sample_weight=None,special_values=None,
                    figure_size=None):

        global breaks_list_adj,vtabs_dict_adj
        
        # set param
        adj_count=0
        var_sum=len(br_adj)
        var_dict=dict(zip(range(len(br_adj)),br_adj.keys()))
        adj_status=False
    
        # set output
        breaks_list_adj={}
        vtabs_dict_adj={}
        
        colname_del=set()
    
        while True:
    
            # default binning and plotting using given breaks 
            if not adj_status:
    
                colname=var_dict[adj_count]
    
                breaks=br_adj[colname] 
    
            print('----Adjusting {}...----'.format(colname))
            print('Current breaks: {}...'.format(breaks))
            print(breaks)
            
            bins=varGroupsReport({colname:breaks},target=y.name,
                                      columns=[column],
                                      sort_columns={column:sort_column} if sort_column else sort_column,
                                      output_psi=True,
                                      psi_base=psi_base,
                                      sample_weight=sample_weight,
                                      row_limit=0,n_jobs=1).fit(X[[colname]+[column]].join(y))            
            
            binx_psi=bins.report_dict['report_psi']
            
            psi_col=sort_column if sort_column else X[column].astype('str').unique()
            
            psi_info=[(i,round(binx_psi.loc[binx_psi.bin=='psi'][i]['count_distr'].values[0],4)) for i in psi_col]
            
            binx_g=pd.concat({col:bins.report_dict_raw[col] for col in psi_col},axis=1).droplevel(0)
            
            print('PSI at current breaks:{}'.format(psi_info))
            
            fig,_=self._get_plot_single_group(binx_g,
                                              sort_column=sort_column,
                                              figure_size=figure_size,
                                              show_plot=True)
            fig                                
            plt.show()
    
            # interactive options
            option=self._menu(adj_count+1,var_sum,colname)
    
            #opt==1:no adjustion,go next variable
            if option==1:
    
                adj_count+=1
    
                breaks_list_adj[colname]=breaks
                vtabs_dict_adj[colname]=binx_g
    
                adj_status=False
                
                print('----Adjusting {} finish.----'.format(colname))
                
            #opt==2:adjusting breaks and re-binning variable 
            elif option==2:   
    
                if is_numeric_dtype(X[colname]):
                    
                    breaks = input(">>> Enter modified breaks: ")
                
                    breaks = re.sub("^[,\.]+|[,\.]+$|\s", "", breaks).split(',')  
    
                    while True:
    
                        if breaks==['']:
                            
                            breaks=binTree(n_jobs=1,coerce_monotonic=True).fit(X[[colname]],y).breaks_list[colname]
    
                            break
    
                        elif all([self._is_numeric(i) for i in breaks]):
                            
                            breaks = sorted(np.float64(breaks).tolist())
    
                            break
    
                        else:
    
                            warnings.warn('Breaks could not be converted to number.')
    
                            breaks = input(">>> Enter modified breaks: ")
    
                            breaks = re.sub("^[,\.]+|[,\.]+$|\s", "", breaks).split(',')
                                
    
                elif is_string_dtype(X[colname]):
                    
                    breaks = input(">>> Enter modified breaks: ")
                
                    breaks = re.sub("^[,\.]+|[,\.]+$|\s", "", breaks)
    
                    
                    if not breaks:
                        
                        breaks = binTree(n_jobs=1,coerce_monotonic=True).fit(X[[colname]],y).breaks_list[colname]
                        
                    else:
    
                        breaks = self._split_by_re(breaks,'[,][^%]')
    
                else:
    
                    raise ValueError("{}'s dtype in ('number' or 'object')".format(colname))
    
    
                adj_status=True
    
            #opt==3:roll back to previous variable     
            elif option==3:
    
                adj_count+=-1 if adj_count else adj_count
                
                print('Roll back to previous variable.')
    
                adj_status=False
                
            #opt==4:remove current variable and go next  
            elif option==4:
    
                adj_count+=1
                
                colname_del.add(colname)
                
                print('variable {} will be removed and go next.'.format(colname))
    
                adj_status=False   
            
            #opt==0:stop adjustion by user     
            elif option==0:
                
                print('Adjustion has not been completed yet,are you sure?')
                
                adj_status=False
                
                if_exit = input("Input 'y' to exit or other to continue :")
                
                # stop condition (1/2):user defined
                if if_exit=='y':
                    
                    print('Stop adjusting...,result store in global variables "breaks_list_adj" and "vtabs_dict_adj"')
                    
                    break
                
            else:
                    
                raise ValueError('option not in (0,1,2,3,4)')
                     
            # stop condition (2/2):all variables done
            if adj_count==var_sum:
    
                print('Adjustion complete...')
    
                break 
            
        if colname_del:
            
            for key in colname_del:   
                
                if key in breaks_list_adj:             
                    
                    del breaks_list_adj[key] 
                    
                if key in vtabs_dict_adj:            
                    
                    del vtabs_dict_adj[key] 
                
        return breaks_list_adj,vtabs_dict_adj
    
    
