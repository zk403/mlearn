#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 21:10:13 2021

@author: zengke
"""
import pandas as pd
from sklearn.base import TransformerMixin
import numpy as np
#import time
from pandas.api.types import is_string_dtype,is_array_like,is_numeric_dtype
from glob import glob
import os
from itertools import product
import warnings
from joblib import Parallel,delayed,effective_n_jobs
from BDMLtools.base import Base
from BDMLtools.fun import raw_to_bin_sc,Specials
from BDMLtools.plotter.base import BaseWoePlotter


class EDAReport(Base,Specials):
    
    """ 
    产生数据质量报告
    Params:
    ------
        missing_values:list or dict,缺失值指代值,
            + None,保持数据默认
            + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换为np.nan
            + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换为np.nan
        is_nacorr:bool,是否输出缺失率相关性报告
        out_path:str or None,将数据质量报告输出到本地工作目录的str文件夹下，None代表不输出            
    
    Attributes:
    -------
        num_report:pd.DataFrame,连续特征质量报告
        char_report:pd.DataFrame,分类特征质量报告
        na_report:pd.DataFrame,单特征缺失率报告
        nacorr_report:pd.DataFrame,缺失率相关性报告
    """
    
    def __init__(self,missing_values=[np.nan, np.inf, -np.inf,'nan','','special','missing'],is_nacorr=False,out_path=None):
        
        self.missing_values=missing_values
        self.is_nacorr=is_nacorr
        self.out_path=out_path
        
    def report(self, X, y=None):

        self._check_X(X)
        
        if X.select_dtypes(exclude=['number','object','category']).shape[1]>0:
            
            warnings.warn("column which not in ['number','object','category'] will not be shown in report")

        X=self._sp_replace(X,self.missing_values,fill_num=np.nan,fill_str=np.nan)

        #产生报告
        self.num_report=self._num_info(X)
        self.char_report=self._char_info(X)
        self.na_report=self._nan_info(X)    
        
        if self.is_nacorr:
            
            self.nacorr_report=self._nan_corr(X)
        
        #输出报告    
        if self.out_path: 
            
            self._writeExcel()                
                                    
        return self 
    
    def _num_info(self,X):
        
        """ 数据质量报告-数值特征
        """
        
        num_col=X.select_dtypes(include='number').columns.tolist()
            
        if len(num_col):

            report=X[num_col].describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).T.assign(
                MissingRate=X.apply(   
                    lambda col:col.isnull().sum()/col.size    
                   )
            ).reset_index().rename(columns={'index':'VarName'})
            
            return report
        
        else:
            
            return None
    

    def _char_info(self,X):
        """ 数据质量报告-分类特征
        """
        
        category_col=X.select_dtypes(include=['object','category']).columns.tolist()
            
        if len(category_col):    
    
            report={}
            for col in category_col:
    
                ColTable=X[col].value_counts(dropna=False).sort_index().rename('Freq').reset_index() \
                    .rename(columns={'index':'Levels'})[['Levels','Freq']]
    
                ColTable['Percent']=ColTable.Freq/X[col].size #占比
                ColTable['CumFreq']=ColTable.Freq.cumsum() #累计(分类特征类别有次序性时有参考价值)
                ColTable['CumPercent']=ColTable.CumFreq/X[col].size #累计占比(分类特征类别有次序性时有参考价值)
    
                report[col]=ColTable
            
            return pd.concat(report).droplevel(1).reset_index().rename(columns={'index':'Name'})
        
        else:
            
            return None
    
   
    def _nan_info(self,X):
        """ 数据质量报告-缺失特征
        """        
                   
        X=X.select_dtypes(include=['object','category','number']).copy()
 
        report=pd.DataFrame(
        {'N':X.apply(   
            lambda col:col.size      
           ),
        'Missings':X.apply(   
            lambda col:col.isnull().sum()   
           ),
        'MissingRate':X.apply(   
            lambda col:col.isnull().sum()/col.size    
               ),
        'dtype':X.dtypes
            })
    
        return report
    
  
    def _nan_corr(self,X):
        """ 数据质量报告-缺失特征相关性
        """        
        
        X=X.select_dtypes(include=['object','category','number']).copy()

        nan_info=X.isnull().sum()
        
        nan_corr_table=X[nan_info[nan_info>0].index].isnull().corr()
        
        return nan_corr_table
    
    
    def _writeExcel(self):
        
        if not glob(self.out_path):
            
            os.mkdir(self.out_path)
                
        if not glob(self.out_path+"/model_report.xlsx"):
            
            #print(self.out_path+"/model_report.xlsx")
            
            writer = pd.ExcelWriter(self.out_path+"/model_report.xlsx")       
            pd.DataFrame(None).to_excel(writer,sheet_name='summary')
            writer.save()                    
            
        writer=pd.ExcelWriter(self.out_path+'/model_report.xlsx',
                              mode='a',
                              if_sheet_exists='replace',
                              #engine_kwargs={'mode':'a','if_sheet_exists':'replace'},
                              engine='openpyxl')
        
        if self.num_report is not None:
            
            self.num_report.to_excel(writer,sheet_name='1.EDA_num')
        
        if self.char_report is not None:
            
            self.char_report.to_excel(writer,sheet_name='1.EDA_char') 
            
        if self.na_report is not None:
            
            self.na_report.to_excel(writer,sheet_name='1.EDA_nan')
            
        if self.is_nacorr and self.nacorr_report is not None:
            
            self.nacorr_report.to_excel(writer,sheet_name='1.EDA_nancorr')
            
        writer.save()     
        print('to_excel done')  

        
class businessReport(Base,TransformerMixin):
    
    """ 
    产生业务报告
    Params:
    ------
        target:str,目标变量名
        index:list,汇总到行的列名
        columns:list,汇总到列的列名
        rename_columns:list,重命名汇总到行的列名
        rename_index:list,重命名汇总到列的列名     
        out_path:将报告输出到本地工作目录的str文件夹下，None代表不输出 
    
    Attributes:
    -------
        ptable:pd.DataFrame,业务透视表
    """    
    
    def __init__(self,target,index,columns,rename_columns=None,rename_index=None,out_path=None):

        self.target = target
        self.index = index
        self.columns=columns
        self.rename_columns=rename_columns
        self.rename_index=rename_index
        self.out_path=out_path
        
    def report(self, X,y=None):
        
        self._check_X(X)
            
        values=self.target
        index=self.index
        columns=self.columns
        rename_columns=self.rename_columns
        rename_index=self.rename_index
        
        aggfunc=['count','sum','mean']
        rename_aggfunc=dict(zip(aggfunc,['#','event#','event%#']))

        ptable=pd.pivot_table(X,index=index,columns=columns,
                          values=values,aggfunc=aggfunc,
                          margins=True).rename(columns=rename_aggfunc,level=0)

        if rename_index:
            ptable.index.names=rename_index

        if rename_columns:
            ptable.columns.names=[None]+rename_columns
            
        
        if self.out_path:
            
            self._writeExcel(ptable)                
                                    
        return ptable
        
    
    def _writeExcel(self,ptable):
        
        if not glob(self.out_path):
            
            os.mkdir(self.out_path)
                
        if not glob(self.out_path+"/model_report.xlsx"):
            
            writer = pd.ExcelWriter('/model_report.xlsx')            
            pd.DataFrame(None).to_excel(writer,sheet_name='summary')
            writer.save()                    
            
        writer=pd.ExcelWriter(self.out_path+'/model_report.xlsx',
                              mode='a',
                              if_sheet_exists='replace',
                              #engine_kwargs={'mode':'a','if_sheet_exists':'replace'},
                              engine='openpyxl')

        ptable.to_excel(writer,sheet_name='2.Businessreport')
            
        writer.save()     
        print('to_excel done') 
        

class varReportSinge(Base,Specials,BaseWoePlotter):
    
    def report(self, X, y,breaks,sample_weight=None,special_values=None,regularization=1e-10):
        """ 
        分箱并产生特征分析报告、特征分析绘图报告  
        Params:
        ------
        X:pd.Series,单特征x
        y:pd.Series,目标特征
        breaks:list,分箱点
        regularization=1e-10:float,计算好坏分布时正则项强度,用于应对iv、woe计算时分母为0的情况，越高则强度越高，不同的regularization将影响iv、ks、woe的结算结果。
        sample_weight:pd.Seires or None,样本权重
        special_values:特殊值指代值,若数据中某些值或某列某些值需特殊对待(这些值不是np.nan)时设定
            请特别注意若使用binSelector产生breaks_list,special_values必须与binSelector的special_values一致,否则报告的special行会产生错误结果
            + None,保证数据默认
            + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan          
        Return:
        ------
        report_var,pd.DataFrame:单特征分析报告       
        """       
        
        self._check_x(X)
        self._check_yname(y)
        breaks=self._check_breaks(breaks)
                
        X=self._sp_replace_single(X,self._check_spvalues(X.name,special_values),fill_num=np.finfo(np.float32).max,fill_str='special')
               
        report_var=self.getReport_Single(X,y,breaks,sample_weight,special_values,regularization)
        
        return report_var 
    
    
    def woe_plot(self,X, y,breaks,sample_weight=None,special_values=None,figure_size=None,show_plot=True):        
        """ 
        分箱并产生特征分析报告、特征分析绘图报告  
        Params:
        ------
        X:pd.Series,单特征x
        y:pd.Series,目标特征
        breaks:list,分箱点
        sample_weight:pd.Seires or None,样本权重
        special_values:特殊值指代值,若数据中某些值或某列某些值需特殊对待(这些值不是np.nan)时设定
            请特别注意若使用binSelector产生breaks_list,special_values必须与binSelector的special_values一致,否则报告的special行会产生错误结果
            + None,保证数据默认
            + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan          
        figure_size=None:matplotlib的画布大小
        show_plot=True:程序运行后打印绘图报告结果
        
        Return:
        ------
        report_var,pd.DataFrame:单特征分析报告
        figure,matplotlib.figure.Figure:单特征分析报告-绘图
        breaks,list:x的分箱点
            
        """ 
        breaks=self._check_breaks(breaks)
        
        report_var=self.report(X,y,breaks,sample_weight,special_values)
        
        figure=self. _get_plot_single(report_var,figure_size,show_plot)
    
        return report_var,figure,breaks
   
        
    def getReport_Single(self,X,y,breakslist_var,sample_weight,special_values,regularization):         

         col=X.name

         #处理缺失值
         var_fillna=X
         
         #breaklist_var=list(breaks_list_dict[col])
         
         #判断数据类型
         if is_numeric_dtype(var_fillna):
           
             
             if special_values:
                 
                 
                 breaks=breakslist_var+[np.finfo(np.float32).max]
                 
                 #按照分箱sc的breaklist的区间进行分箱
                 var_cut=pd.cut(var_fillna,[-np.inf]+breaks+[np.inf],duplicates='drop',right=False).cat.add_categories('missing')                 
                 
                 #add missing codes
                 var_bin=var_cut.fillna('missing')
             
                 #add speical codes
                 var_bin=var_bin.cat.rename_categories(
                                {pd.Interval(left=np.finfo(np.float32).max, right=np.inf,closed='left'):'special'}
                            )             
                 
                 var_bin=var_bin.cat.rename_categories({
                     var_bin.cat.categories[-3]:
                     pd.Interval(left=var_bin.cat.categories[-3].left, right=np.inf,closed='left')
                 })             
                     
             else:
                 
                 breaks=breakslist_var
                 
                 var_cut=pd.cut(var_fillna,[-np.inf]+breaks+[np.inf],duplicates='drop',right=False).cat.add_categories(['special','missing'])                 
                 
                 #add missing codes
                 var_bin=var_cut.fillna('missing')

         
         elif is_string_dtype(var_fillna):    
             
             var_cut=pd.Series(np.where(var_fillna.isnull(),'missing',var_fillna),
                       index=var_fillna.index,
                       name=var_fillna.name)

             #转换字原始符映射到分箱sc的breaklist的字符映射
             var_code_raw=var_cut.unique().tolist()
                          
             map_codes=raw_to_bin_sc(var_code_raw,breakslist_var)
                        
             var_bin=pd.Series(pd.Categorical(var_cut.map(map_codes)),index=var_fillna.index,name=col).cat.set_categories(breakslist_var)
             
             
             if "special" not in var_bin.cat.categories:
                
                var_bin= var_bin.cat.add_categories('special')
             
             
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
         rename_aggfunc=dict(zip(['sample_weight',y.name],['count','bad']))
         result=pd.pivot_table(var_bin,index=col,values=[y.name,'sample_weight'],
                           margins=False,
                           aggfunc='sum').rename(columns=rename_aggfunc,level=0)#.droplevel(level=1,axis=1) 

         var_tab=self._getVarReport_ks(result,col,regularization) 
         
         if is_string_dtype(var_fillna):
             
             var_tab.index=var_tab.index.astype('category') 

         return  var_tab.assign(
             
             breaks=var_tab.index.categories.map(lambda x:x if isinstance(x,str) else x.right))
            
    def _getVarReport_ks(self,var_ptable,col,regularization=1e-10):
        
        var_ptable['badprob']=var_ptable['bad'].div(var_ptable['count'])
        var_ptable['count_distr']=var_ptable['count'].div(var_ptable['count'].sum())
        var_ptable['good']=var_ptable['count'].sub(var_ptable['bad'])
        var_ptable['good_dis']=var_ptable['good'].add(regularization).div(var_ptable['good'].sum()+2*regularization)
        var_ptable['bad_dis']=var_ptable['bad'].add(regularization).div(var_ptable['bad'].sum()+2*regularization)
        var_ptable['bin_iv']=var_ptable['bad_dis'].sub(var_ptable['good_dis']).mul(
            (var_ptable["bad_dis"]).div((var_ptable["good_dis"])).apply(np.log)
        )
        var_ptable['total_iv']=var_ptable['bin_iv'].sum()
        var_ptable['woe']=(var_ptable["bad_dis"]).div((var_ptable["good_dis"])).apply(np.log)
        var_ptable['ks']=var_ptable['good_dis'].cumsum().sub(var_ptable['bad_dis'].cumsum()).abs()
        var_ptable['ks_max']=var_ptable['ks'].max()
        var_ptable['lift']=var_ptable['badprob'].div(var_ptable['bad'].sum()/var_ptable['count'].sum())
        var_ptable['variable']=col
        var_ptable.index.name='bin'
        #var_ptable=var_ptable.reset_index()
        #var_ptable['bin']=var_ptable['bin'].astype('str')
        var_ptable=var_ptable[['variable', 'count', 'count_distr', 'good', 'bad', 'badprob','woe', 'bin_iv', 'total_iv','ks','ks_max','lift']].copy()
        
        if var_ptable.loc['special','count'] == 0:
            
            var_ptable.loc['special','woe'] = 0
            
        if var_ptable.loc['missing','count'] == 0:
            
            var_ptable.loc['missing','woe'] = 0
        
        return var_ptable
    
       
        
class varReport(Base,TransformerMixin,BaseWoePlotter):
    
    """ 
    产生业务报告
    Params:
    ------
        breaks_list_dict:dict,分箱字典结构,{var_name:[bin],...}
        special_values:特殊值指代值,若数据中某些值或某列某些值需特殊对待(这些值不是np.nan)时设定
            请特别注意,special_values必须与binSelector的special_values一致,否则报告的special行会产生错误结果
            + None,保证数据默认
            + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
            + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan  
        sample_weight:numpy.array or pd.Series or None,样本权重，若数据是经过抽样获取的，则可加入样本权重以计算加权的badrate,woe,iv,ks等指标
        regularization=1e-10:float,计算好坏分布时正则项强度,用于应对iv、woe计算时分母为0的情况，越高则正则强度越高，不同的regularization将影响iv、ks、woe的结算结果
        out_path:将报告输出到本地工作目录的str文件夹下，None代表不输出 
        tab_suffix:本地excel报告名后缀
        n_jobs:int,并行计算job数
        verbose:int,并行计算信息输出等级
    
    Attributes:
    -------
        var_report_dict:dict,特征分析报告
        
    Method:
    -------
        fit:产生特征报告
        transform:返回原始数据
        woe_plot:返回各个特征的woe_plot        
        
    """
    
    def __init__(self,breaks_list_dict,special_values=None,sample_weight=None,out_path=None,tab_suffix='',regularization=1e-10,n_jobs=-1,verbose=0):

        self.breaks_list_dict = breaks_list_dict
        self.special_values = special_values
        self.sample_weight = sample_weight
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.out_path = out_path
        self.tab_suffix = tab_suffix
        self.regularization=regularization
        
        self._is_fitted=False
        
    def fit(self, X, y):

        self._check_data(X,y)
        self._check_yname(y)
        self.breaks_list_dict=self._check_breaks(self.breaks_list_dict)
        
        n_jobs=effective_n_jobs(self.n_jobs)
        
        parallel=Parallel(n_jobs=n_jobs,verbose=self.verbose,batch_size=100)
        
        out_list=parallel(delayed(self._get_report_single)(X,y,col,self.breaks_list_dict[col],self.sample_weight,self.special_values,self.regularization)
                          for col in self.breaks_list_dict)
        
        self.var_report_dict={col:total for col,total in out_list}
        
        #输出报告    
        if self.out_path: 
            
            self._writeExcel()
            
        self._is_fitted=True
                    
        return self
    
    def transform(self, X):     
   
        return X
  
    
    def woe_plot(self,figure_size=None,n_jobs=-1,verbose=0):
        
        """ 
        根据特征分析报告批量绘图并输出        
        Params:
        ------
        figure_size=None:matplotlib的画布大小
        n_jobs=-1:joblib并行任务数量
        verbose=0:joblib信息打印等级
        
        Return:
        ------
        fig_out:dict,{特征名:绘图报告figure}
            
        """
        
        self._check_is_fitted()
        
        fig_out=self._woe_plot(self.var_report_dict,figure_size,n_jobs,verbose)
        
        return fig_out
        
        
    def _get_report_single(self,X,y,col_name,breaks,sample_weight,special_values,regularization):
           
        vtabs=varReportSinge().report(X[col_name],y,breaks,sample_weight,special_values,regularization)
        
        return col_name,vtabs

    
    def _writeExcel(self):
        
        if not glob(self.out_path):
            
            os.mkdir(self.out_path)
                
        if not glob(self.out_path+"/var_report"+self.tab_suffix+".xlsx"):
            
            writer = pd.ExcelWriter(self.out_path+'/var_report'+self.tab_suffix+'.xlsx')            
            pd.DataFrame(None).to_excel(writer,sheet_name='summary')
            writer.save()    
            
        writer=pd.ExcelWriter(self.out_path+'/var_report'+self.tab_suffix+'.xlsx',
                              mode='a',
                              if_sheet_exists='replace',
                              #engine_kwargs={'mode':'a','if_sheet_exists':'replace'},
                              engine='openpyxl')
           
        pd.concat(self.var_report_dict).to_excel(writer,sheet_name='bin')                               
        writer.save()     
        
        print('to_excel done') 

        
        
class varGroupsReport(Base,TransformerMixin,BaseWoePlotter):
    
    """ 
    产生组业务报告
    Params:
    ------
    
    breaks_list_dict:dict,分箱字典结构,{var_name:[bin],...}
    columns:list,组变量名,最终报告将组变量置于报告列上,组特征可以在breaks_list_dict中
    sort_columns:dict or None,组变量名输出的列顺序,排序后的报告中多重列索引的列顺序将与设定一致
        + sort_columns必须是dict格式，例如sort_columns={col_name1:[value1,value2,...],col_name2:[...],...}
        + sort_columns的key必须与columns一致，即排序的组变量名要写全，例如columns=['col1','col2'],那么sort_columns={col1:[value1,value2,...],col2:[...]}
            + 可使用X[key].astype('str').unique()去查看组变量的所有值并根据其值进行排序
        + sort_columns中某列的排序值[value1,value2,...]必须与原始数据X的中改列的唯一值一致,即set([value1,value2,...])==set(X[col_name1].unique)
    target:目标变量名
    output_psi:bool,是否输出群组psi报告
    psi_base:str,psi计算的基准,可选all，也可用户自定义
        + 'all':以特征在全量数据的分布为基准
        + user-define:用户输入支持X.query的表达式以确定base           
    row_limit,int,分组行数限制，建议设定该参数至合理水平
        + 默认每组最少1000行，小于限制的组不统计其任何指标，返回空，
        + 当数据中存在组样本过少时，分组进行统计的bin可能会在某些分段内缺失导致concat时出现index overlapping错误，此时可适当提高row_limit以避免此类错误
    special_values:特殊值指代值
            + None,保持数据默认
            + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
            + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan  
    regularization=1e-10:float,计算好坏分布时正则项强度,用于应对iv、woe计算时分母为0的情况，越高则正则强度越高，不同的regularization将影响iv、ks、woe的结算结果
    sample_weight:numpy.array or pd.Series(...,index=X.index) or None,样本权重，若数据是经过抽样获取的，则可加入样本权重以计算加权的badrate,woe,iv,ks等指标以还原抽样对分析影响
    n_jobs:int,并行计算job数
    verbose:int,并行计算信息输出等级
    out_path:将报告输出到本地工作目录的str文件夹下，None代表不输出 
    tab_suffix:本地excel报告名后缀
    
    Attributes:
    -------
        report_dict:dict,所有产生的报告
        
    """        
    
    def __init__(self,breaks_list_dict,columns,sort_columns=None,target='target',row_limit=0,output_psi=False,psi_base='all',
                 special_values=None,sample_weight=None,regularization=1e-10,n_jobs=-1,verbose=0,out_path=None,tab_suffix='_group'):

        self.breaks_list_dict=breaks_list_dict
        self.target=target
        self.columns=columns
        self.sort_columns=sort_columns      
        self.row_limit=row_limit
        self.output_psi=output_psi
        self.psi_base=psi_base
        self.special_values=special_values
        self.sample_weight=sample_weight
        self.regularization=regularization
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.out_path=out_path
        self.tab_suffix=tab_suffix
        
        self._is_fitted=False
       
    def fit(self, X, y=None):               
        
        self._check_X(X)
        
        if self.row_limit>0 and len(X)<=self.row_limit:
            
            raise ValueError('row_limit should less than len(X)')
            
        self.breaks_list_dict=self._check_breaks(self.breaks_list_dict)
     
        self.breaks_list_dict={key:self.breaks_list_dict[key] for key in self.breaks_list_dict if key in X.drop(self.columns,axis=1).columns}   
        
        
        if len(self.breaks_list_dict)==0:
            
            raise ValueError("variables‘ names in breaks_list not in X.columns")
        
        X=pd.concat([X.drop(self.columns,axis=1),X[self.columns].astype('str')],axis=1)
                             
        
        if is_array_like(self.sample_weight):
            
            X['sample_weight']=self.sample_weight
            
        else:
            
            X['sample_weight']=1
        
        X_g_gen=X.groupby(self.columns)
        
        n_jobs=effective_n_jobs(self.n_jobs)
        
        parallel=Parallel(n_jobs=n_jobs,verbose=self.verbose)
        
        out_list=parallel(delayed(self._group_parallel)(X_g_gen,g,self.target,self.columns,
                                                       self.breaks_list_dict,self.row_limit,
                                                       self.special_values,
                                                       self.regularization) for g in X_g_gen.groups)
        
        self.report_dict_raw={columns:vtabs for columns,vtabs in out_list}
        
        if all(np.array([len(self.report_dict_raw[key]) for key in self.report_dict_raw])==0):
            
            raise ValueError('row_limit too high to get group report,reset it and try again')

        report=pd.concat(self.report_dict_raw,axis=1)
        
        report.index.rename([None, 'bin'],inplace=True)

        if self.sort_columns:       
            
            sort_columns_list=self._check_columns_sort(self.sort_columns, self.columns, X)                                    
    
            report=self._vtab_column_sort(sort_columns_list,report)                
                  
        self.report_dict=self._getReport(X,report,self.breaks_list_dict,self.special_values,self.n_jobs,self.verbose,
                                        self.target,self.regularization,self.output_psi,self.psi_base)                    

        if self.out_path:
                
            self._writeExcel()  
            
        self._is_fitted=True                  
                                         
        return self
    
    def transform(self, X):     
     
        return X

    def _group_parallel(self,X_g_gen,g,target,columns,breaks_list_dict,row_limit,special_values,regularization):
    
        group_dt=X_g_gen.get_group(g)
        X_g=group_dt.drop([target]+columns,axis=1)
        y_g=group_dt[target]      
        w_g=group_dt['sample_weight']      
        
        if len(X_g)==0:
        
            warnings.warn('group '+str(g)+' has 0 row,output will return None')         
        
            result=pd.DataFrame(None)                                          
        
        elif len(X_g)<=row_limit:
        
            warnings.warn('group '+str(g)+' has rows less than '+str(row_limit)+',output will return None')         
        
            result=pd.DataFrame(None)  
        
        else:
        
            res=varReport(breaks_list_dict=breaks_list_dict,
                          special_values=special_values,
                          sample_weight=w_g,
                          regularization=regularization,
                          n_jobs=1,                                  
                          verbose=0).fit(X_g,y_g)
        
            result=pd.concat(res.var_report_dict)
            
        return g,result
    
    
   
    def _getReport(self,X,report,breaks_list_dict,special_values,n_jobs,verbose,target,regularization,
                  output_psi=True,psi_base='all'):
        
        report_out={}
            
        report_out['report_all']=report[[i for i in report.columns.tolist() if i[-1] not in \
                                      ['variable']]].reset_index().rename(columns={'level_0':'variable'})
                
        report_out['report_brief']=report[[i for i in report.columns.tolist() if i[-1] in \
                                      ['count','badprob','total_iv','ks_max','lift']]].reset_index().rename(columns={'level_0':'variable'})   
                            
        report_out['report_count']=report[[i for i in report.columns.tolist() if i[-1] in \
                                      ['count']]].reset_index().rename(columns={'level_0':'variable'})  
                
        report_out['report_badprob']=report[[i for i in report.columns.tolist() if i[-1] in \
                                        ['badprob']]].reset_index().rename(columns={'level_0':'variable'}) 
            
        report_out['report_lift']=report[[i for i in report.columns.tolist() if i[-1] in \
                                            ['lift']]].reset_index().rename(columns={'level_0':'variable'})      
                
        report_out['report_iv']=report[[i for i in report.columns.tolist() if i[-1] in \
                                   ['total_iv']]].droplevel(level=1).reset_index().drop_duplicates().rename(columns={'index':'variable'})  
                
        report_out['report_ks']=report[[i for i in report.columns.tolist() if i[-1] in \
                                   ['ks_max']]].droplevel(level=1).reset_index().drop_duplicates().rename(columns={'index':'variable'}) 
        
        if output_psi:
            
            if psi_base=='all':
                
                all_var=varReport(breaks_list_dict=breaks_list_dict,
                                  special_values=special_values,
                                  regularization=regularization,
                                  n_jobs=n_jobs,                                  
                                  verbose=verbose).fit(X.drop(target,axis=1),X[target])
                base=pd.concat(all_var.var_report_dict)['count_distr']
            
                report_distr=report[[i for i in report.columns.tolist() if i[-1] in ['count_distr']]]
                
                
                psi_sum=report_distr.fillna(0).apply(lambda x:self._psi(x,base),axis=0).droplevel(level=1)\
                                      .assign(bin='psi').set_index('bin',append=True).sort_index(axis=1).groupby(level=[0,1]).sum()
                                      
                psi_tab=pd.concat([report_distr,psi_sum]).sort_index()     
                
                if self.sort_columns:       
                    
                    sort_columns_list=self._check_columns_sort(self.sort_columns, self.columns, X)                                    
            
                    psi_tab=self._vtab_column_sort(sort_columns_list,psi_tab)                          
                                      
                report_out['report_psi']=psi_tab.reset_index().rename(columns={'level_0':'variable'})
            
            else:            
                
                X_q=X.query(psi_base)
                
                if not X_q.size:
                    
                    raise ValueError('X.query has 0 row, check the query expr.')
                
                all_var=varReport(breaks_list_dict=breaks_list_dict,
                                  special_values=special_values,
                                  regularization=regularization,
                                  n_jobs=n_jobs,                                  
                                  verbose=verbose).fit(X_q.drop(target,axis=1),X_q[target])
                base=pd.concat(all_var.var_report_dict)['count_distr']
            
                report_distr=report[[i for i in report.columns.tolist() if i[-1] in ['count_distr']]]
                
                psi_sum=report_distr.fillna(0).apply(lambda x:self._psi(x,base),axis=0).droplevel(level=1)\
                                      .assign(bin='psi').set_index('bin',append=True).sort_index(axis=1).groupby(level=[0,1]).sum()
                                      
                psi_tab=pd.concat([report_distr,psi_sum]).sort_index()     
                                
                if self.sort_columns:       
                                    
                    sort_columns_list=self._check_columns_sort(self.sort_columns, self.columns, X)                                    
                            
                    psi_tab=self._vtab_column_sort(sort_columns_list,psi_tab)                            
                                      
                report_out['report_psi']=psi_tab.reset_index().rename(columns={'level_0':'variable'})                                        
                
        return report_out        
            
    
    def _psi(self,base,col):
    
        base=base.replace(0,1e-10)
        col=col.replace(0,1e-10)   
        psi_out=base.sub(col).mul(base.div(col).map(np.log))

        return psi_out            
    
    def _vtab_column_sort(self,sort_columns,report):
        
        sort_columns=sort_columns.copy()
        
        vt_cols=['variable', 'count', 'count_distr', 'good', 'bad', 'badprob', 'woe',
                 'bin_iv', 'total_iv', 'ks', 'ks_max','lift', 'breaks']

        sort_columns.append(vt_cols)
        
        cols_sorted=list(product(*sort_columns))
        
        cols_sorted_c=cols_sorted.copy()
        
        for col in cols_sorted_c:
            
            if not col in report.columns.tolist():
                
                cols_sorted.remove(col)
                
        return report[cols_sorted]
    

    def _check_columns_sort(self,sort_columns,columns,X):
        
        #check format of sort_columns:must dict
        
        if isinstance(sort_columns,dict):     

            #check format of sort_columns:len must equal to self.columns
            if len(sort_columns)==len(columns):
            
                #check format of sort_columns:defined sort value must equal to unique values of sort column
                for col in sort_columns:
                    
                    col_values=X[col].astype('str').unique()
                    
                    if set(col_values) != set(sort_columns[col]):                        
                        
                        raise ValueError("sort_values of '{}' not equal to unique values of X['{}'],use X[col].astype('str').unique() to check values and re-define them".format(col,col))
                        
                #get sort_columns list      
                sort_columns=[sort_columns[col] for col in columns]
                        
                return sort_columns
                        
            else:
                
                raise ValueError("len(sort_columns) must equal to self.columns")
                   
        else:
            
            raise ValueError("sort_columns={col_name1:[value1,value2,...],col_name2:[...],...}")  
    
    
    def _writeExcel(self):
        
        if not glob(self.out_path):
            
            os.mkdir(self.out_path)
                
        if not glob(self.out_path+"/var_report"+str(self.tab_suffix)+".xlsx"):
            
            writer = pd.ExcelWriter(self.out_path+'/var_report'+str(self.tab_suffix)+'.xlsx')            
            pd.DataFrame(None).to_excel(writer,sheet_name='summary')
            writer.save()                    
            
        writer=pd.ExcelWriter(self.out_path+'/var_report'+str(self.tab_suffix)+'.xlsx',
                              mode='a',
                              if_sheet_exists='replace',
                              #engine_kwargs={'mode':'a','if_sheet_exists':'replace'},
                              engine='openpyxl')

        self.report_dict['report_all'].to_excel(writer,sheet_name='bin_all')
        self.report_dict['report_brief'].to_excel(writer,sheet_name='bin_brief')
        self.report_dict['report_count'].to_excel(writer,sheet_name='bin_count')     
        self.report_dict['report_badprob'].to_excel(writer,sheet_name='bin_badprob')     
        self.report_dict['report_iv'].to_excel(writer,sheet_name='bin_iv') 
        self.report_dict['report_ks'].to_excel(writer,sheet_name='bin_ks') 
        self.report_dict['report_lift'].to_excel(writer,sheet_name='bin_lift') 
        
        if self.output_psi:
            
            self.report_dict['report_psi'].to_excel(writer,sheet_name='bin_psi')  
        
            
        writer.save()     
        print('to_excel done') 
        
        
        
class varGroupsPlot(Base,BaseWoePlotter):
    
    
    def __init__(self,breaks_list,column,target='target',psi_base='all',sort_column=None,special_values=None,
                       sample_weight=None,n_jobs=-1,verbose=0):
        
        self.breaks_list = breaks_list
        self.column=column
        self.target = target
        self.psi_base=psi_base
        self.sort_column=sort_column
        self.special_values=special_values
        self.sample_weight=sample_weight
        self.n_jobs=n_jobs
        self.verbose=verbose
        
        
    def plot(self,X,figure_size=None):
        
        self._check_X(X)
        self.breaks_list=self._check_breaks(self.breaks_list)
        
        fig_out=self._woe_plot_group(X,self.breaks_list,self.target,self.column,self.psi_base,
                             self.sort_column,figure_size,self.special_values,self.sample_weight,
                             self.n_jobs,self.verbose)  
        
        self.fig_out=fig_out
        
        return fig_out

    
    def _woe_plot_group(self,X,breaks_list,target,column,psi_base,sort_column=None,figure_size=None,special_values=None,
                       sample_weight=None,n_jobs=-1,verbose=0):
                 
        """ 
        根据组特征分析报告批量绘图并输出          
        考虑到组绘图的美观与性能，woe_plot_group只支持一个组水平的绘图，因此varGroupsReport的参数columns长度必须为1，此外row_limit参数必须设定为0
        Params:
        ------
        X,
        y,
        
        sort_column=None or list,在绘图中排序组水平,要求组水平值必须与原始数据的组水平一致
        figure_size=None:matplotlib的画布大小
        n_jobs=-1:joblib并行任务数量
        verbose=0:joblib信息打印等级
        
        Return:
        ------
        fig_out:dict,{特征名:绘图报告figure}
            
        """
        
        self._check_param_for_plot([column])
        
        n_jobs=effective_n_jobs(n_jobs)   
                              
        p=Parallel(n_jobs=n_jobs,verbose=verbose)
        
        res=p(delayed(self._woe_plot_group_single)(X,breaks_list,colname,target,column,psi_base,sort_column,
                                                   figure_size,special_values,sample_weight) for colname in breaks_list)
        
        fig_out={colname:fig for fig,colname in res}             

        return fig_out
    
    def _check_param_for_plot(self,columns):
        
        # if row_limit != 0:
            
        #     raise ValueError('Param "row_limit" of varGroupsReport instance must set to 0 if call ".woe_plot_group"')
            
        if len(columns) != 1:
            
            raise ValueError('".woe_plot_group" only support single grouper,reset param "columns" in varGroupsReport instance')  
            
    def _woe_plot_group_single(self,X,breaks_list,colname,target,column,psi_base='all',sort_column=None,figure_size=None,
                               special_values=None,sample_weight=None):
    
        varbin_g=varGroupsReport({colname:breaks_list[colname]},
                       n_jobs=1,target=target,
                       output_psi=True,
                       columns=[column],
                       psi_base=psi_base,
                       special_values=special_values,
                       sample_weight=sample_weight,
                       row_limit=0,
                      ).fit(X[[colname]+[column]+[target]]).report_dict_raw
        
        cd_col=sort_column if sort_column else X[column].astype('str').unique()
        
        binx_g=pd.concat({col:varbin_g[col] for col in cd_col},axis=1).droplevel(0)
        
        figure,colname=self._get_plot_single_group(binx_g,sort_column=sort_column,figure_size=figure_size,show_plot=False)
        
        return figure,colname        
        
        
        
class GainsTable(Base):
    
    """ 
    根据模型评分产生适合策略分析的Gains Table
    
    Params:
    ------
        order:str,模型输出顺序假定,顺序假定会影响累积分布(cumsum)的运算结果,默认'auto',
            + 'prob',当模型输出为概率时，概率越高风险越高,反之越低,
            + 'score',当模型输出为评分时，评分越高风险越低,反之越高,
            + 'auto',当输入值范围在0-1之间时推断为'prob',当输入值范围在0-1之外时,推断为'score'
        method:str,默认'freq'为等频分箱,任意其他值为等宽分箱
        bin_num:int,分箱数,默认10
    
    Method:
    -------
        fit_report:产生Gains Table
    """  
    
    
    def __init__(self,order='auto',method='freq',bin_num=10):

        self.order=order
        self.method=method
        self.bin_num=bin_num
        
    def report(self,X,y,group=None,base_grouper=None):
        
        """ 
        根据模型评分产生适合策略分析的Gains Table

        Params:
        ------
            X:series,输入评分
            y:series,实际值(0,1)
            group:series,组变量,用于对数据分组,默认None即无组变量
            base_grouper:string list,基准水平,若组变量存在,可设定组中的基准水平范围以进行分箱,默认None即全部数据都用做分箱
            
        Output:
        -------
            gain_tab,DataFrame or dict,产生的Gains Table
        """  
        
        self._check_x(X)
        self._check_x(y)
        self._check_ind([X,y])
        
        if group is not None:
            
            self._check_x(group)
            
            self._check_ind([X,group])
        
        gain_tab=self._fit_report(X,y,group,base_grouper,self.order,self.method,self.bin_num)

        return gain_tab
       
    
    def _fit_report(self, X, y, group=None,base_grouper=None,order='auto',method='freq',bin_num=10):  
                   
        if group is not None:
            
            X_g=X.groupby(group)
            y_g=y.groupby(group)
            
            if base_grouper is not None:
            
                b_mask=group.isin(base_grouper)      
                
            else:
                
                b_mask=np.repeat(True,group.size)
            
            breaks=self._get_breaks(X[b_mask],method,bin_num)
            
            report_var={items[0]:self._get_single(items[1],items[2],breaks,order) 
                        for items in [(g,X_g.get_group(g),y_g.get_group(g)) for g in X_g.groups]}
                
        else:
            
            breaks=self._get_breaks(X,method,bin_num)
            
            report_var=self._get_single(X,y,breaks,order)
        
        return report_var 
    
    
    def _get_breaks(self,X,method,bin_num):
        
        #equal_freq bin 
        if method=='freq':
            
            breaks=np.unique(np.percentile(X,np.arange(bin_num+1)/bin_num*100)[1:-1]).tolist()
        
        #equal_width bin
        else:
            
            breaks=np.unique(np.histogram_bin_edges(X,bin_num)[1:-1]).tolist()
            
        return breaks
   
        
    def _get_single(self,X,y,breaks,order='auto'):     

        #check dtype
        if not is_numeric_dtype(X):
            
            raise ValueError('only numeric X allowed for gains table')
            
            
        var_bin=pd.cut(X,[-np.inf]+breaks+[np.inf],right=False)
            
        #check order
        if order=='auto':
            
            #infer probability 
            if np.min(X)>=0 and np.max(X)<=1:
                
                var_bin=var_bin
            
            #infer score 
            else:

                var_bin=var_bin.cat.reorder_categories(np.flip(var_bin.cat.categories))
                
                
        elif order=='prob':
                
            var_bin=var_bin
            

        elif order=='score':

            var_bin=var_bin.cat.reorder_categories(np.flip(var_bin.cat.categories))            
            
        var_bin=pd.concat([var_bin,y],axis=1)
        var_bin['sample_weight']=1
         
        #print var_bin
        rename_aggfunc=dict(zip(['sample_weight',y.name],['count','bad']))
        result=pd.pivot_table(var_bin,index=X.name,values=[y.name,'sample_weight'],
                           margins=False,
                           aggfunc='sum').rename(columns=rename_aggfunc,level=0)

        var_tab=self._get_gt(result,X.name) 
        
        return var_tab
        
            
    def _get_gt(self,var_ptable,col):
        
        var_ptable['good']=var_ptable['count'].sub(var_ptable['bad'])
        var_ptable['cum_count']=var_ptable['count'].cumsum()
        var_ptable['cum_bad']=var_ptable['bad'].cumsum()
        var_ptable['cum_good']=var_ptable['good'].cumsum()        
        var_ptable['badprob']=var_ptable['bad'].div(var_ptable['count'])
        var_ptable['count_distr']=var_ptable['count'].div(var_ptable['count'].sum())
        var_ptable['approval_rate']=var_ptable['count_distr'].cumsum()
        var_ptable['cum_posprob']=var_ptable['cum_bad'].div(var_ptable['cum_count'])        
        var_ptable['model']=col
        var_ptable.index.name='bin'

        var_ptable=var_ptable[['model','count','cum_count','good','cum_good','bad','cum_bad','count_distr','badprob','approval_rate','cum_posprob']]
        
        return var_ptable        
        