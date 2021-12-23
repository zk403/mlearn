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
import warnings
from joblib import Parallel,delayed
from BDMtools.fun import raw_to_bin_sc,sp_replace

class EDAReport(TransformerMixin):
    
    """ 
    产生数据质量报告
    Params:
    ------
        categorical_col:list,类别特征列名
        numeric_col:list,连续特征列名
        special_values:缺失值指代值
            + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
            + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan  
        is_nacorr:bool,是否输出缺失率相关性报告
        out_path:str or None,将数据质量报告输出到本地工作目录的str文件夹下，None代表不输出            
    
    Attributes:
    -------
        num_report:pd.DataFrame,连续特征质量报告a
        char_report:pd.DataFrame,分类特征质量报告
        na_report:pd.DataFrame,单特征缺失率报告
        nacorr_report:pd.DataFrame,缺失率相关性报告
    """
    
    def __init__(self,categorical_col=None,numeric_col=None,special_values=[np.nan,'nan'],is_nacorr=False,out_path="report"):
        
        self.categorical_col = categorical_col
        self.numeric_col = numeric_col
        self.special_values=special_values
        self.is_nacorr=is_nacorr
        self.out_path=out_path
        
    def fit(self, X, y=None):

        if X.size:
            
            #填充缺失值
            X=sp_replace(X,self.special_values)   
            
            #产生报告
            self.num_report=self.num_info(X)
            self.char_report=self.char_info(X)
            self.na_report=self.nan_info(X)        
            if self.is_nacorr:
                self.nacorr_report=self.nan_corr(X)
            
            #输出报告    
            if self.out_path: 
                
                self.writeExcel()                
                                    
        return self
    
    def transform(self, X):       
        
        if X.size:
            
            return X
        
        else:
            
            warnings.warn('0 rows in input X,return None')
            
            return pd.DataFrame(None)        
    

    def num_info(self,X):
        
        """ 数据质量报告-数值特征
        """
        
        if self.numeric_col is None:
            num_col=X.select_dtypes(include='number').columns
        else:
            num_col=self.num_col

        report=X[num_col].describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).T.assign(
            MissingRate=X.apply(   
                lambda col:col.isnull().sum()/col.size    
               )
        ).reset_index().rename(columns={'index':'VarName'})
        
        return report
    

    def char_info(self,X):
        """ 数据质量报告-分类特征
        """
        
        if self.categorical_col is None:
            category_col=X.select_dtypes(include=['object','category']).columns
        else:
            category_col=self.category_col
    
        report=pd.DataFrame()
        for Col in category_col:

            ColTable=X[Col].value_counts().sort_index().rename('Freq').reset_index() \
                .rename(columns={'index':'Levels'}).assign(VarName=Col)[['VarName','Levels','Freq']]

            ColTable['Percent']=ColTable.Freq/X[Col].size #占比
            ColTable['CumFreq']=ColTable.Freq.cumsum() #累计(分类特征类别有次序性时有参考价值)
            ColTable['CumPercent']=ColTable.CumFreq/X[Col].size #累计占比(分类特征类别有次序性时有参考价值)

            report=pd.concat([report,ColTable])
        
        return report
    
   
    def nan_info(self,X):
        """ 数据质量报告-缺失特征
        """        
        
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
            } ).reset_index().rename(columns={'index':'VarName'})
    
        return report
    
  
    def nan_corr(self,X):
        """ 数据质量报告-缺失特征相关性
        """        
      
        nan_info=X.isnull().sum()
        nan_corr_table=X[nan_info[nan_info>0].index].isnull().corr()
        return nan_corr_table
    
    def writeExcel(self):
        
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
    
        self.num_report.to_excel(writer,sheet_name='1.EDA_num')
        self.char_report.to_excel(writer,sheet_name='1.EDA_char')        
        self.na_report.to_excel(writer,sheet_name='1.EDA_nan')
        if self.is_nacorr:
            self.nacorr_report.to_excel(writer,sheet_name='1.EDA_nancorr')
            
        writer.save()     
        print('to_excel done')  

        
class businessReport(TransformerMixin):
    
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
        
    def fit(self, X, y=None):
        

        if X.size:
            
            values=self.target
            index=self.index
            columns=self.columns
            rename_columns=self.rename_columns
            rename_index=self.rename_index
            
            aggfunc=['count','sum','mean']
            rename_aggfunc=dict(zip(aggfunc,['#','event#','event%#']))

            self.ptable=pd.pivot_table(X,index=index,columns=columns,
                              values=values,aggfunc=aggfunc,
                              margins=True).rename(columns=rename_aggfunc,level=0)

            if rename_index:
                self.ptable.index.names=rename_index

            if rename_columns:
                self.ptable.columns.names=[None]+rename_columns
                
            
            if self.out_path:
                
                self.writeExcel()                
                                    
        return self
    
    def transform(self, X):     
        
        if X.size:
            
            return X
        
        else:
            
            warnings.warn('0 rows in input X,return None')

            return pd.DataFrame(None)
    
    
    def writeExcel(self):
        
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

        self.ptable.to_excel(writer,sheet_name='2.Businessreport')
            
        writer.save()     
        print('to_excel done') 
        


class varReportSinge:
    
    def report(self, X, y,breaks,sample_weight=None):
        
        report_var=self.getReport_Single(X,y,breaks,sample_weight)
        
        return report_var
 
        
    def getReport_Single(self,X,y,breakslist_var,sample_weight):
         

         col=X.name

         #处理缺失值
         var_fillna=X
         
         #breaklist_var=list(breaks_list_dict[col])
         
         #判断数据类型
         if is_numeric_dtype(var_fillna):
           
             #按照分箱sc的breaklist的区间进行分箱
             var_cut=pd.cut(var_fillna,[-np.inf]+breakslist_var+[np.inf],duplicates='drop',right=False).cat.add_categories('missing')
             
             var_bin=var_cut.fillna('missing')
         
         elif is_string_dtype(var_fillna):    
             
             var_cut=pd.Series(np.where(var_fillna.isnull(),'missing',var_fillna),
                       index=var_fillna.index,
                       name=var_fillna.name)

             #转换字原始符映射到分箱sc的breaklist的字符映射
             var_code_raw=var_cut.unique().tolist()
                          
             map_codes=raw_to_bin_sc(var_code_raw,breakslist_var)
                        
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
        
        

class varReport(TransformerMixin):
    
    """ 
    产生业务报告
    Params:
    ------
        breaks_list_dict:dict,分箱字典结构,{var_name:[bin],...},支持scorecardpy与toad的breaks_list结构，
        special_values:缺失值指代值
            + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
            + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan  
        sample_weight:numpy.array or pd.Series or None,样本权重，若数据是经过抽样获取的，则可加入样本权重以计算加权的badrate,woe,iv,ks等指标
        out_path:将报告输出到本地工作目录的str文件夹下，None代表不输出 
        tab_suffix:本地excel报告名后缀
        n_jobs:int,并行计算job数
        verbose:int,并行计算信息输出等级
    
    Attributes:
    -------
        var_report_dict:dict,特征分析报告字典
    """
    
    def __init__(self,breaks_list_dict,special_values=[np.nan,'nan'],sample_weight=None,out_path=None,tab_suffix='',n_jobs=-1,verbose=0):

        self.breaks_list_dict = breaks_list_dict
        self.special_values=special_values
        self.sample_weight=sample_weight
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.out_path = out_path
        self.tab_suffix=tab_suffix
        
    def fit(self, X, y=None):
        

        if X.size:
            
            X=sp_replace(X, self.special_values)
            
            self.breaks_list_dict=self.get_Breaklist_sc(self.breaks_list_dict,X,y)
            
            parallel=Parallel(n_jobs=self.n_jobs,verbose=self.verbose)
            
            out_list=parallel(delayed(self.getReport_Single)(X,y,col,self.breaks_list_dict[col]) 
                               for col in list(self.breaks_list_dict.keys())) 
            
            self.var_report_dict={col:total for col,total,_ in out_list}
            self.var_report_psi={col:psi for col,_,psi in out_list}
            
            #输出报告    
            if self.out_path: 
                
                self.writeExcel()
            
                                    
        return self
    
    def transform(self, X):     
        
        if X.size:
            
            return X
        
        else:
            
            warnings.warn('0 rows in input X,return None')

            return pd.DataFrame(None)
        
    
    def getReport_Single(self,X,y,col,breaklist_var):
         
         #print(col)
         #global psi_base_mon1,dis_mon
         
         #breaklist_var=list(breaks_list_dict[col])
         
         #第1步:判断数据类型
         if is_numeric_dtype(X[col]):
             
             var_fillna=X[col]
           
             #按照分箱sc的breaklist的区间进行分箱
             var_cut=pd.cut(var_fillna,[-np.inf]+breaklist_var+[np.inf],duplicates='drop',right=False).cat.add_categories('missing')
             
             var_bin=var_cut.fillna('missing')
         
         elif is_string_dtype(X[col]):   

             var_fillna=X[col]
             
             var_cut=pd.Series(np.where(var_fillna.isnull(),'missing',var_fillna),
                       index=var_fillna.index,
                       name=var_fillna.name)

             #转换字原始符映射到分箱sc的breaklist的字符映射
             var_code_raw=var_cut.unique().tolist()
                          
             map_codes=raw_to_bin_sc(var_code_raw,breaklist_var)
                        
             var_bin=pd.Series(pd.Categorical(var_cut.map(map_codes)),index=var_fillna.index,name=col)
             
             if 'missing' in var_bin.cat.categories:
                 
                 var_bin= var_bin.fillna('missing') 
            
             else:
                 
                 var_bin= var_bin.cat.add_categories('missing').fillna('missing')
            
         else:
             
             raise ValueError('dtypes in X in (number,object),others not support')
         
         #加权         
         if is_array_like(self.sample_weight):         
             var_bin=pd.concat([var_bin,y.mul(self.sample_weight).rename(y.name)],axis=1) 
             var_bin['sample_weight']=self.sample_weight
         else:
             var_bin=pd.concat([var_bin,y],axis=1)
             var_bin['sample_weight']=1         
         
         #print var_bin
         rename_aggfunc=dict(zip(['sample_weight',y.name],['count','bad']))
         result=pd.pivot_table(var_bin,index=col,values=[y.name,'sample_weight'],
                           margins=False,
                           aggfunc='sum').rename(columns=rename_aggfunc,level=0)#.droplevel(level=1,axis=1)          
         
         
         var_tab=self.getVarReport_ks(result,col) 
         
         if is_string_dtype(var_fillna):
             
             var_tab.index=var_tab.index.astype('category') 

         var_tab_out=var_tab.assign(
             breaks=var_tab.index.categories.map(lambda x:x if isinstance(x,str) else x.right))

         return col,var_tab_out,None   

            
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
    
    
    def psi(self,base,col):
    
        base=base.replace(0,1e-10)
        col=col.replace(0,1e-10)   
        psi_out=base.sub(col).mul(base.div(col).map(np.log))

        return psi_out
    
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

    
    
    def writeExcel(self):
        
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
        
        
class varGroupsReport(TransformerMixin):
    
    """ 
    产生组业务报告
    Params:
    ------
    
    breaks_list_dict:dict,分箱字典结构,{var_name:[bin],...},支持scorecardpy与toad的breaks_list结构，
    columns:list,组变量名,最终报告将组变量置于报告列上,组特征可以在breaks_list_dict中
    target:目标变量名
    output_psi:bool,是否输出群组psi报告
    psi_base:str,psi计算的基准,可选all，也可用户自定义
        + 'all':以特征在全量数据的分布为基准
        + user-define:用户输入支持X.query的表达式以确定base           
    row_limit,int,分组行数限制，建议设定该参数至合理水平
        + 默认每组最少1000行，小于限制的组不统计其任何指标，返回空，
        + 当数据中存在组样本过少时，分组进行统计的bin可能会在某些分段内缺失导致concat时出现index overlapping错误，此时可适当提高row_limit以避免此类错误
    special_values:list,缺失值指代值
            + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
            + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan  
    sample_weight:numpy.array or pd.Series(...,index=X.index) or None,样本权重，若数据是经过抽样获取的，则可加入样本权重以计算加权的badrate,woe,iv,ks等指标以还原抽样对分析影响
    n_jobs:int,并行计算job数
    verbose:int,并行计算信息输出等级
    out_path:将报告输出到本地工作目录的str文件夹下，None代表不输出 
    tab_suffix:本地excel报告名后缀
    
    Attributes:
    -------
        report_dict:dict,所有产生的报告
    """
        
    
    
    def __init__(self,breaks_list_dict,columns,sort_columns=None,target='target',row_limit=1000,output_psi=False,psi_base='all',
                 special_values=['nan',np.nan],sample_weight=None,
                 n_jobs=-1,verbose=0,out_path=None,tab_suffix='_group'):

        self.breaks_list_dict=breaks_list_dict
        self.target=target
        self.columns=columns
        self.sort_columns=sort_columns      
        self.row_limit=row_limit
        self.output_psi=output_psi
        self.psi_base=psi_base
        self.special_values=special_values
        self.sample_weight=sample_weight
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.out_path=out_path
        self.tab_suffix=tab_suffix
       
    def fit(self, X, y=None):               
        

        if X.size:
            
            X=X.copy()
            
            
            self.breaks_list_dict={key:self.breaks_list_dict[key] for key in self.breaks_list_dict if key in X.drop(self.columns,axis=1).columns}    

            X=pd.concat([X.drop(self.columns,axis=1),X[self.columns].astype('str')],axis=1)

            
            if self.sort_columns:
                
                X=X.drop(list(self.sort_columns.keys()),axis=1).join(
                    pd.DataFrame(                      
                    {col:pd.Categorical(X[col],categories=self.sort_columns[col],ordered=True) for col in self.sort_columns},               
                    index=X.index)
                )
                       
            result={}            
            
            if is_array_like(self.sample_weight):
                
                X['sample_weight']=self.sample_weight
                
            else:
                
                X['sample_weight']=1
            
            X_g_gen=X.groupby(self.columns)
            
            for i in X_g_gen.groups:
                
                group_dt=X_g_gen.get_group(i)
                X_g=group_dt.drop([self.target]+self.columns,axis=1)
                y_g=group_dt[self.target]    
                w_g=group_dt['sample_weight']    
                
                if len(X_g)>self.row_limit:
                    
                    res=varReport(breaks_list_dict=self.breaks_list_dict,
                                  special_values=self.special_values,
                                  sample_weight=w_g,
                                  n_jobs=self.n_jobs,                                  
                                  verbose=self.verbose).fit(X_g,y_g)
                    
                    result[i]=pd.concat(res.var_report_dict)
                    
                else:
                    
                    warnings.warn('group '+str(i)+' has rows less than '+str(self.row_limit)+',output will return None')         
                    
                    result[i]=pd.DataFrame(None)       
            
   
            report=pd.concat(result,axis=1)
                
            self.report_dict=self.getReport(X,report,output_psi=self.output_psi,psi_base=self.psi_base)                    
                
            if self.out_path:
                    
                self.writeExcel()   
                    
                                         
        return self
    
    def transform(self, X):     
        
        if X.size:
            
            return X
        
        else:
            
            warnings.warn('0 rows in input X,return None')

            return pd.DataFrame(None)
   
    def getReport(self,X,report,output_psi=True,psi_base='all'):
        
        report_out={}
            
        report_out['report_all']=report[[i for i in report.columns.tolist() if i[-1] not in \
                                      ['variable']]].reset_index().rename(columns={'level_0':'variable'})
                
        report_out['report_brief']=report[[i for i in report.columns.tolist() if i[-1] in \
                                      ['count','badprob','woe','total_iv','ks_max']]].reset_index().rename(columns={'level_0':'variable'})   
                            
        report_out['report_count']=report[[i for i in report.columns.tolist() if i[-1] in \
                                      ['count']]].reset_index().rename(columns={'level_0':'variable'})  
                
        report_out['report_badprob']=report[[i for i in report.columns.tolist() if i[-1] in \
                                        ['badprob']]].reset_index().rename(columns={'level_0':'variable'})     
                
        report_out['report_iv']=report[[i for i in report.columns.tolist() if i[-1] in \
                                   ['total_iv']]].droplevel(level=1).drop_duplicates().reset_index().rename(columns={'index':'variable'})  
                
        report_out['report_ks']=report[[i for i in report.columns.tolist() if i[-1] in \
                                   ['ks_max']]].droplevel(level=1).drop_duplicates().reset_index().rename(columns={'index':'variable'}) 
        
        if self.output_psi:
            
            if psi_base=='all':
                
                all_var=varReport(breaks_list_dict=self.breaks_list_dict,
                                  special_values=self.special_values,
                                  n_jobs=self.n_jobs,                                  
                                  verbose=self.verbose).fit(X.drop(self.target,axis=1),X[self.target])
                base=pd.concat(all_var.var_report_dict)['count_distr']
            
                report_distr=report[[i for i in report.columns.tolist() if i[-1] in ['count_distr']]]
                psi_sum=report_distr.fillna(0).apply(lambda x:self.psi(x,base),axis=0).droplevel(level=1)\
                                      .assign(bin='psi').set_index('bin',append=True).sort_index(axis=1).groupby(level=[0,1]).sum()
                                      
                report_out['report_psi']=pd.concat([report_distr,psi_sum]).sort_index().reset_index().rename(columns={'level_0':'variable'})
            
            else:            
                
                X_q=X.query(psi_base)
                
                if not X_q.size:
                    
                    raise ValueError('X.query has 0 row, check the query expr.')
                
                all_var=varReport(breaks_list_dict=self.breaks_list_dict,
                                  special_values=self.special_values,
                                  n_jobs=self.n_jobs,                                  
                                  verbose=self.verbose).fit(X_q.drop(self.target,axis=1),X[self.target])
                base=pd.concat(all_var.var_report_dict)['count_distr']
            
                report_distr=report[[i for i in report.columns.tolist() if i[-1] in ['count_distr']]]
                psi_sum=report_distr.fillna(0).apply(lambda x:self.psi(x,base),axis=0).droplevel(level=1)\
                                      .assign(bin='psi').set_index('bin',append=True).sort_index(axis=1).groupby(level=[0,1]).sum()
                                      
                report_out['report_psi']=pd.concat([report_distr,psi_sum]).sort_index().reset_index().rename(columns={'level_0':'variable'})                                        
                
        return report_out        
            
    
    def psi(self,base,col):
    
        base=base.replace(0,1e-10)
        col=col.replace(0,1e-10)   
        psi_out=base.sub(col).mul(base.div(col).map(np.log))

        return psi_out            
    
    
    def writeExcel(self):
        
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
        
        if self.output_psi:
            
            self.report_dict['report_psi'].to_excel(writer,sheet_name='bin_psi')  
        
            
        writer.save()     
        print('to_excel done') 
        