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
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from glob import glob
from itertools import product
import os
import warnings
from joblib import Parallel,delayed


class EDAReport(TransformerMixin):
    
    def __init__(self,categorical_col=None,numeric_col=None,special_values=[np.nan,'nan'],is_nacorr=False,out_path="report"):
        """ 
        产生数据质量报告
        Params:
        ------
            categorical_col:list,类别特征列名
            numeric_col:list,连续特征列名
            miss_value:list,缺失值指代值
            is_nacorr:bool,是否输出缺失率相关性报告
            out_path:str or None,将数据质量报告输出到本地工作目录的str文件夹下，None代表不输出            
        
        Attributes:
        -------
            num_report:pd.DataFrame,连续特征质量报告a
            char_report:pd.DataFrame,分类特征质量报告
            na_report:pd.DataFrame,单特征缺失率报告
            nacorr_report:pd.DataFrame,缺失率相关性报告
        """
        self.categorical_col = categorical_col
        self.numeric_col = numeric_col
        self.special_values=special_values
        self.is_nacorr=is_nacorr
        self.out_path=out_path
        
    def fit(self, X, y=None):
        
        if X.size:
            
            #填充缺失值
            X=X.replace(self.special_values,np.nan)   
            
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
    
    def __init__(self,target,index,columns,rename_columns=None,rename_index=None,out_path=None):
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



class varReport(TransformerMixin):
    
    def __init__(self,breaks_list_dict,special_values=[np.nan],out_path=None,tab_suffix='',n_jobs=-1,verbose=0):
        """ 
        产生业务报告
        Params:
        ------
            breaks_list_dict:dict,分箱字典结构,{var_name:[bin],...},支持scorecardpy与toad的breaks_list结构，
            special_values:list,缺失值指代值
            out_path:将报告输出到本地工作目录的str文件夹下，None代表不输出 
            tab_suffix:本地excel报告名后缀
            n_jobs:int,并行计算job数
            verbose:int,并行计算信息输出等级
        
        Attributes:
        -------
            var_report_dict:dict,特征分析报告字典
        """
        self.breaks_list_dict = breaks_list_dict
        self.special_values=special_values
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.out_path = out_path
        self.tab_suffix=tab_suffix
        
    def fit(self, X, y=None):
        

        if X.size:
            
            self.breaks_list_dict=self.get_Breaklist_sc(self.breaks_list_dict,X,y)
            
            parallel=Parallel(n_jobs=self.n_jobs,verbose=self.verbose)
            
            out_list=parallel(delayed(self.getReport_Single)(X,y,col,self.breaks_list_dict[col],self.special_values) 
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
        
    
    def getReport_Single(self,X,y,col,breaklist_var,special_values):
         
         #print(col)
         #global psi_base_mon1,dis_mon

         #处理缺失值
         var_fillna=X[col].replace(special_values,np.nan)
         
         #breaklist_var=list(breaks_list_dict[col])
         
         #第1步:判断数据类型
         if is_numeric_dtype(var_fillna):
           
             #按照分箱sc的breaklist的区间进行分箱
             var_cut=pd.cut(var_fillna,[-np.inf]+breaklist_var+[np.inf],duplicates='drop',right=False)
             
             var_bin=pd.Series(np.where(var_cut.isnull(),'missing',var_cut),
                       index=var_cut.index,
                       name=col)
         
         elif is_string_dtype(var_fillna):    
             
             var_cut=pd.Series(np.where(var_fillna.isnull(),'missing',var_fillna),
                       index=var_fillna.index,
                       name=var_fillna.name)
             
             
             
             #转换字原始符映射到分箱sc的breaklist的字符映射
             var_code_raw=var_cut.unique().tolist()
                                   
             map_codes=self.raw_to_bin_sc(var_code_raw,breaklist_var,special_values)
             
             var_bin=var_cut.map(map_codes)
             
         else:
             
             raise IOError('dtypes in X in (number,object),others not support')
         
         #若只计算全量数据则只输出全量的特征分析报告
             
         var_bin=pd.concat([var_bin,y],axis=1)
    
         #print var_bin
         rename_aggfunc=dict(zip(['count','sum','mean'],['count','bad','badprob']))
         result=pd.pivot_table(var_bin,index=col,values=y.name,
                           margins=False,
                           aggfunc=['count','sum','mean']).rename(columns=rename_aggfunc,level=0).droplevel(level=1,axis=1) 

         return col,self.getVarReport_ks(result,col),None   

            
    def getVarReport_ks(self,var_ptable,col):
        
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
        
        #toad格式时转换为sc格式
        if count>0:
        
            cate_colname=X.select_dtypes(exclude='number')
            num_colname=X.select_dtypes(include='number')

            break_list_sc=dict()

            #将toad的breaklist转化为scorecardpy的breaklist
            for key in break_list.keys():
                if key in cate_colname and break_list[key]:#防止分箱结果为空

                    bin_value_list=[]
                    for value in break_list[key]:
                        #if 'nan' in value:
                        #    value=pd.Series(value).replace('nan','missing').tolist()
                        bin_value_list.append('%,%'.join(value))

                    break_list_sc[key]=bin_value_list

                elif key in num_colname and break_list[key]:#防止分箱结果为空
                    break_list_sc[key]=break_list[key]

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
    
    def __init__(self,breaks_list_dict,columns,sort_columns=None,target='target',row_limit=1000,output_psi=False,psi_base='all',
                 special_values=[np.nan],n_jobs=-1,verbose=0,out_path=None,tab_suffix='_group'):
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
        row_limit,int,分组行数限制，小于限制的组不统计其任何指标，返回空，建议设定此参数以保证分组统计计算不出错
        special_values:list,缺失值指代值
        n_jobs:int,并行计算job数
        verbose:int,并行计算信息输出等级
        out_path:将报告输出到本地工作目录的str文件夹下，None代表不输出 
        tab_suffix:本地excel报告名后缀
        
        Attributes:
        -------
            report_dict:dict,所有产生的报告
        """
        
        self.breaks_list_dict=breaks_list_dict
        self.target=target
        self.columns=columns
        self.sort_columns=sort_columns      
        self.row_limit=row_limit
        self.output_psi=output_psi
        self.psi_base=psi_base
        self.special_values=special_values
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.out_path=out_path
        self.tab_suffix=tab_suffix
       
    def fit(self, X, y=None):               
        

        if X.size:
            
            
            self.breaks_list_dict={key:self.breaks_list_dict[key] for key in self.breaks_list_dict if key in X.drop(self.columns,axis=1).columns}            
            
            if self.sort_columns:
                
                X=X.drop(list(self.sort_columns.keys()),axis=1).join(
                    pd.DataFrame(                      
                    {col:pd.Categorical(X[col],categories=self.sort_columns[col],ordered=True) for col in self.sort_columns},               
                    index=X.index)
                )
                       
            result={}
            
            X_g_gen=X.groupby(self.columns)
            
            for i in X_g_gen.groups:
            
                group_dt=X_g_gen.get_group(i)
                X_g=group_dt.drop([self.target]+self.columns,axis=1)
                y_g=group_dt[self.target]    
            
                if X.size>self.row_limit:
                    
                    res=varReport(breaks_list_dict=self.breaks_list_dict,
                                  special_values=self.special_values,
                                  n_jobs=self.n_jobs,                                  
                                  verbose=self.verbose).fit(X_g,y_g)
                    
                    result[i]=pd.concat(res.var_report_dict)
                    
                else:
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
                                      .reset_index().assign(bin='psi').groupby(['index','bin']).sum()
                                      
                report_out['report_psi']=pd.concat([report_distr,psi_sum]).sort_index().reset_index().rename(columns={'level_0':'variable'})
            
            else:            
                
                X_q=X.query(psi_base)
                
                if not X_q.size:
                    
                    raise IOError('X.query has 0 row, check the query expr.')
                
                all_var=varReport(breaks_list_dict=self.breaks_list_dict,
                                  special_values=self.special_values,
                                  n_jobs=self.n_jobs,                                  
                                  verbose=self.verbose).fit(X_q.drop(self.target,axis=1),X[self.target])
                base=pd.concat(all_var.var_report_dict)['count_distr']
            
                report_distr=report[[i for i in report.columns.tolist() if i[-1] in ['count_distr']]]
                psi_sum=report_distr.fillna(0).apply(lambda x:self.psi(x,base),axis=0).droplevel(level=1)\
                                      .reset_index().assign(bin='psi').groupby(['index','bin']).sum()
                                      
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
        