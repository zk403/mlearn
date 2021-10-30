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
    
    def __init__(self,categorical_col=None,numeric_col=None,miss_value=[np.nan,'nan'],is_nacorr=False,out_path="report"):
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
            num_report:pd.DataFrame,连续特征质量报告
            char_report:pd.DataFrame,分类特征质量报告
            na_report:pd.DataFrame,单特征缺失率报告
            nacorr_report:pd.DataFrame,缺失率相关性报告
        """
        self.categorical_col = categorical_col
        self.numeric_col = numeric_col
        self.miss_value=miss_value
        self.is_nacorr=is_nacorr
        self.out_path=out_path
        
    def fit(self, X, y=None):
        
        if X.size:
            #填充缺失值
            self.X=X.copy().replace(self.miss_value,np.nan)   
            
            #产生报告
            self.num_report=self.num_info()
            self.char_report=self.char_info()
            self.na_report=self.nan_info()        
            if self.is_nacorr:
                self.nacorr_report=self.nan_corr()
            
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
        
    

    def num_info(self):
        
        """ 数据质量报告-数值特征
        """
        
        data =self.X
        
        if self.numeric_col is None:
            num_col=data.select_dtypes(include='number').columns
        else:
            num_col=self.num_col

        report=data[num_col].describe(percentiles=[0.2,0.4,0.6,0.8]).T.assign(
            MissingRate=data.apply(   
                lambda col:col.isnull().sum()/col.size    
               )
        ).reset_index().rename(columns={'index':'VarName'})
        
        return report
    

    def char_info(self):
        """ 数据质量报告-分类特征
        """
        
        data=self.X
        
        if self.categorical_col is None:
            category_col=data.select_dtypes(include=['object','category']).columns
        else:
            category_col=self.category_col
    
        report=pd.DataFrame()
        for Col in category_col:

            ColTable=data[Col].value_counts().sort_index().rename('Freq').reset_index() \
                .rename(columns={'index':'Levels'}).assign(VarName=Col)[['VarName','Levels','Freq']]

            ColTable['Percent']=ColTable.Freq/data[Col].size #占比
            ColTable['CumFreq']=ColTable.Freq.cumsum() #累计(分类特征类别有次序性时有参考价值)
            ColTable['CumPercent']=ColTable.CumFreq/data[Col].size #累计占比(分类特征类别有次序性时有参考价值)

            report=pd.concat([report,ColTable])
        
        return report
    
   
    def nan_info(self):
        """ 数据质量报告-缺失特征
        """        
        data=self.X
        
        report=pd.DataFrame(
        {'N':data.apply(   
            lambda col:col.size      
           ),
        'Missings':data.apply(   
            lambda col:col.isnull().sum()   
           ),
        'MissingRate':data.apply(   
            lambda col:col.isnull().sum()/col.size    
               ),
        'dtype':data.dtypes
            } ).reset_index().rename(columns={'index':'VarName'})
    
        return report
    
    #@property     
    def nan_corr(self):
        """ 数据质量报告-缺失特征相关性
        """        
        data=self.X        
        nan_info=data.isnull().sum()
        nan_corr_table=data[nan_info[nan_info>0].index].isnull().corr()
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
    
    def __init__(self,breaks_list_dict,special_values=[np.nan],apply_dt=None,psi_base_mon='latest',out_path=None,sheet_name='',n_jobs=-1,verbose=0):
        """ 
        产生业务报告
        Params:
        ------
            breaks_list_dict:dict,分箱字典结构,{var_name:[bin],...},支持scorecardpy与toad的breaks_list结构，
            special_values:list,缺失值指代值
            apply_dt:pd.Series,用于标示X的时期的字符型列且需要能转化为int
                + eg:pd.Series(['202001','202002‘...],name='apply_mon',index=X.index)
            psi_base_mon:str,当apply_dt非空时,psi计算的基准,可选earliest和latest，也可用户自定义
                + earliest:选择数据中apply_dt中最早的时期的分布作为psi基准
                + latest:选择数据中apply_dt中最晚的时期的分布作为psi基准
                + all: 选择总分布作为psi基准
            out_path:将报告输出到本地工作目录的str文件夹下，None代表不输出 
            sheet_name:str,out_path非None时，输出到模型Excel报告的sheet_name后缀,例如"_in_sample"
            n_jobs:int,并行计算job数
            verbose:int,并行计算信息输出等级
        
        Attributes:
        -------
            var_report_dict:dict,特征分析报告字典
            var_report_dict_simplified:dict,apply_dt非None时产生的简化版特征分析报告
            var_psi_report_dict:dict,apply_dt非None时产生的特征分析报告
        """
        self.breaks_list_dict = breaks_list_dict
        self.special_values=special_values
        self.apply_dt = apply_dt
        self.psi_base_mon=psi_base_mon
        self.out_path = out_path
        self.sheet_name=sheet_name
        self.n_jobs=n_jobs
        self.verbose=verbose
        
    def fit(self, X, y=None):
        

        if X.size:
            
            parallel=Parallel(n_jobs=self.n_jobs,verbose=self.verbose)
            
            out_list=parallel(delayed(self.getReport_Single)(X,y,col,self.breaks_list_dict[col],self.apply_dt,self.psi_base_mon,self.special_values) 
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
        
    
    def getReport_Single(self,X,y,col,breaklist_var,apply_dt,psi_base_mon,special_values):
         
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
             
         
         #第2步:判断是否需按月汇总
         if apply_dt is not None:                
             
             var_bin_dt=pd.concat([var_bin,y],axis=1).join(apply_dt)
             
             var_report_dict_interval={} #每期的特征报告字典
             
             #var_report_dict_interval_simplified={} #每期的简化特征报告字典
             
             psi_ts_var={} ##每期的PSI报告字典
                         
             #定义的psi计算的基准月份
             if psi_base_mon=='earliest':
                 
                 psi_base_mon=min(apply_dt.unique().tolist())
                 
             elif psi_base_mon=='latest':
                 
                 psi_base_mon=max(apply_dt.unique().tolist())
             
             elif psi_base_mon=='all':
                 
                 psi_base_mon=var_bin.value_counts().div(var_bin.size)
                 psi_base_mon.index.name='bin'
             
             #elif isinstance(psi_base_mon,pd.core.series.Series):
                 
             #    psi_base_mon=psi_base_mon
             
             else:
                 
                 raise ValueError("psi_base_mon in ('earliest','latest' or ‘all’)")
             
             #计算所有指标
             for mon in apply_dt.unique().tolist():
             
                 var_bin_mon=var_bin_dt[var_bin_dt[apply_dt.name].eq(mon)]
                 rename_aggfunc=dict(zip(['count','sum','mean'],['count','bad','badprob']))
                 result=pd.pivot_table(var_bin_mon,index=col,values=y.name,
                                   margins=False,
                                   aggfunc=['count','sum','mean']).rename(columns=rename_aggfunc,level=0).droplevel(1,1) 
                 #print(result)
                 
                 if result.size:
                     
                     #全部指标
                     var_report_dict_interval[mon]=self.getVarReport_ks(result,col)
                     
                     #print(getVarReport_ks(result,col))
                     
                     #简化版，简化版指标在这里定义
                     #var_report_dict_interval_simplified[mon]=var_report_dict_interval[mon][['count','badprob','total_iv','ks','ks_max']]
                     
                     #PSI指标
                     psi_ts_var[mon]=var_report_dict_interval[mon]['count_distr']
 
                 else:
                     
                     var_report_dict_interval[mon]=None
 
             #计算PSI
             dis_mon=pd.concat(psi_ts_var,axis=1).fillna(0)
             
             if isinstance(psi_base_mon,str):
             
                 dis_mon_psi=dis_mon.apply(lambda x:self.psi(dis_mon[psi_base_mon],x),0)
                 
             else:
                 
                 dis_mon_psi=dis_mon.apply(lambda x:self.psi(psi_base_mon,x),0)
             
             dis_mon=pd.concat([dis_mon,pd.DataFrame(dis_mon_psi.sum().rename('psi')).T],axis=0)
             dis_mon.index.name='bin'
             
             #dis_mon_psi.columns=dis_mon_psi.columns+'_psi'                
             #dis_mon_psi=pd.concat([dis_mon_psi,pd.DataFrame(dis_mon_psi.sum().rename('psi')).T],axis=0)
             #dis_mon_psi_all=dis_mon.join(dis_mon_psi,how='right')
               
                 
             #汇总所有表          
             
             var_report_df_interval=pd.concat(var_report_dict_interval,axis=1)               
             #return var_report_df_interval,var_report_df_interval_simplified,dis_mon  
             return col,var_report_df_interval,dis_mon
         
         
         #若只计算全量数据则只输出全量的特征分析报告
         else:
             
             var_bin=pd.concat([var_bin,y],axis=1)
         
             #print var_bin
             rename_aggfunc=dict(zip(['count','sum','mean'],['count','bad','badprob']))
             result=pd.pivot_table(var_bin,index=col,values=y.name,
                               #columns=apply_dt.name,
                               margins=False,
                               aggfunc=['count','sum','mean']).rename(columns=rename_aggfunc,level=0).droplevel(1,1) 
 
             return col,self.getVarReport_ks(result,col)   

            
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
        
        breaklist_var_new=[i.replace(special_values,'missing').unique().tolist()
                                   for i in [pd.Series(i.split('%,%')) 
                                             for i in breaklist_var]]
        
        map_codes={}
        
        for raw,map_code in product(var_code_raw,breaklist_var_new):
            
            if raw in map_code:
                
                map_codes[raw]='%,%'.join(map_code)
        
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
                
        if not glob(self.out_path+"/var_report.xlsx"):
            
            writer = pd.ExcelWriter(self.out_path+'/var_report.xlsx')            
            pd.DataFrame(None).to_excel(writer,sheet_name='summary')
            writer.save()    
        
        sheet_name=self.sheet_name
            
        writer=pd.ExcelWriter(self.out_path+'/var_report.xlsx',
                              mode='a',
                              if_sheet_exists='replace',
                              #engine_kwargs={'mode':'a','if_sheet_exists':'replace'},
                              engine='openpyxl')
        
        if self.apply_dt is not None:
            
            var_report_df=pd.concat(self.var_report_dict)            
            var_report_psi_df=pd.concat(self.var_report_psi).reset_index().rename(columns={'level_0':'variable'})
        
            var_report_df.to_excel(writer,sheet_name='bin_mon'+sheet_name)
            
            var_report_df.loc[:,product(var_report_df.columns.levels[0].tolist(),['count'])].reset_index().rename(columns={'level_0':'variable'}).to_excel(writer,sheet_name='bin_mon_c'+sheet_name)
            var_report_df.loc[:,product(var_report_df.columns.levels[0].tolist(),['badprob'])].reset_index().rename(columns={'level_0':'variable'}).to_excel(writer,sheet_name='bin_mon_b'+sheet_name)
            var_report_df.loc[:,product(var_report_df.columns.levels[0].tolist(),['ks_max'])].reset_index().rename(columns={'level_0':'variable'}).to_excel(writer,sheet_name='bin_mon_k'+sheet_name)

            var_report_psi_df.to_excel(writer,sheet_name='psi_mon'+sheet_name)
        
        else:
            
            pd.concat(self.var_report_dict).to_excel(writer,sheet_name='4.bin'+sheet_name)            
        
            
        writer.save()     
        print('to_excel done') 