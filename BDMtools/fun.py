#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 09:56:44 2021

@author: zengke
"""

import pandas as pd
import numpy as np
from itertools import product
from pandas.api.types import is_string_dtype,is_numeric_dtype

def raw_to_bin_sc(var_code_raw,breakslist_var):
    
    """ 
    分箱转换，将分类特征的值与breaks对应起来,用于分类levels的合并后的重编码
    Params:
    ------
    
        var_code_raw,list,原始字符编码
        breakslist_var,list,合并后的字符编码
        special_values,字符列的特殊值，其将被替换会
        
    Return:
    -------
        map_codes,dict,原始编码与新编码的对应关系。通过str_new=str_old.map(raw_to_bin_sc)完成编码
    """ 
    
    breakslist_var_new=[i.unique().tolist() for i in [pd.Series(i.split('%,%')) for i in breakslist_var]]
    
    map_codes={}
    
    for raw,map_code in product(var_code_raw,breakslist_var_new):
        
        
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



def sp_replace_df(X,special_values,fill_num=np.nan,fill_str='missing'):
    
    """ 
    特殊值替换,数值特征缺失值替换为np.nan,分类特征缺失值替换为'missing'
    Params:
    ------
    
        X,pd.DataFrame,原始数据
        special_values,list,字符列的特殊值
        fill_num,float,int,数值列填充值
        fill_str,str,int,字符列填充值
        
    Return:
    -------
        X,pd.DataFrame,替换后的数据
    """      
    
    X_num=X.select_dtypes('number').replace(special_values,fill_num)
    X_str=X.select_dtypes('object').replace(special_values,fill_str)
    X_oth=X.select_dtypes(exclude=['object','number'])
    X_all=pd.concat([X_num,X_str,X_oth],axis=1)

    return X_all


def sp_replace_col(col,special_values,fill_num=np.nan,fill_str='missing'):
    
    """ 
    特殊值替换,数值特征缺失值替换为np.nan,分类特征缺失值替换为'missing'
    Params:
    ------
    
        var_code_raw,list,原始字符编码
        sp_dict,dict,替换字典，{colname:sp_values_list,...}格式
        fill_num,float,int,数值列填充值
        fill_str,str,int,字符列填充值
        
    Return:
    -------
        X,pd.DataFrame,替换后的数据
    """             

    if is_numeric_dtype(col):
        
        if col.name in special_values:
        
            return col.replace(special_values[col.name],fill_num)
        
        else:
            
            return col    
    
    elif is_string_dtype(col):
        
        if col.name in special_values:
        
            return col.replace(special_values[col.name],fill_str)
        
        else:
            
            return col
        
    else:
        return col
       
    
def sp_replace(X,special_values,fill_num=np.nan,fill_str='missing'):
    
    """ 
    特殊值替换,数值特征缺失值替换为np.nan,分类特征缺失值替换为'missing'
    Params:
    ------
    
        X,pd.DataFrame,原始数据
        special_values,list or dict,列的特殊值指代
        fill_num,float,int,数值列填充值
        fill_str,str,int,字符列填充值
        
    Return:
    -------
        X,pd.DataFrame,替换后的数据
    """    
    
    if special_values is None:
        
        X=X
    
    elif isinstance(special_values, list):
        
        X=sp_replace_df(X,special_values,fill_num,fill_str)
        
    elif isinstance(special_values, dict):
        
        X=X.apply(sp_replace_col,args=(special_values,fill_num,fill_str,))
        
    else:
        
        raise ValueError('special_values is list,dict or None')
       
    return X



def get_Breaklist_sc(break_list,X,y):
 
    
    count=0
    for var_list in list(break_list.values()):
        
        for value in var_list:
            if isinstance(value,list):
                count=count+1           
            break
        
    columns=list(break_list.keys())
    
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



