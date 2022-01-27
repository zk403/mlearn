#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:43:15 2022

@author: zengke
"""

from BDMLtools.report import varGroupsReport
from BDMLtools.plotter.base import BaseWoePlotter
from BDMLtools.base import Base
import pandas as pd
from joblib import Parallel,delayed,effective_n_jobs


class varGroupsPlot(Base,BaseWoePlotter):
    
    
    def __init__(self,breaks_list,column,target='target',psi_base='all',sort_column=None,special_values=None,
                       sample_weight=None,b_dtype='float64',n_jobs=-1,verbose=0):
        
        self.breaks_list = breaks_list
        self.column=column
        self.target = target
        self.psi_base=psi_base
        self.sort_column=sort_column
        self.special_values=special_values
        self.sample_weight=sample_weight
        self.b_dtype=b_dtype
        self.n_jobs=n_jobs
        self.verbose=verbose
        
        
    def plot(self,X,figure_size=None):
        
        self._check_X(X)
            
        fig_out=self._woe_plot_group(X,self.breaks_list,self.target,self.column,self.psi_base,
                             self.sort_column,figure_size,self.special_values,self.sample_weight,self.b_dtype,
                             self.n_jobs,self.verbose)  
        
        self.fig_out=fig_out
        
        return fig_out

    
    def _woe_plot_group(self,X,breaks_list,target,column,psi_base,sort_column=None,figure_size=None,special_values=None,
                       sample_weight=None,b_dtype='float64',n_jobs=-1,verbose=0):
                 
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
                                                   figure_size,special_values,sample_weight,b_dtype) for colname in breaks_list)
        
        fig_out={colname:fig for fig,colname in res}             

        return fig_out
    
    def _check_param_for_plot(self,columns):
        
        # if row_limit != 0:
            
        #     raise ValueError('Param "row_limit" of varGroupsReport instance must set to 0 if call ".woe_plot_group"')
            
        if len(columns) != 1:
            
            raise ValueError('".woe_plot_group" only support single grouper,reset param "columns" in varGroupsReport instance')  
            
    def _woe_plot_group_single(self,X,breaks_list,colname,target,column,psi_base='all',sort_column=None,figure_size=None,
                               special_values=None,sample_weight=None,b_dtype='float64'):
    
        varbin_g=varGroupsReport({colname:breaks_list[colname]},
                       n_jobs=1,target=target,
                       output_psi=True,
                       columns=[column],
                       psi_base=psi_base,
                       special_values=special_values,
                       sample_weight=sample_weight,
                       b_dtype=b_dtype,
                       row_limit=0,
                      ).fit(X[[colname]+[column]+[target]]).report_dict_raw
        
        binx_g=pd.concat(varbin_g,axis=1).droplevel(0)
        
        figure,colname=self._get_plot_single_group(binx_g,sort_column=sort_column,figure_size=figure_size,show_plot=False)
        
        return figure,colname