#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:49:51 2022

@author: zengke
"""

from BDMLtools.base import BaseEval
from BDMLtools.plotter.base import BaseEvalPlotter


class perfEval(BaseEval,BaseEvalPlotter):
    
    """ 
    排序类模型评估,改写自R scorecard::perf_eval，绘图使用plotnine-0.8.0
    
    Params:
    ------
        show_plot=('ks','roc'):tuple,产生的模型评估图种类,可选('ks', 'lift', 'gain', 'roc', 'lz', 'pr', 'f1', 'density')
            + 'ks':ks曲线
            + 'lift':lift曲线
            + 'gain':gain曲线
            + 'roc':roc曲线
            + 'pr':pr曲线
            + 'lz':lorenz曲线
            + 'f1':f1曲线
            + 'density':核密度分布曲线       
        title=None,str,评估图的title
        pred_desc=None,是否反相排序y_pred,
            + pred_desc=False情况下即y_pred越大代表event的概率估计越大，若y_pred越小代表event的概率估计越大时，请设定为pred_desc=True
            + pred_desc=None时,将自动检测y_pred，若y_pred范围在[0,1]之内时pred_desc将设定为False，y_pred范围在[0,1]之外时pred_desc将设定为True
        
    Method:
    -------
        plot:绘制模型评估图
        
    """
    
    
    def __init__(self,show_plot=('ks','roc'),title=None,pred_desc=None):
        

        self.show_plot=show_plot
        self.title=title
        self.pred_desc=pred_desc

        
    @property
    def _plot_funs(self):
        
         return {'density':self._plot_density,
             'ks':self._plot_ks,
             'lift':self._plot_lift,
             'gain':self._plot_gain,
             'roc':self._plot_roc,
             'lz':self._plot_lz,
             'pr':self._plot_pr,
             'f1':self._plot_f1}
        
    def plot(self,y_pred,y_true,group=None,sample_weight=None,figure_size=(4,4)):
        
        """ 
        绘制模型评估图,改写自R scorecard::perf_eval，绘图使用plotnine-0.8.0
        
        Params:
        ------
            y_pred:pandas.Series,预测值,可以是预测概率也可以是预测评分
            y_true:pandas.Series,实际值,包含0,1的实际值Series,其index必须与y_pred一致
            group=None:pandas.Series,组变量,用于指代预测值与实际值的组，其index必须与y_pred一致，默认None即无组变量
            sample_weight=None:pandas.Series,样本权重,用于标识样本点的权重,非负非0，默认None即每个样本点权重都是1,其index必须与y_pred或y_true的index一致
            figure_size=(4,4),图的大小
            
        Return:
        -------
            figs_d:dict,{plot_type:fig}形式
            
        """
        
        
        self._check_plot_params(self.show_plot,self.pred_desc)
        self._check_params(y_pred, y_true, group,sample_weight)
        self._check_values(y_pred, y_true, group,sample_weight)
        
        if self.pred_desc is None:
        
            if y_pred.max()>1 or y_pred.min()<0:
                
                self.pred_desc=True
            
            else:
                
                self.pred_desc=False                            
                  
        dt_plt=self._get_df(y_pred,y_true,group).sample(frac=1,random_state=182)
        
        
        if sample_weight is not None:
            
            sample_weight=sample_weight[dt_plt.index]

            
        dt_plt_dict=self._get_dfev_dict(dt_plt,sample_weight)
           
        
        figs_d={}
        
        for ty in self.show_plot:
            
            if ty=='density':
                
                dt_dens=self._get_df_density(dt_plt,sample_weight)
                
                figs_d[ty]=self._plot_funs[ty](dt_dens,figure_size,self.title)
                
            else:
                
                figs_d[ty]=self._plot_funs[ty](dt_plt_dict,figure_size,self.title)                   

        return figs_d        
               
