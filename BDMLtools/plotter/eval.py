#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:49:51 2022

@author: zengke
"""

import pandas as pd
from BDMLtools.base import BaseEval
from BDMLtools.plotter.base import BaseEvalPlotter


class perfEval(BaseEval,BaseEvalPlotter):
    
    def __init__(self,show_plot=('ks', 'lift', 'gain', 'roc', 'lz', 'pr', 'f1', 'density'),title=None,sample_weight=None,n_jobs=1,verbose=0):

        self.show_plot=show_plot
        self.title=title
        self.n_jobs=n_jobs
        self.verbose=verbose   
        
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
        
        self._check_plot_params(self.show_plot)
        self._check_params(y_pred, y_true, group,sample_weight)
        self._check_values(y_pred, y_true, group,sample_weight)
        
        if y_pred.max()>1 or y_pred.min()<0:
            
            self.pred_desc=True
        
        else:
            
            self.pred_desc=False            
                  
        dt_plt=self._get_df(y_pred,y_true,group)
        
        
        if sample_weight is not None :
            
            sample_weight=pd.Series(sample_weight,index=dt_plt.index) 
            
        dt_plt_dict=self._get_dfev_dict(dt_plt,sample_weight,self.n_jobs,self.verbose)
           
        
        figs_d={}
        
        for ty in self.show_plot:
            
            if ty=='density':
                
                figs_d[ty]=self._plot_funs[ty](dt_plt,figure_size,sample_weight,self.title)
                
            else:
                
                figs_d[ty]=self._plot_funs[ty](dt_plt_dict,figure_size,self.title)                   

        return figs_d        
               
