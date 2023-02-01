#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 21:58:45 2021

@author: zengke
"""

import pandas as pd
from BDMLtools.selector import binSelector
from BDMLtools.encoder import woeTransformer
import shap
from scipy.stats import pearsonr,spearmanr
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from joblib import effective_n_jobs,Parallel,delayed
import numpy as np

class shapCheck:
    
    """ 
    对树模型结果进行woe/shap相关性校验
    
    对于binary loss function:
        
        shap=ln(P_event/P_nonevent) --> shap值越高,预测结果越倾向于事件发生，反之则倾向于事件不发生
        
    对于weight of evidence:
    
        woe=ln(% of event/% of non-events) --> woe值越高,事件更倾向于发生，反之则倾向于事件倾向于不发生
        
    因此两者理应具有高度相关性，woe/shap相关性校验过程如下
    
    1.对原始数据进行分箱并进行woe编码
    2.根据构建好的模型对原始数据计算shap值
    3.根据同一特征的woe编码与shap值进行对比(相关性分析)
    4.[可选]绘制特征的woe编码与shap值的散点图
    
    Params:
    ------
        Estimator:训练好的二分类(binary loss function)的boosting模型(xgboost,lightgbm,catboost)
        woe_raw:bool,woe编码后的数据是否作为原始输入数据，若为True将忽略bin_method与bin_num_limit参数
        bin_method:str,对训练好的数据进行woe编码时分箱方式的选择,默认'freq'
        bin_num_limit:int,设定的分箱个数
        plot:bool,
        n_jobs:int,并行计算job数
        verbose:int,并行计算信息输出等级
    
    Attributes:
    -------
        report:DataFrame,相关性分析报告
        figs:dict,绘制的散点图
        
    Method:
    -------
        fit_plot:拟合数据并产生报告与绘图 
    
    """    
    
    def __init__(self,Estimator,woe_raw=False,bin_method='freq',bin_num_limit=8,plot=True,n_jobs=1,verbose=0):
        
        self.Estimator=Estimator
        self.woe_raw=woe_raw
        self.bin_method=bin_method
        self.bin_num_limit=bin_num_limit
        self.plot=plot
        self.n_jobs=n_jobs
        self.verbose=verbose       
          
    def fit_plot(self,X,y,sample_weight=None,figure_size=None): 
        
        if self.woe_raw:
            
            X_woe=X
            
        else:    

            bins=binSelector(bin_num_limit=self.bin_num_limit,method=self.bin_method,
                                iv_limit=0,sample_weight=sample_weight,n_jobs=self.n_jobs,verbose=self.verbose).fit(X,y)
            
            X_woe=woeTransformer(bins.bins,woe_missing=0,distr_limit=0.05).fit_transform(X)
        

        explainer = shap.TreeExplainer(self.Estimator)        
            
        if isinstance(self.Estimator,LGBMClassifier):
            
            cate_cols=X.select_dtypes(['object','category']).columns.tolist()
        
            if cate_cols:
                
                X=X.apply(lambda x:x.astype('category') if x.name in cate_cols else x)
                
            X_shap=pd.DataFrame(explainer.shap_values(X)[1],columns=X.columns,index=X.index)
            

        elif self.Estimator is CatBoostClassifier:
            
            cate_cols=X.select_dtypes(['object','category']).columns.tolist()

            if cate_cols:
                
                X=X.apply(lambda x:x.astype('str') if x.name in cate_cols else x)
                
        else:
                
            X_shap=pd.DataFrame(explainer.shap_values(X),columns=X.columns,index=X.index)
        
        self.report=self._check(X_woe, X_shap)
        
        if self.plot==True:
            
            n_jobs=effective_n_jobs(self.n_jobs)

            parallel=Parallel(n_jobs=n_jobs,verbose=self.verbose)

            out_list=parallel(delayed(self._plot_single)(X_woe,X_shap,col,figure_size=figure_size)
                              for col in X_woe.columns)
            
            self.figs={col:fig for col,fig in out_list}
                      
            shap.summary_plot(X_shap.values, X)
        
        return self   
    
    def _plot_single(self,X_woe,X_shap,col,figure_size=None):
              
        dt=pd.concat([X_woe[col].rename('woe'),X_shap[col].rename('shap')],axis=1)
        
        fig=self._plot(dt,col,figure_size)
        
        return col,fig
    
    
    def _plot(self,dt,col,figure_size=None):
        
        from plotnine import ggplot,geom_point,stat_smooth,theme_bw,labs,theme,aes,annotate
        
        x=dt['shap'].quantile(0.4)
        y=dt['woe'].max()
        
        p=pearsonr(dt['shap'],dt['woe'])[0]
        
        p=(ggplot(aes(dt['shap'], dt['woe']))
         + geom_point(size=1,color='red',alpha=0.5,shape=10)
         + stat_smooth(method='lm',se=True,size=0.5,linetype="dashed",color="blue")
         + theme_bw()
         + labs(x='Shapley Value',y='Weight of Evidence',title='Scatter Plot of Shapley-Value and WoE:{}'.format(col))
         + theme(figure_size=figure_size)
         + annotate("text", label = "Pearson's r:{}".format(np.round(p,3)), x = x, y = y,color='blue'))

        return p
    
    def _check(self,X_woe,X_shap):
        
        result=pd.concat(
            [pd.DataFrame([pearsonr(X_woe[col],X_shap[col]) for col in X_woe.columns],
                        columns=['pearsonr','pr_p'],index=X_woe.columns),
             pd.DataFrame([spearmanr(X_woe[col],X_shap[col]) for col in X_woe.columns],
                        columns=['spearmanr','sr_p'],index=X_woe.columns)],
            axis=1)     
        
        return result