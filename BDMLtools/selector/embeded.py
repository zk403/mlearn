#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:05:03 2022

@author: zengke
"""

from sklearn.svm import l1_min_c
from sklearn.base import TransformerMixin
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import Parallel,delayed,effective_n_jobs
from BDMLtools.base import Base
import numpy as np
from itertools import product
import pandas as pd
from sklearn.metrics import log_loss,roc_auc_score
import matplotlib.pyplot as plt
import os

class LassoLogit(Base,TransformerMixin):    
    
    """
    Lasso-Logistic(L1) Regression
    加入L1正则、使用交叉验证、无统计检验的logit回归，适合较大数据量较多特征的二分类建模需求

    参考资料:
    [sklearn-lr]https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    [sklearn-lr L1 path]https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_path.html#sphx-glr-auto-examples-linear-model-plot-logistic-path-py
    [stanford-glmnet]https://glmnet.stanford.edu/articles/glmnet.html
    [R--glmnet source code]https://github.com/cran/glmnet
    Parameters:
    --        
        c_num=20,int:l1正则项的粒度，越大的数值代表越多的l1正则项参与寻优
        method='1se':str,若c_num非None，LassoLogit进行重新拟合时的c值
            + '1se':一倍标准误差原则，同R的glmnet。交叉验证中最优指标(logloss或其他优化指标)下的c值对应一倍标准误差范围内的最小c值重拟合模型                    
        standard=True:bool,是否进行数据标准化
        metric:str,交叉验证评估标准，可选'roc_auc','neg_log_loss'
        repeats:int,RepeatedStratifiedKFold交叉验证重复次数
        cv:int,交叉验证的折数
        keep:list 需要保留的列的列名list
        n_jobs,int,交叉验证的joblib的并行数,默认1
        verbose,int,并行信息输出等级
        random_state:随机种子
        
    Method:    
    --     
        refit_with_C(X,y,C,sample_weight):以自定义C值重新拟合回归模型，要求LassoLogit对象需拟合过
     
    Attribute:    
    --
        model_refit:sklearn.linear_model.LogisticRegression对象，重新拟合的回归模型
        feature_names_:list,输入X的列名list
        cv_path_:pd.DataFrame,L1正则收缩路径结果
        cv_res_:pd.DataFrame,L1正则交叉验证结果
        plot_path:matplotlib.figure.Figure,L1正则收缩路径图
        plot_valscore:matplotlib.figure.Figure,L1正则交叉验证图
        
    Examples
    --

    """  
    
    def __init__(self,c_num=20,method='1se',standard=True,metric='neglogloss',repeats=1,cv=5,
                 n_jobs=-1,verbose=0,random_state=123):
      
        self.c_num=c_num
        self.method=method
        self.standard=standard
        self.metric=metric
        self.cv=cv
        self.repeats=repeats
        self.n_jobs=n_jobs
        self.verbose=verbose     
        self.random_state=random_state
        self._is_fitted=False
        
    def predict_proba(self,X,y=None):
        '''
        最优参数下的模型的预测
        Parameters:
        --
        X:pd.DataFrame对象
        '''      
        self._check_is_fitted()
        pred = self.model_refit.predict_proba(np.array(X))[:,1]        
        return pred

    
    def transform(self,X,y=None): 
        
        self._check_is_fitted()
        self._check_X(X)
        
        columns=np.array(self.feature_names_)[(self.model_refit.coef_!=0)[0]]

        return X[columns]
          
    def fit_plot(self,X,y,sample_weight=None,figure_size=(15,6)):
        '''
        进行参数优化
        Parameters:
        --
        X:pd.DataFrame对象
        y:目标变量,pd.Series对象
        '''   
        self._check_data(X,y)
        self._check_ws(y,sample_weight)
        
        self.feature_names_=X.columns.tolist()
        
        X=np.array(X)
        y=np.array(y)
        
        if sample_weight is not None:
        
            sample_weight=np.array(sample_weight)
            

        if self.standard:
            
            X=StandardScaler().fit_transform(X)
        
        cv_path_,cv_res_=self._logit_cv(X,y,sample_weight)
        max_s,max_s_se=cv_res_[cv_res_['valscore_avg']==cv_res_['valscore_avg'].max()].to_numpy()[0][[2,4]]
        
        self.plot_path=self._plot_path(cv_path_,figure_size)
        self.plot_valscore=self._plot_valscore(cv_res_,figure_size)
        
        self.c_best=cv_res_[cv_res_['valscore_avg']==cv_res_['valscore_avg'].max()]['C'].ravel()[0]
        self.c_1se=cv_res_[(cv_res_['valscore_avg']<=max_s) & \
                           (cv_res_['valscore_avg']>=max_s-max_s_se)].sort_values('log(C)').iloc[0,0]
            
        self.cv_path_=cv_path_
        self.cv_res_=cv_res_
            
#         if self.method=='best':    
            
#             refit_c=self.c_best
            
        if self.method=='1se':    
            
            refit_c=self.c_1se
            
        else:
            
            raise ValueError('method in ("1se")')
            
        self.model_refit = LogisticRegression(
            C=refit_c,
            penalty="l1",
            solver="saga",
            tol=1e-6,
            max_iter=int(1e6)
        ).fit(X,y)
            
        self._is_fitted=True
        
        return self    
    
    def _logit_cv(self,X,y,ws):
        
        cs = l1_min_c(X, y, loss="log") * np.logspace(0, 7, self.c_num)

        cv = RepeatedStratifiedKFold(n_splits=self.cv,n_repeats=self.repeats,random_state=self.random_state)  
        
        n=self.cv*self.repeats
        
        n_jobs=effective_n_jobs(self.n_jobs)
        
        parallel=Parallel(n_jobs=n_jobs,verbose=self.verbose)
        out_list=np.array(parallel(delayed(self._logit_cv_parallel)(X,y,i,ws,'neglogloss')
                        for i in product(cs,cv.split(X, y))),dtype=object)

        cs_cv=out_list[:,0]
        val_score_cv=out_list[:,1]
        coefs_cv=out_list[:,2]
        
        cv_path_=pd.DataFrame({np.log10(c):np.mean(np.array(coefs_cv)[np.array(cs_cv)==c],0) for c in np.unique(cs_cv)}).T.join(
            pd.Series({np.log10(c):np.sum(np.mean(np.array(coefs_cv)[np.array(cs_cv)==c],0)!=0) for c in np.unique(cs_cv)},name='coef_cnt')
        )
        cv_path_['log(C)']=cv_path_.index

        cv_res_=pd.DataFrame([(
          c,np.log10(c),
          np.mean(np.array(val_score_cv)[np.array(cs_cv)==c]),
          np.std(np.array(val_score_cv)[np.array(cs_cv)==c]),
          np.std(np.array(val_score_cv)[np.array(cs_cv)==c])/np.sqrt(n-1),
          np.sum(np.mean(np.array(coefs_cv)[np.array(cs_cv)==c],0)!=0)) for c in np.unique(cs_cv)],
          columns=['C','log(C)','valscore_avg','valscore_std','valscore_err','coef_cnt'])
        
        return cv_path_,cv_res_
        

    def _logit_cv_parallel(self,X,y,i,ws=None,metric='neglogloss'):
    
        c=i[0]
        train=i[1][0]
        test=i[1][1]

        if ws is None:

            ws=np.ones(len(X))
            
        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
            tol=1e-6,
            max_iter=int(1e6),
            warm_start=True
        )

        clf.set_params(C=c)
        clf.fit(X[train], y[train],ws[train])    

        if self.metric=='neglogloss':

            score=-log_loss(y[test],clf.predict_proba(X[test])[:,1],sample_weight=ws[test])  

        elif self.metric=='roc_auc':

            score=roc_auc_score(y[test],clf.predict_proba(X[test])[:,1],sample_weight=ws[test]) 

        else:

            raise ValueError('metric in ("neglogloss","roc_auc")')

        coefs=clf.coef_.ravel().copy()

        return i[0],score,coefs
    
    
    def refit_with_C(self,X,y,C,sample_weight=None):
        
        self._check_is_fitted()
        self._check_data(X,y)
        self._check_ws(y,sample_weight)
        
        self.feature_names_=X.columns.tolist()
        
        X=np.array(X)
        y=np.array(y)
        
        
        self.model_refit = LogisticRegression(
            C=C,
            penalty="l1",
            solver="saga",
            tol=1e-6,
            max_iter=int(1e6)
        ).fit(X,y,sample_weight)
        
    
    def _plot_path(self,cv_path_,figure_size):

        from plotnine import ggplot,geom_point,geom_line,ggtitle,labs,theme_bw,theme,scale_color_discrete,element_text,aes

        cv_path_=pd.concat(
            [
                cv_path_[cv_path_['log(C)']==cv_path_[cv_path_.coef_cnt==0]['log(C)'].max()],
                cv_path_[cv_path_.coef_cnt!=0]
            ]
        ).copy()

        cv_path_m_=pd.melt(
            cv_path_.drop(['coef_cnt'],axis=1),id_vars='log(C)'
        )

        shift=(cv_path_['log(C)'].max()-cv_path_['log(C)'].min())*0.05
        xticks=[cv_path_['log(C)'].min()-shift]+cv_path_['log(C)'].tolist()+[cv_path_['log(C)'].max()+shift] 
        lb=np.array(cv_path_['coef_cnt'],dtype=object)
        
        if len(lb)>50:
            
            lb[[i for i in range(0,len(lb),2)]]=''

        labels=['']+lb.tolist()+['']

        fig=(ggplot(cv_path_m_,aes(x='log(C)', y='value',group='variable',color='variable'))+
            geom_point(color='red',size=2)+
            geom_line()+
            ggtitle('Lasso-Logistic Regularization Path')+
            labs(x = "Log(C)", y = "Coefficients") +
            scale_color_discrete(guide=False)+ 
            theme_bw() +
            theme(figure_size=figure_size,plot_title=element_text(y=0.98))
        ).draw()
        plt.close()

        ax1=fig.get_axes()[-1]
        ay=fig.get_axes()[0]
        ay.set_yscale('linear')
        ax1.set_xticks(xticks)
        ax1.set_xscale('linear')
        ax1.set_xlim(min(xticks),max(xticks))
        
        if os.name!='nt':
        
            ax2=ax1.twiny()
            ax2.set_xticks(xticks,labels)
            ax2.set_xlabel('Num of non-zero coefficients')
        
        return fig         
    
    
    def _plot_valscore(self,cv_res_,figure_size):
        
        from plotnine import ggplot,theme,element_text,theme_bw,ggtitle,labs,geom_errorbar,aes,geom_line,geom_point       

        cv_res_=pd.concat(
            [
                cv_res_[cv_res_['log(C)']==cv_res_[cv_res_['coef_cnt']==0]['log(C)'].max()],
                cv_res_[cv_res_['coef_cnt']!=0]
            ]
        ).copy()

        shift=(cv_res_['log(C)'].max()-cv_res_['log(C)'].min())*0.05
        xticks=[cv_res_['log(C)'].min()-shift]+cv_res_['log(C)'].tolist()+[cv_res_['log(C)'].max()+shift] 
        lb=np.array(cv_res_['coef_cnt'],dtype=object)
        if len(lb)>50:
            lb[[i for i in range(0,len(lb),2)]]=''

        labels=['']+lb.tolist()+['']

        fig=(ggplot(cv_res_,aes(x='log(C)', y='valscore_avg'))+
            geom_point(color='red',size=3)+
            geom_line()+
            ggtitle('Lasso-Logistic Log(C) vs Valscore')+
            geom_errorbar(aes(ymin=cv_res_['valscore_avg']-cv_res_['valscore_err'],
                              ymax=cv_res_['valscore_avg']+cv_res_['valscore_err']),width=.2)+
            labs(x = "Log(C)", y = "Val-Score") +
            theme_bw() +
            theme(figure_size=figure_size,
                  plot_title=element_text(y=0.98),
                 ) 
        ).draw()
        plt.close()

        ax1=fig.get_axes()[-1]
        ay=fig.get_axes()[0]
        ay.set_yscale('linear')
        ax1.set_xticks(xticks)
        ax1.set_xscale('linear')
        ax1.set_xlim(min(xticks),max(xticks))
        
        if os.name!='nt':
            
            ax2=ax1.twiny()
            ax2.set_xticks(xticks,labels)
            ax2.set_xlabel('Num of non-zero coefficients')

        return fig