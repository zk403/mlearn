#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:49:51 2022

@author: zengke
"""

from BDMLtools.base import BaseEval
from BDMLtools.plotter.base import BaseEvalPlotter
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_recall_curve,confusion_matrix


class perfEval(BaseEval,BaseEvalPlotter):
    
    """ 
    二分类排序类模型评估,改写自R scorecard::perf_eval，绘图使用plotnine-0.9.0
    
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

               

class perfEval2(BaseEval):
    
    """ 
    二分类决策类模型评估,以混淆矩阵指标为主,绘图使用plotnine-0.9.0
    
    Params:
    ------
    pred_prob=False:bool,预测值是否是连续值,False时不做处理直接产生混淆矩阵,True时将根据fbeta值寻找最优的切分点并产生混淆矩阵
    pred_desc=False:bool,当预测值为连续值时的顺序,当连续值越大越倾向于事件发生(预测概率)，该参数应设定为False;当连续值越小越倾向于事件发生时应设定为True
    normalize=None,str or None,混淆矩阵标准化方式，同sklearn.metrics.confusion_matrix的normalize参数:
                    normalize='all'(总数标准化),用于查看accurany指标
                    normalize='pred'(列标准化),用于查看precison指标
                    normalize='true'(行标准化),用于查看recall指标
                    normalize=None,不进行任何标准化
    labels=None or list,设定二分类标签,labels=[0标签,1标签],None为保持数据默认,例如labels=['good','bad']
    sort_group=None or list,当评估的数据存在分组时，使用sort_group排序混淆矩阵绘图中的组顺序,例如sort_group=['train','test']
    beta=1 or float,fbeta指标的beta预定值,fbeta值将用于预测结果评估与最优切分点的选择;
                    beta=1时即平衡precison与recall(等价于f1),beta<1时更侧重precison,beta>1时更侧重recall
        
    Method:
    -------
        plot:绘制混淆矩阵评估图
        
    """
    
    
    def __init__(self,pred_prob=False,pred_desc=False,normalize=None,labels=None,sort_group=None,beta=1):
        

        self.pred_prob=pred_prob
        self.pred_desc=pred_desc
        self.normalize=normalize
        self.labels=labels
        self.sort_group=sort_group
        self.beta=beta       
        
        
    def plot(self,y_pred,y_true,group=None,base_group=None,sample_weight=None,figure_size=(5,5)):
        
        """ 
        二分类决策类模型评估,以混淆矩阵指标为主,绘图使用plotnine-0.9.0

        Params:
        ------
        y_pred:pd.Series,预测值,可以是预测二分类(0为事件不发生,1为事件发生)也可以是预测概率(需设定pred_prob=True),index必须与y_true一致
        y_true,pd.Series,实际值,二分类实际值,0为事件不发生,1为事件发生,index必须与y_pred一致
        group:None or pd.Series,组变量,用于区分不同组,index必须与y_pred和y_true一致
        base_group:None or list,当y_pred为预测概率且group存在时,指定group中的某一组或某几组用于产生最优切分点,例如base_group=['train']
        sample_weight:None or pd.Series,样本权重,index必须与y_pred或y_true一致
        figure_size:tuple,图像大小,默认(5,5)

        Return:
        -------
            p:绘制的混淆矩阵评估图
        """
       
        self._check_params(y_pred, y_true, group,sample_weight)
        self._check_values(y_pred, y_true, group,sample_weight)
                
        if self.pred_prob:
            
            if self.pred_desc:
                
                y_pred=-y_pred
                
            if (group is not None) and (base_group is not None):

                g_mask=group.isin(base_group)

                thre=self._get_bestcut(y_true[g_mask],y_pred[g_mask],
                                       sample_weight=None if sample_weight is None else sample_weight[g_mask],
                                       beta=self.beta)                    

            else:

                thre=self._get_bestcut(y_true,y_pred,sample_weight=sample_weight,beta=self.beta)  

            cm,dt_plot=self._get_pltdt(y_true,y_pred.ge(thre).astype('float'),group,sample_weight=sample_weight,
                   labels=self.labels,normalize=self.normalize,
                   sort_group=self.sort_group) 

            self._print_binarycm(cm,beta=self.beta)

            p=self._plot_cm(dt_plot,figure_size=figure_size,normalize=self.normalize)
                
        else:
            
            if y_pred.unique().size>2:
                
                raise ValueError('y_pred should be binary,try set pred_prob=True if y_pred is probability')
                
            cm,dt_plot=self._get_pltdt(y_true,y_pred,group,sample_weight=sample_weight,
                       labels=self.labels,normalize=self.normalize,
                       sort_group=self.sort_group) 
                
            self._print_binarycm(cm,beta=self.beta)
                
            p=self._plot_cm(dt_plot,figure_size=figure_size,normalize=self.normalize)           
            
        return p      

    def _print_binarycm(self,cm,beta=1):
        
        if isinstance(cm,dict):
            
            for key in cm:
                
                tn, fp, fn, tp=cm[key].flatten()
                
                recall=tp/(tp+fn)
                precision=tp/(tp+fp)
                accurancy=(tp+tn)/(tn+fp+fn+tp)
                f1=2*tp/(2*tp+fp+fn)
                fbeta=(1+beta**2)*(precision*recall/(beta**2*precision+recall))
            
                print("Binary Confusion Matrix Report:{}".format(key).center(100,"-"))
                print("Recall : {:0>0.8f}".format(round(recall,8)).center(100," "))
                print("Precision : {:0>0.8f}".format(round(precision,8)).center(96," "))
                print("Accurancy : {:0>0.8f}".format(round(accurancy,8)).center(96," "))
                print("F1 : {:0>0.8f}".format(round(f1,8)).center(104," "))
                print("Fbeta(beta={:0>0.1f}) : {:0>0.8f}".format(beta,round(fbeta,8)).center(90," "))
                print("-".center(100,"-"))
            
        else:
            
            tn, fp, fn, tp=cm.flatten()

            recall=tp/(tp+fn)
            precision=tp/(tp+fp)
            accurancy=(tp+tn)/(tn+fp+fn+tp)
            f1=2*tp/(2*tp+fp+fn)
            fbeta=(1+beta**2)*(precision*recall/(beta**2*precision+recall))

            print("Binary Confusion Matrix Report".center(100,"-"))
            print("Recall : {:0>0.8f}".format(round(recall,8)).center(100," "))
            print("Precision : {:0>0.8f}".format(round(precision,8)).center(96," "))
            print("Accurancy : {:0>0.8f}".format(round(accurancy,8)).center(96," "))
            print("F1 : {:0>0.8f}".format(round(f1,8)).center(104," "))
            print("Fbeta(beta={:0>0.1f}) : {:0>0.8f}".format(beta,round(fbeta,8)).center(90," "))
            print("-".center(100,"-"))                   
            
            
    def _get_bestcut(self,y_true,y_pred,sample_weight=None,beta=1):
        
        precision,recall,thresholds=precision_recall_curve(y_true,y_pred,
                                                       sample_weight=sample_weight)
        precision=precision[:-1]
        recall=recall[:-1]
    
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fbeta=(1+beta**2)*(precision*recall/(beta**2*precision+recall))
            thre=thresholds[fbeta==np.nanmax(fbeta)][0]
            
        #print(np.nanmax(fbeta))
            
        return thre


    def _get_pltdt(self,y_true,y_pred,group=None,labels=None,sample_weight=None,normalize=None,sort_group=None):
        
        #check value and index at y_true,y_pred,group,sample_weight
        
        if group is not None:
            
            cm={
                g:confusion_matrix(y_true[group==g],y_pred[group==g],
                                   sample_weight=sample_weight[group==g] if sample_weight is not None else None,
                                   normalize=normalize) 
                for g in group.unique()
            }
    
            if normalize is not None:
    
                cm_raw={
                    g:confusion_matrix(y_true[group==g],y_pred[group==g],
                                       sample_weight=sample_weight[group==g] if sample_weight is not None else None,
                                       normalize=None) 
                    for g in group.unique()
                }    
    
            else:
    
    
                cm_raw=cm.copy()
            
        else:
            
            cm=confusion_matrix(y_true,y_pred,sample_weight=sample_weight,normalize=normalize)
            
            if normalize is not None:
                
                cm_raw=confusion_matrix(y_true,y_pred,sample_weight=sample_weight,normalize=None)
                
            else:
                        
                cm_raw=cm.copy()
            
        
        true_label=pd.Categorical([0,0,1,1],categories=[1,0]).map({0:0,1:1} if labels is None else {0:labels[0],1:labels[1]})    
        pred_label=pd.Categorical([0,1,0,1],categories=[0,1]).map({0:0,1:1} if labels is None else {0:labels[0],1:labels[1]})      
        
        
        if isinstance(cm,dict):
            
            dt_plot=pd.concat([pd.DataFrame(
                {
                 'true':true_label,
                 'pred':pred_label,
                 'count':cm[g].flatten(),
                 'textcolor':pd.Series(cm[g].flatten()>(cm[g].max()/2)).map({True:'white',False:'darkblue'})
                }
            ).assign(group=g) for g in cm])
            
            if sort_group is not None:
            
                dt_plot['group']=pd.Categorical(dt_plot['group'],categories=sort_group)
                
            else:
                
                dt_plot['group']=pd.Categorical(dt_plot['group'],categories=sorted(dt_plot['group'].unique()))    
            
        else:
            
            dt_plot=pd.DataFrame(
                {
                 'true':true_label,
                 'pred':pred_label,
                 'count':cm.flatten(),
                 'textcolor':pd.Series(cm.flatten()>(cm.max()/2)).map({True:'white',False:'darkblue'})
                }
            ).assign(group='data') 
            
            
        if normalize is not None:
                
            dt_plot['percent']=dt_plot['count'].apply(lambda x:str(np.round(100*x,2))+'%')
            
        return cm_raw,dt_plot


    def _plot_cm(self,dt_plot,figure_size=(5,5),normalize=None):
        
        if normalize is not None:
            
            fill_col = 'percent'
            
            if normalize=='true':
            
                fill_lab = 'Row%' 
                
                title='Confusion Matrix Normalized by Row#'
                
            elif normalize=='pred':
            
                fill_lab = 'Column%' 
                
                title='Confusion Matrix Normalized by Column#'
                    
            else:
                
                fill_lab = 'All%' 
                
                title='Confusion Matrix Normalized by Total#'
            
        else:
            
            fill_col = 'count'
            
            fill_lab = 'Count'
            
            title='Confusion Matrix'
        
        from plotnine import ggplot,aes,geom_tile,facet_wrap,ggtitle,labs,geom_text,theme,scale_fill_cmap,theme_bw,element_text
        
        
        if fill_col == 'count' and fill_lab == 'Count':
            
            dt_plot['count']=dt_plot['count'].apply(np.round,args=(2,))
        
        p=(ggplot(dt_plot, aes('pred','true', fill='count'))
             + geom_tile(aes(width=.95, height=.95))
             + facet_wrap('group')
             + ggtitle(title)
             + labs(x='Predicted labels',y='True labels',fill=fill_lab)
             + geom_text(aes(label=fill_col), size=10,color=dt_plot['textcolor'].tolist())  # modified
             + scale_fill_cmap('Blues')
             + theme_bw()
             + theme(figure_size=figure_size,
                     legend_position = 'right',
                     strip_text = element_text(size=12),
                     plot_title = element_text(size=14,color='darkblue'),
                     legend_title = element_text(size=10,color='darkblue'),
                     legend_text = element_text(color='darkblue'),
                     axis_text = element_text(size=10,color='darkblue',rotation=90),
                     axis_title = element_text(size=12,color='darkblue'))
            )
        
        return(p)