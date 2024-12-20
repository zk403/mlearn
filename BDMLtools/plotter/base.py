#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 23:08:38 2021

@author: zengke
"""

import pandas as pd
import numpy as np
from plotnine import __version__,ggplot,geom_density,geom_text,guides,theme_bw,theme,ggtitle,labs,scale_y_continuous,\
    scale_x_continuous,coord_fixed,aes,guide_legend,element_blank,geom_line,geom_segment,geom_point,annotate,geom_ribbon,geom_bar,geom_path,facet_wrap,element_text,geom_abline
import matplotlib.pyplot as plt
from joblib import Parallel,delayed
from scipy.stats import iqr
import statsmodels.api as sm

class BaseWoePlotter:   
    
    def _woe_plot(self,varbin,ptype,figure_size,n_jobs,verbose):
        
        #n_jobs=effective_n_jobs(n_jobs)   
                              
        p=Parallel(n_jobs=1,verbose=verbose)
        
        res=p(delayed(self._get_plot_single)(varbin[key],ptype,figure_size,False) for key in varbin)
        
        out={colname:fig for fig,colname in res}
        
        return out     
         

    def _get_bin(self,binx,ptype):
        
        if not binx.loc['missing','count']:
    
            binx=binx[binx.index!='missing']
    
        if not binx.loc['special','count']:
    
            binx=binx[binx.index!='special']
    
        binx=binx.reset_index()
        binx['rowid']=binx.index+1      
        
        if ptype == 'woe':
            
            binx['lineval_l']=binx['woe'].round(2).map(str)  
        
        elif ptype == 'badprob':
            
            binx['lineval_l']=binx['badprob'].mul(100).round(2).map(str)+'%' 

        binx['pos']=binx['good']/binx['count'].sum()
        binx['neg']=binx['bad']/binx['count'].sum()      
        
        
        return binx
    
    def _get_plot_single(self,binx,ptype,figure_size=None,show_plot=True):
    
        #dt transform
        binx=self._get_bin(binx,ptype)
                    
        
        binx_melt=pd.melt(binx,id_vars = ["bin","rowid"], value_vars =["pos", "neg"], 
                    value_name = "negpos").rename(columns={'variable':''})
        
        #title
        title_string = binx['variable'].iloc[0]+'(iv:{},ks:{})'.format(round(binx['total_iv'].iloc[0],4),round(binx['ks'].iloc[0],4))

        labels=(binx['count'].astype('str')+','+binx['count_distr'].mul(100).round(2).astype('str')+'%').tolist()

        if ptype == 'woe':         
            
            binx['woe_std']=binx['woe'].sub(binx['woe'].min()) 
            binx['count_distr_adj']=binx['count_distr'].mul(binx['woe_std'].max()/binx['count_distr'].max())
 
            #adjust max val of y-left axis
            y_right_max = binx['woe_std'].max()*1.2
        
            y_left_max = binx['count_distr'].max()*1.2
                
            y_max=max(y_right_max,y_left_max)
            
            if y_right_max>y_left_max:
                
                binx_melt['negpos']=binx_melt['negpos'].mul(binx['woe_std'].max()/binx['count_distr'].max())
            
            #print(binx[['variable','woe','badprob']])
            #base plot using ggplot2(no sec_axis function in plotnine - 0.8.0)
            figure=(ggplot() 
                + geom_bar(binx_melt,aes(x='bin',y='negpos',fill=''), stat="identity",show_legend = True)
                + geom_point(data=binx,mapping=aes(x='rowid',y='woe_std'), stat="identity",color='blue')
                + geom_abline(intercept=-binx['woe'].min(),slope=0,color='blue',linetype='dotted')
                + geom_path(data=binx,mapping=aes(x='rowid',y='woe_std'),color='blue')
                + scale_y_continuous(limits = [0, y_max],breaks=None)
                + theme_bw()
                + labs(x='',y='Bin count distribution',title=title_string)
                + theme(
                  figure_size=figure_size,
                  legend_position=(0.5,-0.15) if __version__!="0.12.4" else (0.5,0),legend_direction="horizontal",
                  plot_title = element_text(ha='left'),
                  legend_key_size=10)
                + geom_text(data=binx, mapping=aes(x ='rowid',y='woe_std',label='lineval_l'),va = 'bottom',color='blue')
                + geom_text(data=binx, mapping=aes(x='rowid',y='count_distr_adj',label=labels),va = 'bottom',color='black',alpha=0.6)
            ).draw()
        
            #add subplot(second axis) using matplotlib 
            ax1=figure.get_axes()[-1]
            ax2=ax1.twinx()
            ax2.set_ylabel('Weight of Evidence', color='blue')
            ax2.set_ylim(top=y_max)
            ax2.set_yticks([])
            ax2.tick_params(axis='y', colors='blue')             
                
        elif ptype == 'badprob':
                         
            #adjust max val of y-left axis
            y_right_max = np.ceil(binx['badprob'].max()*10)
            
            if y_right_max % 2 == 1: 
            
                y_right_max=y_right_max+1
            
            if y_right_max - binx['badprob'].max()*10 <= 0.3: 
            
                y_right_max = y_right_max+2
            
            #adjust max val of y-right axis    
            y_right_max = y_right_max/10
            
            if y_right_max>1 or y_right_max<=0 or y_right_max is np.nan or y_right_max is None: 
            
                y_right_max=1
            
            y_left_max = np.ceil(binx['count_distr'].max()*10)/10
            
            if y_left_max>1 or y_left_max<=0 or y_left_max is np.nan or y_left_max is None:
            
                y_left_max=1
                
            y_max=max(y_right_max,y_left_max)  
            
            #print(binx[['variable','woe','badprob']])
            
            figure=(ggplot() + geom_bar(binx_melt,aes(x='bin',y='negpos',fill=''), stat="identity",show_legend = True)
                + geom_point(data=binx,mapping=aes(x='rowid',y='badprob'), stat="identity",color='blue')
                + geom_path(data=binx,mapping=aes(x='rowid',y='badprob'),color='blue')
                + scale_y_continuous(limits = [0, y_max])
                + theme_bw()
                + labs(x='',y='Bin count distribution',title=title_string)
                + theme(
                  figure_size=figure_size,
                  legend_position=(0.5,-0.15) if __version__!="0.12.4" else (0.5,0),legend_direction="horizontal",
                  plot_title = element_text(ha='left'),
                  legend_key_size=10)
                + geom_text(data=binx, mapping=aes(x = 'rowid',y='badprob',label='lineval_l'),va = 'bottom',color='blue')
                + geom_text(data=binx, mapping=aes(x='rowid',y='count_distr',label=labels),va = 'bottom',color='black',alpha=0.6)
            ).draw()
        
            #add subplot(second axis) using matplotlib 
            ax1=figure.get_axes()[-1]
            ax2=ax1.twinx()
            ax2.set_ylabel('Bad probability', color='blue')
            ax2.set_ylim(top=y_max)
            ax2.set_yticks(np.arange(0, y_max, 0.2))
            ax2.tick_params(axis='y', colors='blue')                 
            
        else:
            
            raise ValueError("ptype in ('woe','badprob')")       
        
        if not show_plot:
            
            plt.close()
        
        return figure,binx['variable'].iloc[0]
    
    def _get_plot_single_group(self,binx_g,ptype,sort_column=None,figure_size=None,show_plot=True):
    
        #dt transform
        gs=np.unique([i[0] for i in binx_g.columns.tolist() if i[0] not in ['variable','bin']]).tolist()
        
        self._check_plot_sort(sort_column,gs)
    
        binx_g_h=pd.concat({g:self._get_bin(binx_g[g].assign(g=g),ptype=ptype) for g in gs}) 
                
        binx_g_melt=pd.concat({g:pd.melt(self._get_bin(binx_g[g],ptype),
                      id_vars = ["bin","rowid"], 
                      value_vars =["pos", "neg"], 
                      value_name = "negpos").rename(columns={'variable':''}).assign(g=g) for g in gs})
        
        if sort_column:
    
            binx_g_h['g']=binx_g_h['g'].astype('category').cat.reorder_categories(sort_column)
            binx_g_melt['g']=binx_g_melt['g'].astype('category').cat.reorder_categories(sort_column)
    
        #title
        binx_g_ivks=binx_g_h[['total_iv','ks']].droplevel(1)
        binx_g_ivks=binx_g_ivks.loc[~binx_g_ivks.index.duplicated(),:]
        
        iv_d=binx_g_ivks['total_iv'].to_dict()
        ks_d=binx_g_ivks['ks'].to_dict()
    
        keys=sort_column if sort_column else iv_d.keys()
        
        #title
        title_string=binx_g_h['variable'].iloc[0]+':'+(',').join(['{}(iv:{},ks:{})'.format(key,round(iv_d[key],4),round(ks_d[key],4)) for key in keys])
    
        if ptype == 'woe':  
        
            binx_g_h['woe_std']=binx_g_h['woe'].sub(binx_g_h['woe'].min()) 
            binx_g_h['count_distr_adj']=binx_g_h['count_distr'].mul(binx_g_h['woe_std'].max()/binx_g_h['count_distr'].max())

    
            y_right_max = binx_g_h['woe_std'].max()*1.2
        
            y_left_max = binx_g_h['count_distr'].max()*1.2    
    
            if y_right_max>y_left_max:
                
                binx_g_melt['negpos']=binx_g_melt['negpos'].mul(binx_g_h['woe_std'].max()/binx_g_h['count_distr'].max())
                  
            y_max=max(y_right_max,y_left_max)
            
            labels=(binx_g_h['count'].astype('str')+','+binx_g_h['count_distr'].mul(100).round(2).astype('str')+'%').tolist()        
            
            #print(binx_g_h[['variable','woe','badprob']])
            #base plot using ggplot2(no sec_axis function in plotnine - 0.8.0)
            figure=(ggplot() + geom_bar(binx_g_melt,aes(x='bin',y='negpos',fill=''), stat="identity",show_legend = True)
                + geom_point(data=binx_g_h,mapping=aes(x='rowid',y='woe_std'), stat="identity",color='blue')
                + geom_path(data=binx_g_h,mapping=aes(x='rowid',y='woe_std'),color='blue')
                + geom_abline(intercept=-binx_g_h['woe'].min(),slope=0,color='blue',linetype='dotted')
                + facet_wrap(('g'))           
                + scale_y_continuous(limits = [0, y_max],breaks=None)
                + theme_bw()
                + labs(x='',y='Bin count distribution',title=title_string)
                + theme(
                  figure_size=figure_size,
                  legend_position=(0.5,-0.15) if __version__!="0.12.4" else (0.5,0),legend_direction="horizontal",
                  plot_title = element_text(ha='left'),
                  legend_key_size=10)
                + geom_text(data=binx_g_h, mapping=aes(x ='rowid',y='woe_std',label='lineval_l'),va = 'bottom',color='blue')
                + geom_text(data=binx_g_h, mapping=aes(x='rowid',y='count_distr_adj',label=labels),va = 'bottom',color='black',alpha=0.6)
            ).draw()
        
            #add subplot(second axis) using matplotlib 
            ax1=figure.get_axes()[-1]
            ax2=ax1.twinx()
            ax2.set_ylabel('Weight of Evidence', color='blue')
            ax2.set_ylim(top=y_max)
            ax2.set_yticks([])
            ax2.tick_params(axis='y', colors='blue')
            
        elif ptype == 'badprob':  
            
            #adjust max val of y-left axis
            y_right_max = np.ceil(binx_g_h['badprob'].max()*10)
        
            if y_right_max % 2 == 1: 
        
                y_right_max=y_right_max+1
        
            if y_right_max - binx_g_h['badprob'].max()*10 <= 0.3: 
        
                y_right_max = y_right_max+2
        
            #adjust max val of y-right axis    
            y_right_max = y_right_max/10
        
            if y_right_max>1 or y_right_max<=0 or y_right_max is np.nan or y_right_max is None: 
        
                y_right_max=1
        
            y_left_max = np.ceil(binx_g_h['count_distr'].max()*10)/10
        
            if y_left_max>1 or y_left_max<=0 or y_left_max is np.nan or y_left_max is None:
        
                y_left_max=1
            
            y_max=max(y_right_max,y_left_max)
            
            labels=(binx_g_h['count'].astype('str')+','+binx_g_h['count_distr'].mul(100).round(2).astype('str')+'%').tolist()
            
            #print(binx_g_h[['variable','woe','badprob']])
            #base plot using ggplot2(no sec_axis function in plotnine - 0.8.0)
            figure=(ggplot() + geom_bar(binx_g_melt,aes(x='bin',y='negpos',fill=''), stat="identity",show_legend = True)
                + geom_point(data=binx_g_h,mapping=aes(x='rowid',y='badprob'), stat="identity",color='blue')
                + geom_path(data=binx_g_h,mapping=aes(x='rowid',y='badprob'),color='blue')
                + facet_wrap(('g'))
                + scale_y_continuous(limits = [0, y_max])
                + theme_bw()
                + labs(x='',y='Bin count distribution',title=title_string)
                + theme(
                  figure_size=figure_size,
                  legend_position=(0.5,-0.15) if __version__!="0.12.4" else (0.5,0),legend_direction="horizontal",
                  plot_title = element_text(ha='left'),
                  legend_key_size=10)
                + geom_text(data=binx_g_h, mapping=aes(x = 'rowid',y='badprob',label='lineval_l'),va = 'bottom',color='blue')
                + geom_text(data=binx_g_h, mapping=aes(x='rowid',y='count_distr',label=labels),va = 'bottom',color='black',alpha=0.6)
            ).draw()
        
            #add subplot(second axis) using matplotlib 
            ax1=figure.get_axes()[-1]
            ax2=ax1.twinx()
            ax2.set_ylabel('Bad probability', color='blue')
            ax2.set_ylim(top=y_max)
            ax2.set_yticks(np.arange(0, y_max, 0.2))
            ax2.tick_params(axis='y', colors='blue')
            
        else:                
            raise ValueError("ptype in ('woe','badprob')")             
        
        if not show_plot:
            
            plt.close()     
        
        return figure,binx_g_h['variable'].iloc[0]
    
    
    def _check_plot_sort(self,sort_column,gs):
        
        if sort_column:
            
            if isinstance(sort_column,list):
                
            
                if set(gs)!=set(sort_column):    
                    
                    
                    raise ValueError("val in sort_column not equal to val in dat[col]")
                    
            else:
                    
                raise ValueError("sort_column is list")                      
                           
                
class BaseEvalFuns:
    
    def _R_pretty(self,low, high, n):
        '''
        pretty breakpoints, the same as pretty function in R
        
        Params
        ------
        low: minimal value 
        low: maximal value 
        n: number of intervals
        
        Returns
        ------
        numpy.ndarray
            returns a breakpoints array
        '''
        # nicenumber
        def nicenumber(x):
            exp = np.floor(np.log10(abs(x)))
            f   = abs(x) / 10**exp
            if f < 1.5:
                nf = 1.
            elif f < 3.:
                nf = 2.
            elif f < 7.:
                nf = 5.
            else:
                nf = 10.
            return np.sign(x) * nf * 10.**exp
        
        # pretty breakpoints
        d     = abs(nicenumber((high-low)/(n-1)))
        miny  = np.floor(low  / d) * d
        maxy  = np.ceil (high / d) * d
        return np.arange(miny, maxy+0.5*d, d).round(1).tolist()
       
    
    def _compute_density(self,x,weight=None,kernel='gau',bw='nrd0'):
        '''
        estimate density for density plot
    
        re-write form plotnine-0.8.0
    
        similar to R stat::density
    
        Params
        ------
        x: array_like  
        range_x: (minimal value,maximal value) 
        weight
    
        Returns
        ------
        numpy.ndarray
            returns a breakpoints array
        '''
    
        x = np.asarray(x, dtype=float)
        not_nan = ~np.isnan(x)
        x = x[not_nan]
        n = len(x)
        #n = len(x)
    
        # kde is computed efficiently using fft. But the fft does
        # not support weights and is only available with the
        # gaussian kernel. When weights are relevant we
        # turn off the fft.
    
        if weight is None:
    
            if kernel != 'gau':
                weight = np.ones(n) / n
    
        else:
    
            weight = np.asarray(weight, dtype=float)
    
    
        if kernel == 'gau' and weight is None:
            fft = True
        else:
            fft = False
            
        if bw == 'nrd0':
            bw = self.nrd0(x)
    
        kde = sm.nonparametric.KDEUnivariate(x)
        kde.fit(
            kernel=kernel,
            bw=bw, 
            fft=fft,
            weights=weight,
            adjust=1,
        )
    
        x2 = np.linspace(np.nanmin(x), np.nanmax(x), 512) 
    
        try:
            y = kde.evaluate(x2)
    
            if np.isscalar(y) and np.isnan(y):
    
                raise ValueError('kde.evaluate returned nan')
    
        except ValueError:
    
            y = []
    
            for _x in x2:
    
                result = kde.evaluate(_x)
    
                try:
                    y.append(result[0])
                except TypeError:
                    y.append(result)
    
        y = np.asarray(y)
    
        # Evaluations outside the kernel domain return np.nan,
        # these values and corresponding x2s are dropped.
        # The kernel domain is defined by the values in x, but
        # the evaluated values in x2 could have a much wider range.
        not_nan = ~np.isnan(y)
        x2 = x2[not_nan]
        y = y[not_nan]
    
    #     return pd.DataFrame({'x': x2,
    #                          'density': y,
    #                          'scaled': y / np.max(y) if len(y) else [],
    #                          'count': y * n,
    #                          'n': n})['density']    
    
        return pd.Series(y)
    
    
    def nrd0(self,x):
        """
        Port of R stats::bw.nrd0
        This is equivalent to statsmodels silverman when x has more than
        1 unique value. It can never give a zero bandwidth.
        
        form plotnine-0.8.0
        
        Parameters
        ----------
        x : array_like
            Values whose density is to be estimated
        Returns
        -------
        out : float
            Bandwidth of x
        """
        n = len(x)
        if n < 1:
            raise ValueError(
                "Need at leat 2 data points to compute the nrd0 bandwidth."
            )
    
        std = np.std(x, ddof=1)
        
        std_estimate = iqr(x)/1.349
        
        low_std = np.min((std, std_estimate))
        
        if low_std == 0:
            low_std = std_estimate or np.abs(np.asarray(x)[0]) or 1
            
        return 0.9 * low_std * (n ** -0.2)                            
                
                
                
class BaseEvalData:    
    
    def _get_df(self,y_pred,y_true,group):
        
        
        if group is not None:
            
            dt_df=pd.concat(
                 [
                    pd.Series(y_pred,name='pred'),
                    pd.Series(y_true,name='label',dtype='category'),
                    pd.Series(group,name='group',dtype='category')               
                 ],axis=1
                ).dropna()
            
        else:
            
            dt_df=pd.concat(
                 [
                    pd.Series(y_pred,name='pred'),
                    pd.Series(y_true,name='label',dtype='category')             
                 ],axis=1
                ).assign(group='data').dropna()
            
        return dt_df
    
    def _get_dfev(self,dt_df,sample_weight=None,groupnum=None): 
        
        dt_df=dt_df.copy()
        
        ws=np.ones(len(dt_df)) if sample_weight is None else sample_weight
        
        if self.pred_desc:
    
            dt_df['pred']=-dt_df['pred']
        
        dt_ev=dt_df.assign(nP=ws*dt_df['label'].astype('float'),
                           nN=ws*dt_df['label'].astype('float').map({0:1,1:0})).groupby('pred',observed=False)[['nP','nN']].sum().reset_index()
        
        if groupnum is not None:
            
            if groupnum<=len(dt_df):
    
                pred2=np.ceil(dt_ev[['nP','nN']].sum(1).cumsum()/(np.sum(ws)/groupnum)).rename('pred2')
                dt_ev=dt_ev.groupby(pred2,observed=False)[['nP','nN']].sum().join(
                    dt_ev.groupby(pred2,observed=False)['pred'].mean() 
                ).reset_index(drop=True)[['pred','nP','nN']]

    
        dt_ev=dt_ev.sort_values('pred',ascending=False)
        dt_ev['tp']=dt_ev['nP'].cumsum()
        dt_ev['fp']=dt_ev['nN'].cumsum()
        dt_ev['fn']=dt_ev['nP'].sum()-dt_ev['tp']
        dt_ev['tn']=dt_ev['nN'].sum()-dt_ev['fp']
        dt_ev['cumpop']=dt_ev[['tp','fp']].sum(1)/(dt_ev['nP'].sum()+dt_ev['nN'].sum())    
        dt_ev['tpr']=dt_ev['tp']/dt_ev['nP'].sum() #tpr/recall
        dt_ev['fpr']=dt_ev['fp']/dt_ev['nN'].sum()#fpr
        dt_ev['precision']=dt_ev['tp'].div(dt_ev['tp'].add(dt_ev['fp'])) #precision
        dt_ev['recall']=dt_ev['tp']/dt_ev['nP'].sum() #recall/tpr
        
        return dt_ev
    
    def _get_dfev_dict(self,dt_df,sample_weight):
        

        groupnum=1000 if len(dt_df)>1000 else None
        
        g_dict=dt_df.groupby('group',observed=False).groups

        g_dtev={group:self._get_dfev(dt_df.loc[g_dict[group]],
                                     sample_weight=sample_weight[dt_df.loc[g_dict[group]].index] if sample_weight is not None else None,
                                     groupnum=groupnum
                                     ) for group in g_dict}
        
        return g_dtev
    
    def _get_df_density(self,dt_df,sample_weight=None):
    
        dt_df=dt_df.copy()
        
        dt_df['ws']=np.ones(len(dt_df)) if sample_weight is None else sample_weight
        
        groupnum=1000 if len(dt_df)>1000 else None
        
        def get_df_density_g(dt_df,group,groupnum=None):
        
            dt_df=dt_df.copy()
    
            dt_ev_density=dt_df.groupby('pred',observed=False)['ws'].sum().reset_index()
    
            if groupnum is not None:
    
                if groupnum<=len(dt_df):
    
                    pred2=np.ceil(dt_ev_density['ws'].cumsum()/(dt_ev_density['ws'].sum()/groupnum)).rename('pred2')
    
                    out=dt_df[dt_df['pred'].isin(dt_ev_density.groupby(pred2,observed=False)['pred'].max())][['pred','label','ws']]\
                            .drop_duplicates(subset='pred').assign(group=group)
    
                else:
    
                    out=dt_df
    
            else:
    
                out=dt_df
    
            return out
        
        g_dict=dt_df.groupby('group',observed=False).groups
    
        g_dtdens=[get_df_density_g(dt_df.loc[g_dict[group]],group,groupnum=groupnum) for group in g_dict]
        
        return pd.concat(g_dtdens,ignore_index=True)

    
    def _get_dt_ks(self,dt_ev,group):

        dt_ks=pd.concat(
           [
             dt_ev[['cumpop','pred']],
             dt_ev['tpr'].rename('cumpos'),
             dt_ev['fpr'].rename('cumneg')
           ],axis=1
        )
        
        dt_ks['ks']=dt_ks['cumpos'].sub(dt_ks['cumneg']).abs()
        
        max_ks=dt_ks.loc[dt_ks['ks'].eq(dt_ks['ks'].max())].head(1)
    
        dt_ks['group']='{},KS={},\np={},({},{})'.format(group,max_ks['ks'].round(4).values[0],
                                       -max_ks['pred'].round(2).values[0] if self.pred_desc else max_ks['pred'].round(2).values[0],
                                       max_ks['cumpop'].round(2).values[0],
                                       max_ks['ks'].round(2).values[0])
        return dt_ks 
    
    
    def _get_dt_lift(self,dt_ev,group):
        
        dt_lift=dt_ev[['cumpop','pred']].assign(
            lift = dt_ev['precision']/(dt_ev['tp'].add(dt_ev['fn']).div(dt_ev[['tp','fp','tn','fn']].sum(1)))
        )

        dt_lift['group']=group
        
        return dt_lift 
    
    def _get_dt_roc(self,dt_ev,group):
        
        dt_roc=dt_ev[['pred','fpr','tpr']].assign(co=(dt_ev['tpr']-dt_ev['fpr'])**2/2).sort_values(['fpr','tpr'])
        
        auc=np.sum(dt_roc['tpr'].add(dt_roc['tpr'].shift(1).fillna(0)).div(2)*dt_roc['fpr'].sub(dt_roc['fpr'].shift(1).fillna(0)))   
     
        max_cutoff=dt_roc.loc[dt_roc['co'].eq(dt_roc['co'].max())].head(1)
    
        dt_roc['group']='{},AUC={},\np={},({},{})'.format(group,round(auc,4),
                                       -max_cutoff['pred'].round(2).values[0] if self.pred_desc else max_cutoff['pred'].round(2).values[0],
                                       max_cutoff['fpr'].round(2).values[0],
                                       max_cutoff['tpr'].round(2).values[0])
        
        return dt_roc
    
    def _get_dt_lz(self,dt_ev,group):
    
        dt_lz=dt_ev[['cumpop','nP','fpr','tpr']].assign(
            cumposrate=dt_ev['nP'].cumsum()/dt_ev['nP'].sum()
            ).sort_values(['fpr','tpr'])

        auc=np.sum(dt_lz['tpr'].add(dt_lz['tpr'].shift(1).fillna(0)).div(2)*dt_lz['fpr'].sub(dt_lz['fpr'].shift(1).fillna(0)))   
        gini=2*auc-1
        
        dt_lz['group']='{},GINI={}'.format(group,round(gini,4))
        
        return dt_lz
    
    def _get_dt_gain(self,dt_ev,group):
    
        dt_gain=dt_ev[['cumpop','precision']].copy()
        dt_gain['group']=group

        return dt_gain
    
    def _get_dt_pr(self,dt_ev,group):
        
        dt_pr=dt_ev[['recall','precision']].copy()
        dt_pr['group']=group

        return dt_pr  
    
    def _get_dt_f1(self,dt_ev,group):
        
        dt_f1=dt_ev[['cumpop','pred']].assign(
            f1=dt_ev['tp'].mul(2).div(dt_ev['tp'].mul(2)+dt_ev['fp']+dt_ev['fn'])
        )
        
        max_f1=dt_f1.loc[dt_f1['f1'].eq(dt_f1['f1'].max())].head(1)
    
        dt_f1['group']='{},\np={},({},{})'.format(group,
                                                  -max_f1['pred'].round(2).values[0] if self.pred_desc else max_f1['pred'].round(2).values[0],
                                                  max_f1['cumpop'].round(2).values[0],
                                                  max_f1['f1'].round(2).values[0])

        return dt_f1 
    
    def _get_dt_sloping(self,dt_ev,group,n=10):

        dt_sloping=dt_ev[['cumpop','pred','nP','nN']].copy()
        dt_sloping['n']=dt_ev[['nP','nN']].sum(1)
        dt_sloping['cumpop_bin']=pd.qcut(-dt_sloping['cumpop'],n,labels=np.linspace(0,1,n+1)[:-1]).astype('float')
        
        dt_sloping_m=pd.concat(
            [
                dt_sloping.groupby('cumpop_bin')['nP'].sum().div(dt_sloping.groupby('cumpop_bin',)['n'].sum()).rename('true'),
                dt_sloping.groupby('cumpop_bin')['pred'].mean().rename('pred')
                ],axis=1
            ).assign(group=group)
        
        return dt_sloping_m


class BaseEvalPlotter(BaseEvalData,BaseEvalFuns):    
          
    def _plot_density(self,dt_density,figure_size,title=None):
        
        if dt_density['pred'].mean()<-1:
    
            dt_density['pred']=dt_density['pred'].abs()
            
        max_pred = dt_density['pred'].max()
        min_pred = dt_density['pred'].min()
        
        max_density_by_datset_label=dt_density.groupby(['group','label'],observed=False)['pred'].apply(lambda x:self._compute_density(x,dt_density['ws'][x.index])).rename('dens').droplevel(2).reset_index()       
        
        max_density=np.ceil(max_density_by_datset_label['dens'].max())
        
        if max_density==1:
            
            max_density=max_density_by_datset_label['dens'].max()+max_density_by_datset_label['dens'].max()/10
        
        coord_label=max_density_by_datset_label.groupby(['label'],observed=False)[['dens']].max().join(
            dt_density.groupby('label',observed=False)['pred'].median()
        )
        
        fig=(ggplot(data = dt_density) +
            geom_density(aes(x='pred',linetype='label',color='factor(group)'),fill='gray', alpha=0.1,show_legend = True) +
            geom_text(coord_label, aes(x='pred', y='dens', label=coord_label.index.map({0:'Neg',1:'Pos'}))) +
            guides(linetype=None, color=guide_legend(title='')) +
            ggtitle(title+' Density' if title else 'Density') +
            theme_bw() +
            theme(figure_size=figure_size,
                  legend_position=(0.9,0.9) if __version__!='0.12.4' else (0.8,0.8),
                  legend_direction='',
                  #legend_justification=(1,1),
                  legend_background=element_blank(),
                  plot_title = element_text(ha='left'),
                  legend_key=element_blank()) +
            labs(x = "Prediction", y = "Density") +
            scale_y_continuous(labels=self._R_pretty(0,max_density,5), breaks=self._R_pretty(0,max_density,5)) +
            scale_x_continuous(labels=self._R_pretty(min_pred,max_pred,5), breaks=self._R_pretty(min_pred,max_pred,5)) +
            coord_fixed(ratio = (max_pred-min_pred)/(max_density), xlim = (min_pred,max_pred), 
                ylim = (0,max_density), expand = False)
        )     
        
        if __version__!='0.12.4':
        
            fig.show()
            
        else:
                
            print(fig)
            
        return fig.draw()
    
    
    def _plot_ks(self,g_dtev,figure_size,title=None):

        dt_ks=pd.concat([self._get_dt_ks(g_dtev[group],group) for group in g_dtev],axis=0,ignore_index=True)
        dtks=dt_ks[dt_ks['ks']==dt_ks.groupby('group',observed=False)['ks'].transform('max')].drop_duplicates(subset='group')
        
        x_posi = 0.95 if dt_ks['cumpos'].mean()<dt_ks['cumneg'].mean() else 0.4 
        x_neg = 0.4 if dt_ks['cumpos'].mean()<dt_ks['cumneg'].mean() else 0.95
    
        fig=(ggplot(dt_ks,aes(x='cumpop'))+
            geom_line(aes(y='cumneg', color='group'), linetype='dotted')+
            geom_line(aes(y='cumpos', color='group'), linetype='dotted')+
            geom_line(aes(y='ks', color='group')) +
            geom_segment(dtks, aes(x = 'cumpop', y = 0, xend = 'cumpop', yend = 'ks', color='group',label='group'), linetype = "dashed") +
            geom_point(dtks, aes(x='cumpop', y='ks'), color='red') +
            annotate("text", x=x_posi, y=0.7,  label="Pos", colour = "gray") +
            annotate("text", x=x_neg, y=0.7, label="Neg", colour = "gray") +
            ggtitle(title+' K-S' if title else 'K-S') +
            theme_bw() +
            theme(figure_size=figure_size,
                  legend_position=(0.1,0.9) if __version__!='0.12.4' else (0.26,0.8),
                  legend_direction='',
                  #legend_justification=(0,1),
                  legend_background=element_blank(),
                  legend_key=element_blank(),
                  plot_title = element_text(ha='left'),
                  #panel_border=element_rect(color="black",size=1.5)
                  #legend_key_size = unit(1.5, 'lines')
                 ) +
            guides(color=guide_legend(title='')) +
            labs(x = "% of population", y = "% of total Neg/Pos") +
            scale_y_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) +
            scale_x_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) +
            coord_fixed(xlim = (0,1), ylim = (0,1), expand = False)
        )
        
        if __version__!='0.12.4':
        
            fig.show()   
            
        else:
                
            print(fig)
            
        return fig.draw()  
    
    def _plot_lift(self,g_dtev,figure_size,title=None):

        dt_lift=pd.concat([self._get_dt_lift(g_dtev[group],group) for group in g_dtev],axis=0,ignore_index=True)

        max_lift = np.ceil(dt_lift['lift'].max())
        
        if max_lift==1:
            
            max_lift=dt_lift['lift'].max()

        legend_xposition=0.1 if __version__!='0.12.4' else 0.2

        if dt_lift.query("cumpop<0.1")['lift'].mean()>dt_lift.query("cumpop>0.9")['lift'].mean():

            legend_xposition=0.9 if __version__!='0.12.4' else 0.8

        fig=(ggplot(dt_lift, aes(x='cumpop', color = 'factor(group)')) +
            geom_line(aes(y = 'lift'), na_rm = True) +
            ggtitle(title+' Lift' if title else 'Lift') +
            theme_bw() +
            theme(legend_position=(legend_xposition,0.9) if __version__!='0.12.4' else (legend_xposition,0.8),
                  #legend.justification=c(legend_xposition,1),
                  legend_direction='',
                  figure_size=figure_size,
                  legend_background=element_blank(),
                  plot_title = element_text(ha='left'),
                  legend_key=element_blank()) +
            guides(color=guide_legend(title='')) +             
            labs(x = "% of population", y = "Lift") +
            scale_y_continuous(labels=self._R_pretty(0,max_lift,5), breaks=self._R_pretty(0,max_lift,5)) +
            scale_x_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) +
            coord_fixed(ratio = 1/(max_lift-1) if max_lift!=1 else 1,xlim = (0,1), ylim = (1,max_lift), expand = False) 
            )
        
        if __version__!='0.12.4':
        
            fig.show()   
            
        else:
                
            print(fig)    
            
        return fig.draw()  
    
    
    def _plot_gain(self,g_dtev,figure_size,title=None):

        dt_gain=pd.concat([self._get_dt_gain(g_dtev[group],group) for group in g_dtev],axis=0,ignore_index=True)
        
        legend_xposition = 0.1 if __version__!='0.12.4' else 0.2

        if dt_gain[dt_gain['cumpop']<0.1]['precision'].mean()>dt_gain[dt_gain['cumpop']>0.9]['precision'].mean():

            legend_xposition=0.9 if __version__!='0.12.4' else 0.8

        fig=(ggplot(dt_gain, aes(x='cumpop', color = 'factor(group)')) +
                geom_line(aes(y = 'precision'), na_rm = True) +
                ggtitle(title+' Gain' if title else 'Gain') +
                theme_bw() +
                theme(figure_size=figure_size,
                      legend_direction='',
                      legend_position=(legend_xposition,0.9) if __version__!='0.12.4' else (legend_xposition,0.8),
                      legend_background=element_blank(),
                      plot_title = element_text(ha='left'),
                      legend_key=element_blank()) +
                guides(color=guide_legend(title='')) +
                labs(x = "% of population", y = "Precision / PPV") +
                scale_y_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) +
                scale_x_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) +
                coord_fixed(xlim = (0,1), ylim = (0,1), expand = False)
            )
        
        if __version__!='0.12.4':
        
            fig.show()  
            
        else:
                
            print(fig)    
            
        return fig.draw()  
    
    def _plot_roc(self,g_dtev,figure_size,title=None):
        
        dt_roc=pd.concat([self._get_dt_roc(g_dtev[group],group) for group in g_dtev],axis=0,ignore_index=True)
        
        dt_cut=dt_roc[dt_roc['co']==dt_roc.groupby('group',observed=False)['co'].transform('max')].drop_duplicates(subset='group')
        
        
        fig=(ggplot(dt_roc, aes(x='fpr')) +
            geom_line(aes(y='tpr', color='group')) +
            geom_line(aes(y='fpr'), linetype = "dashed", colour="gray") +
            geom_ribbon(aes(ymin=0, ymax='tpr', fill='group'), alpha=0.1) +
            geom_point(dt_cut, aes(x='fpr', y='tpr'), color='red') +
            # geom_text(dt_cut, aes(x=FPR, y=TPR, label=oc, color=datset), vjust=1) +
            # geom_segment(aes(x=0, y=0, xend=1, yend=1), linetype = "dashed", colour="red") +
            ggtitle(title+' ROC' if title else 'ROC') +
            theme_bw() +
            theme(legend_position=(0.8,0.2) if __version__!='0.12.4' else (0.7,0.2),
                  legend_direction='',
                  figure_size=figure_size,
                  #legend_justification=c(1,0),
                  legend_background=element_blank(),
                  legend_key=element_blank(),
                  plot_title = element_text(ha='left'),
                  #legend_key_size = unit(1.5, 'lines')
                 ) +
            guides(color=guide_legend(title=''), fill=False) +  
            labs(x = "1-Specificity / FPR", y = "Sensitivity / TPR") +
            scale_y_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) +
            scale_x_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) +
            coord_fixed(xlim = (0,1), ylim = (0,1), expand = False)
            )
        
        if __version__!='0.12.4':
        
            fig.show()   
            
        else:
                    
            print(fig)
            
        return fig.draw()  
    
    
    def _plot_lz(self,g_dtev,figure_size,title=None):
        
        dt_lz=pd.concat([self._get_dt_lz(g_dtev[group],group) for group in g_dtev],axis=0,ignore_index=True)
        
        fig=(ggplot(dt_lz, aes(x='cumpop')) +
            geom_line(aes(y='cumposrate', color='group')) +
            geom_line(aes(y='cumpop'), linetype = "dashed", colour="gray") +
            # geom_segment(aes(x=0, y=0, xend=1, yend=1), linetype = "dashed", colour="red") +
            geom_ribbon(aes(ymin='cumpop', ymax='cumposrate', fill='group'), alpha=0.1) +
            ggtitle(title+' Lorenz' if title else 'Lorenz') +
            theme_bw() +
            theme(figure_size=figure_size,
                  legend_position=(0.1,0.9) if __version__!='0.12.4' else (0.2,0.8),
                  legend_direction='',
                  #legend_justification=c(0,1),
                  legend_background=element_blank(),
                  plot_title = element_text(ha='left'),
                  legend_key=element_blank()) +
            guides(color=guide_legend(title=''), fill=False)+
            labs(x = "% of population", y = "% of total positive") +
            scale_y_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) +
            scale_x_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) +
            coord_fixed(xlim = (0,1), ylim = (0,1), expand = False)
            )
        
        if __version__!='0.12.4':
        
            fig.show()   
            
        else:
                    
            print(fig)
            
        return fig.draw()  
        
    
    def _plot_pr(self,g_dtev,figure_size,title=None):

        dt_pr=pd.concat([self._get_dt_pr(g_dtev[group],group) for group in g_dtev],axis=0,ignore_index=True)

        fig=(ggplot(dt_pr) +
                geom_line(aes(x='recall', y='precision', color='factor(group)'), na_rm = True) +
                geom_line(aes(x='recall', y='recall'), na_rm = True, linetype = "dashed", colour="gray") +
                ggtitle(title+' P-R' if title else 'P-R') +
                theme_bw() +
                theme(figure_size=figure_size,
                      legend_position=(0.1,0.1) if __version__!='0.12.4' else (0.2,0.2),
                      legend_background=element_blank(),
                      plot_title = element_text(ha='left'),
                      legend_direction='',
                      legend_key=element_blank()) +
                guides(color=guide_legend(title='')) +
                labs(x = "Recall / TPR", y = "Precision / PPV") +
                scale_y_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) +
                scale_x_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) +
                coord_fixed(xlim = (0,1), ylim = (0,1), expand = False)
            )
        
        if __version__!='0.12.4':
        
            fig.show()   
            
        else:
            
            print(fig)
            
        return fig.draw()  

            
    def _plot_f1(self,g_dtev,figure_size,title=None):

        dt_f1=pd.concat([self._get_dt_f1(g_dtev[group],group) for group in g_dtev],axis=0,ignore_index=True)
        
        dtf1=dt_f1[dt_f1['f1']==dt_f1.groupby('group',observed=False)['f1'].transform('max')].drop_duplicates(subset='group')
  
        fig=(ggplot(dt_f1, aes(x='cumpop')) +
                geom_line(aes(y='f1', color='group'), na_rm = True) +
                geom_point(dtf1, aes(x='cumpop', y='f1'), color='red') +
                geom_segment(dtf1, aes(x = 'cumpop', y = 0, xend = 'cumpop', yend = 'f1', color='group'), linetype = "dashed") +
                ggtitle(title+' F1' if title else 'F1') +
                theme_bw() +
                theme(figure_size=figure_size,
                      legend_position=(0.8,0.2) if __version__!='0.12.4' else (0.7,0.2),      
                      legend_direction='',
                      plot_title = element_text(ha='left'),
                      legend_background=element_blank(),
                      legend_key=element_blank()) +
                guides(color=guide_legend(title=''), fill=False)+
                labs(x = "% of population", y = 'F1') +
                scale_y_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) +
                scale_x_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) +
                coord_fixed(xlim = (0,1), ylim = (0,1), expand = False)
            )
        
        if __version__!='0.12.4':
        
            fig.show()   
            
        else:
                
            print(fig)
            
        return fig.draw()  
    
    
    def _plot_sloping(self,g_dtev,figure_size,title=None,n=10):

        dt_sloping=pd.concat([self._get_dt_sloping(g_dtev[group],group,n) for group in g_dtev],axis=0,ignore_index=False).reset_index()               
        
        fig=(
           ggplot(dt_sloping, aes(x='cumpop_bin',color='factor(group)')) +
            geom_point(aes(y = 'true'), na_rm = True) +
            geom_line(aes(y = 'true'), na_rm = True) +
            geom_point(aes(y = 'pred'), na_rm = True)  +
            geom_line(aes(y = 'pred'), na_rm = True,linetype='dotted') +
            guides(color = guide_legend(title = "Group")) +
            ggtitle(title+' Sloping' if title else 'Sloping') +
            theme_bw() +
            theme(
                  legend_position=(0.1,1) if __version__!='0.12.4' else (0.2,0.8),
                  legend_direction='',
                  figure_size=figure_size,
                  legend_background=element_blank(),
                  plot_title = element_text(ha='left'),
                  legend_key=element_blank()) +
            guides(color=guide_legend(title='')) +             
            labs(x = "% of population", y = "Fraction of Positives") +
            scale_x_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) 
            )
        
        if __version__!='0.12.4':
        
            fig.show()
            
        else:  
            
            print(fig)
            
        return fig.draw()
                
    def _plot_calibration(self,g_dtev,figure_size,title=None,n=10):

        dt_sloping=pd.concat([self._get_dt_sloping(g_dtev[group],group,n) for group in g_dtev],axis=0,ignore_index=False).reset_index()               
        
        fig=(
           ggplot(dt_sloping, aes(x='pred',y='true',color='factor(group)')) +
            geom_line(na_rm = True)+
            geom_line(aes(x='pred',y='pred'), linetype = "dashed", colour="gray")+
            geom_point(na_rm = True)  +
            guides(color = guide_legend(title = "group")) +
            ggtitle(title+' Calibration Plot' if title else 'Calibration Plot') +
            theme_bw() +
            theme(
                  legend_position=(0.1,1) if __version__!='0.12.4' else (0.2,0.8),
                  legend_direction='',
                  figure_size=figure_size,
                  legend_background=element_blank(),
                  plot_title = element_text(ha='left'),
                  legend_key=element_blank()) +
            guides(color=guide_legend(title='')) +             
            labs(x = "Mean Predicted Probability", y = "Fraction of Positives") +
            scale_x_continuous(labels=self._R_pretty(0,1,5), breaks=self._R_pretty(0,1,5)) 
            )
        
        if __version__!='0.12.4':
        
            fig.show()   
            
        else:
                
            print(fig)
            
        return fig.draw() 