import pandas as pd
import numpy as np
from plotnine import ggplot,geom_bar,geom_point,aes,geom_path,geom_text,facet_wrap,theme,theme_bw,scale_y_continuous,labs,element_text
import matplotlib.pyplot as plt
from joblib import Parallel,delayed,effective_n_jobs



class BaseWoePlotter:   
    
    def _woe_plot(self,varbin,figure_size,n_jobs,verbose):
        
        n_jobs=effective_n_jobs(n_jobs)   
                              
        p=Parallel(n_jobs=n_jobs,verbose=verbose)
        
        res=p(delayed(self._get_plot_single)(varbin[key],figure_size,False) for key in varbin)
        
        out={colname:fig for fig,colname in res}
        
        return out     

    
    def _woe_plot_group(self,varbin_g,sort_column,figure_size,n_jobs,verbose):

        n_jobs=effective_n_jobs(n_jobs)   
                              
        p=Parallel(n_jobs=n_jobs,verbose=verbose)
        
        res=p(delayed(self._get_plot_single_group)(varbin_g[varbin_g['variable']==col],sort_column,figure_size,False) for col in varbin_g['variable'].unique())
        
        out={colname:fig for fig,colname in res}     
        
        return out
         

    def _get_bin(self,binx):
    
        binx=binx.copy().set_index('bin')
        
        if not binx.loc['missing','count']:
    
            binx=binx[binx.index!='missing']
    
        if not binx.loc['special','count']:
    
            binx=binx[binx.index!='special']
    
        binx=binx.reset_index()
        binx['pos']=binx['good']/binx['count'].sum()
        binx['neg']=binx['bad']/binx['count'].sum()
        binx['rowid']=binx.index+1
        binx['lineval_l']=binx['badprob'].mul(100).round(2).map(str)+'%'
        binx['bin']=binx['bin'].astype('object')
        
        return binx
    
    def _get_plot_single(self,binx,figure_size=None,show_plot=True):
    
        #dt transform
        binx=self._get_bin(binx.reset_index())
        
        binx_melt=pd.melt(binx,id_vars = ["bin","rowid"], value_vars =["pos", "neg"], 
                    value_name = "negpos").rename(columns={'variable':''})
        
        #title
        title_string = binx['variable'][0]+'(iv:{},ks:{})'.format(round(binx['total_iv'][0],4),round(binx['ks_max'][0],4))
    
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
    
        #base plot using ggplot2(no sec_axis function in plotnine - 0.8.0)
        figure=(ggplot() + geom_bar(binx_melt,aes(x='bin',y='negpos',fill=''), stat="identity",show_legend = True)
            + geom_point(data=binx,mapping=aes(x='rowid',y='badprob'), stat="identity",color='blue')
            + geom_path(data=binx,mapping=aes(x='rowid',y='badprob'),color='blue')
            + scale_y_continuous(limits = (0, y_max))
            + theme_bw()
            + labs(x='',y='Bin count distribution',title=title_string)
            + theme(
              figure_size=figure_size,
              legend_position=(0.5,0),legend_direction="horizontal",
              legend_key_size=10)
            + geom_text(data=binx, mapping=aes(x = 'rowid',y='badprob',label='lineval_l'),va = 'bottom',color='blue')
        ).draw()
    
        #add subplot(second axis) using matplotlib 
        ax1=figure.get_axes()[-1]
        ax2=ax1.twinx()
        ax2.set_ylabel('Bad probability', color='blue')
        ax2.set_ylim(top=y_max)
        ax2.set_yticks(np.arange(0, y_max, 0.2))
        ax2.tick_params(axis='y', colors='blue')   
        
        if not show_plot:
            
            plt.close()
        
        return figure,binx['variable'][0]
    
    def _get_plot_single_group(self,binx_g,sort_column=None,figure_size=None,show_plot=True):
    
        #dt transform
        gs=pd.unique([i[0] for i in binx_g.columns.tolist() if i[0] not in ['variable','bin']]).tolist()
        
        self._check_plot_sort(sort_column,gs)
    
        binx_g_h=pd.concat({g:self._get_bin(binx_g[g].join(binx_g['variable']).join(binx_g['bin']).assign(g=g)) for g in gs})    
    
        binx_g_melt=pd.concat({g:pd.melt(self._get_bin(binx_g[g].join(binx_g['variable']).join(binx_g['bin'])),
                      id_vars = ["bin","rowid"], 
                      value_vars =["pos", "neg"], 
                      value_name = "negpos").rename(columns={'variable':''}).assign(g=g) for g in gs})
    
        if sort_column:
    
            binx_g_h['g']=binx_g_h['g'].astype('category').cat.reorder_categories(sort_column)
            binx_g_melt['g']=binx_g_melt['g'].astype('category').cat.reorder_categories(sort_column)
    
        #title
        binx_g_ivks=binx_g_h[['total_iv','ks_max']].droplevel(1)
        binx_g_ivks=binx_g_ivks.loc[~binx_g_ivks.index.duplicated(),:]
        
        iv_d=binx_g_ivks['total_iv'].to_dict()
        ks_d=binx_g_ivks['ks_max'].to_dict()
    
        keys=sort_column if sort_column else iv_d.keys()
        
        #title
        title_string=binx_g_h['variable'][0]+':'+(',').join(['{}(iv:{},ks:{})'.format(key,round(iv_d[key],4),round(ks_d[key],4)) for key in keys])
    
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
        
        #base plot using ggplot2(no sec_axis function in plotnine - 0.8.0)
        figure=(ggplot() + geom_bar(binx_g_melt,aes(x='bin',y='negpos',fill=''), stat="identity",show_legend = True)
            + geom_point(data=binx_g_h,mapping=aes(x='rowid',y='badprob'), stat="identity",color='blue')
            + geom_path(data=binx_g_h,mapping=aes(x='rowid',y='badprob'),color='blue')
            + facet_wrap(('g'))
            + scale_y_continuous(limits = (0, y_max))
            + theme_bw()
            + labs(x='',y='Bin count distribution',title=title_string)
            + theme(
              figure_size=figure_size,
              legend_position=(0.5,0),legend_direction="horizontal",
              legend_key_size=10)
            + geom_text(data=binx_g_h, mapping=aes(x = 'rowid',y='badprob',label='lineval_l'),va = 'bottom',color='blue')
        ).draw()
    
        #add subplot(second axis) using matplotlib 
        ax1=figure.get_axes()[-1]
        ax2=ax1.twinx()
        ax2.set_ylabel('Bad probability', color='blue')
        ax2.set_ylim(top=y_max)
        ax2.set_yticks(np.arange(0, y_max, 0.2))
        ax2.tick_params(axis='y', colors='blue')
        
        if not show_plot:
            
            plt.close()     
        
        return figure,binx_g_h['variable'][0]
    
    
    def _check_plot_sort(self,sort_column,gs):
        
        if sort_column:
            
            if isinstance(sort_column,list):
                
            
                if set(gs)!=set(sort_column):    
                    
                    
                    raise ValueError("val in sort_column not equal to val in datq[col]")
                    
            else:
                    
                raise ValueError("sort_column is list")
        
    
    
    
    
    
    
    
    