# -*- coding: utf-8 -*-
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
import scorecardpy as sc
import toad


class finbinSelector(TransformerMixin):
    
    def __init__(self,ivlimit=0.02,bin_num=20,special_values=[np.nan],y='target'):
        """ 
        IV预筛选,适用于二分类模型
        Parameters:
        ----------
            ivlimit=0.02:float,IV阈值,IV低于该阈值特征将被剔除
            y='target':str,目标变量的列名
            bin_num=20:int,连续特征等频分箱的分箱数,默认20,分类特征则将使用其原始类别
            special_values=[np.nan]:list,特殊值指代,列表,默认
            
        Attribute:
        ----------
            features_info:dict,每一步筛选的特征进入记录
            iv_info:pd.Series,细分箱后所有特征的iv统计结果
            keep_col:list,经细分箱及指定ivlimit筛选后保留的特征名列表
            breaks_list:dict,细分箱产生的切分点信息列表
            finebin:所有特征细分箱明细dict(经iv筛选后)
            finebin_nonmonotonic:dict,非单调特征细分箱明细dict(经iv筛选后)
            finebin_monotonic:dict,单调特征细分箱明细dict(经iv筛选后)
        """
        self.ivlimit=ivlimit
        self.y=y
        self.bin_num=bin_num
        self.special_values=special_values
    
    def fit(self,X,y):
        """ 
        细分箱筛选
        Parameters:   
        ----------
            X:特征数据,pd.DataFrame
            y:目标变量列,pd.Series,必须与X索引一致
        """ 
        
        df=X.join(y)
        ivlimit=self.ivlimit
        
        #获取细分箱分箱点
        self.breaks_list=self.getBreakslistFinbin(X,y)
        
        #进行细分箱
        finebin=sc.woebin(df,y=self.y,check_cate_num=False,count_distr_limit=0.01,bin_num_limit=self.bin_num,breaks_list=self.breaks_list)

        #筛选合适变量
        FacCol=df.select_dtypes(exclude='number').columns.tolist()
        finebindf=pd.concat(finebin.values())
        
        #结果写出
        #now=time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
        #finebindf.to_excel(self.out_path+'/finebin'+now+'.xlsx')

        #开始筛选
        Numfinbindf=finebindf[~finebindf.variable.isin(FacCol)] #数值列
        Facfinbindf=finebindf[finebindf.variable.isin(FacCol)] #列别列

        self.iv_info=pd.concat([Numfinbindf.groupby('variable')['bin_iv'].sum().rename('iv'),Facfinbindf.groupby('variable')['bin_iv'].sum().rename('iv')])
        self.keep_col=self.iv_info[self.iv_info>=ivlimit].index.tolist()
        self.finebin={column:finebin.get(column) for column in self.keep_col}
        self.finebin_monotonic,self.finebin_nonmonotonic=self.checkMonotonicFeature(self.finebin,ivlimit)
        
        return self
    
    def transform(self,X):
        return X[self.keep_col]
    
    def getBreakslistFinbin(self,X,y):

        """
        等频分箱产生sc.woebin可用的breaklist,用于细分箱
        Parameters:
        --
            X:特征数据,pd.DataFrame
            y:目标变量列,pd.Series,必须与X索引一致
        """
        df=X.join(y)
        bin_num=self.bin_num
        
        FacCol=X.select_dtypes(exclude='number').columns.tolist() #数值列
        NumCol=X.select_dtypes(include='number').columns.tolist() #分类列

        breaklist={}

        #bin_num下若特征分布过于集中,等频分箱将合并集中的分箱区间为最终分箱区间
        for numcol in NumCol:
            index_drop=df[numcol][df[numcol].isin(self.special_values)].index
            numcol_drop=df[numcol].drop(index_drop)           
            if index_drop.size>0:
                bin_num_adj=bin_num-1
            else:
                bin_num_adj=bin_num            
            breaklist[numcol]=np.unique([round(j,2) for j in numcol_drop.quantile([i/(bin_num_adj) for i in range(1,bin_num_adj)]).unique().tolist()])

        #分类变量则根据类别数直接分箱    
        for faccol in FacCol:
            charcol_drop=df[faccol].drop(df[faccol][df[faccol].isin(self.special_values)].index)
            breaklist[faccol]=charcol_drop.unique()
                
        return breaklist
    
    def checkMonotonicFeature(self,varbin,ivlimit):
        """
        检查细分箱后特征的分箱woe是否单调
        Parameters:
        --
        Attribute:
        --
            finebin_nonmonotonic:dict,非单调特征dict
            finebin_monotonic:dict,单调特征dict
        """        
        varbin_monotonic={}
        varbin_nonmonotonic={}
        for column in varbin.keys():
            var_woe=varbin[column].query('bin!="missing"').woe
            if var_woe.is_monotonic_decreasing or var_woe.is_monotonic_increasing:
                if varbin[column].total_iv[0]>ivlimit:
                    varbin_monotonic[column]=varbin[column]    
            else:
                if varbin[column].total_iv[0]>ivlimit:
                    varbin_nonmonotonic[column]=varbin[column]
                    
        return(varbin_monotonic,varbin_nonmonotonic)
    
    
    

class optbinSelector(TransformerMixin):
    
    def __init__(self,y='target',method='dt',n_bins=10,min_samples=0.05,iv_limit=0.02):
        """ 
        最优分箱与交互分箱
        Parameters:
        ----------
            corr_limit:float,相关系数阈值,当两个特征相关性高于该阈值,将剔除掉IV较低的一个       
        Attribute:
        ----------
            features_info:dict,每一步筛选的特征进入记录
        """
        self.target=y
        self.method=method
        self.n_bins=n_bins
        self.min_samples=min_samples
        self.iv_limit=iv_limit
        
        
    def transform(self,X,y=None):
        
        return X[self.keep_col]
              
    def fit(self,X,y):
        """ 
        最优分箱
        """          
        self.X=X.copy()
        self.y=y.copy()
        
        #使用toad进行最优分箱
        self.break_list_toad=toad.transform.Combiner().fit(self.X.join(self.y),y = self.target,\
                                              method = self.method, n_bins=self.n_bins,min_samples=self.min_samples)
        
            
        self.break_list_sc,colname_list=self.get_Breaklist_sc(self.break_list_toad.export())
        
        #使用scorecardpy进行分箱结果输出
        self.optimalbin=sc.woebin(self.X[colname_list].join(self.y),y=self.target,breaks_list=self.break_list_sc,check_cate_num=False)
        
        #最优分箱的特征iv统计值
        self.iv_info=pd.concat(self.optimalbin).groupby('variable')['bin_iv'].sum().rename('iv')

        #IV筛选特征
        self.keep_col=self.iv_info[self.iv_info>=self.iv_limit].index.tolist()
        
        #将单调与不单调特征进行分开
        self.optimalbin_monotonic,self.optimalbin_nonmonotonic=self.checkMonotonicFeature(self.optimalbin,self.iv_limit)        
        
        return self

    def get_Breaklist_sc(self,break_list_toad):
        
        """
        将toad的breaklist结构转化为scorecardpy可用的结构
        """      
        
        cate_colname=self.X.select_dtypes(exclude='number')
        num_colname=self.X.select_dtypes(include='number')
        
        break_list_sc=dict()
        colname_list=list()
        
        #将toad的breaklist转化为scorecardpy的breaklist
        for key in break_list_toad.keys():
            if key in cate_colname and break_list_toad[key]:#防止分箱结果为空

                bin_value_list=[]
                for value in break_list_toad[key]:
                    #if 'nan' in value:
                    #    value=pd.Series(value).replace('nan','missing').tolist()
                    bin_value_list.append('%,%'.join(value))

                break_list_sc[key]=bin_value_list
                colname_list.append(key)

            elif key in num_colname and break_list_toad[key]:#防止分箱结果为空
                break_list_sc[key]=break_list_toad[key]
                colname_list.append(key)
                
        return break_list_sc,colname_list
    
    def checkMonotonicFeature(self,varbin,iv_limit):
        """
        检查细分箱后特征的分箱woe是否单调
        Parameters:
        --
        Attribute:
        --
            finebin_nonmonotonic:dict,非单调特征dict
            finebin_monotonic:dict,单调特征dict
        """        
        varbin_monotonic={}
        varbin_nonmonotonic={}
        for column in varbin.keys():
            var_woe=varbin[column].query('bin!="missing"').woe
            if var_woe.is_monotonic_decreasing or var_woe.is_monotonic_increasing:
                if varbin[column].total_iv[0]>iv_limit:
                    varbin_monotonic[column]=varbin[column]    
            else:
                if varbin[column].total_iv[0]>iv_limit:
                    varbin_nonmonotonic[column]=varbin[column]
                    
        return(varbin_monotonic,varbin_nonmonotonic)
    
    
    def adjustBin(self,bins=None,only_nonmonotonic_var=True):
        """
        调整分箱，需先运行fit
        plt.rcParams["figure.figsize"] = [10, 5]
        """          
        
        if not bins and only_nonmonotonic_var is True:
            
            break_list_adj=sc.woebin_adj(
                dt=self.X[self.keep_col].join(self.y),
                y=self.target,
                adj_all_var=True,
                count_distr_limit=0.05,
                bins=self.optimalbin_nonmonotonic,
                method=self.method
            )
            
            adjustedbin_nonmonotonic=sc.woebin(self.X[self.keep_col].join(self.y),
                                       y=self.target,breaks_list=break_list_adj)
            
            self.adjustedbin=self.optimalbin_monotonic.update(adjustedbin_nonmonotonic)
            
        elif not bins and only_nonmonotonic_var is False:
            
            break_list_adj=sc.woebin_adj(
                dt=self.X[self.keep_col].join(self.y),
                y=self.target,
                adj_all_var=True,
                count_distr_limit=0.05,
                bins=self.optimalbin,
                method=self.method
            )
            self.adjustedbin=sc.woebin(self.X[self.keep_col].join(self.y),
                                y=self.target,breaks_list=break_list_adj)
            
        else:            
            break_list_adj=sc.woebin_adj(
                dt=self.X[self.keep_col].join(self.y),
                y=self.target,
                adj_all_var=True,
                count_distr_limit=0.05,
                bins=bins,
                method=self.method
            ) 
            self.adjustedbin=sc.woebin(self.X[self.keep_col].join(self.y),
                                y=self.target,breaks_list=break_list_adj)
                       
        return self


    