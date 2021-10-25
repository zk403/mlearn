# -*- coding: utf-8 -*-
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
import scorecardpy as sc
import toad
from binarymodels.report_report import varReport
from pandas.api.types import is_numeric_dtype

class finbinSelector(TransformerMixin):
    
    def __init__(self,ivlimit=0.02,bin_num=20,special_values=[np.nan],apply_dt=None,out_path=None,psi_base_mon='latest'):
        """ 
        IV预筛选,适用于二分类模型
        Parameters:
        ----------
            ivlimit=0.02:float,IV阈值,IV低于该阈值特征将被剔除
            y='target':str,目标变量的列名
            bin_num=20:int,连续特征等频分箱的分箱数,默认20,分类特征则将使用其原始类别
            special_values=[np.nan]:list,特殊值指代,列表,默认
            out_path:str,输出分箱结果报告至指定路径的模型文档中
            
                注意:finbinSelector所产生的特征分箱取决于fit中的in-sample数据,若希望基于in-sample信息对out-sample进行分箱并输出报告则可使用如下代码:
                    
                    breaks_list_insample=finbinSelector(bin_num=10).getBreakslistFinbin(X_train_1,y_train)

                    varReport(breaks_list_dict=breaks_list_insample,
                                  out_path='report',sheet_name='_oot').fit(X_oot,y_oot)
              
            apply_dt:pd.Series,产生报告时使用,用于标示X的时期的字符型列且需要能转化为int
                + eg:pd.Series(['202001','202002‘...],name='apply_mon',index=X.index)
            psi_base_mon:str,当apply_dt非空时,psi计算的基准,可选earliest和latest，也可用户自定义
                + earliest:选择apply_dt中最早的时期的分布作为psi基准
                + latest:选择apply_dt中最晚的时期的分布作为psi基准
                + str:在apply_dt中任选一个时期的分布作为psi基准  
                
            
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
        self.bin_num=bin_num
        self.special_values=special_values
        self.out_path=out_path
        self.apply_dt=apply_dt
        self.psi_base_mon='latest'
    
    def fit(self,X,y):
        """ 
        细分箱筛选
        Parameters:   
        ----------
            X:特征数据,pd.DataFrame
            y:目标变量列,pd.Series,必须与X索引一致
        """ 
        ivlimit=self.ivlimit
        
        #获取细分箱分箱点
        self.breaks_list=self.getBreakslistFinbin(X,y)
        
        #进行细分箱
        #finebin=sc.woebin(df,y=self.y,check_cate_num=False,count_distr_limit=0.01,bin_num_limit=self.bin_num,breaks_list=self.breaks_list)
        

        bin_res=varReport(breaks_list_dict=self.breaks_list,out_path=self.out_path,
                          sheet_name='_fin',apply_dt=None).fit(X,y)
        
        if self.apply_dt is not None:
 
            varReport(breaks_list_dict=self.breaks_list,
                              out_path=self.out_path,apply_dt=self.apply_dt,
                              psi_base_mon=self.psi_base_mon,
                              sheet_name='_fin').fit(X,y) 
                  

        #筛选合适变量
        col_cate=X.select_dtypes(include='object').columns.tolist()
        finebindf=pd.concat(bin_res.var_report_dict.values())

        #开始筛选
        Numfinbindf=finebindf[~finebindf.variable.isin(col_cate)] #数值列
        Facfinbindf=finebindf[finebindf.variable.isin(col_cate)] #分类列

        self.iv_info=pd.concat([Numfinbindf.groupby('variable')['bin_IV'].sum().rename('iv'),Facfinbindf.groupby('variable')['bin_IV'].sum().rename('iv')])
        self.keep_col=self.iv_info[self.iv_info>=ivlimit].index.tolist()
        self.finebin={column:bin_res.var_report_dict.get(column) for column in self.keep_col}
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
        
        FacCol=X.select_dtypes(include='object').columns.tolist() #数值列
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
            finebin_nonmonotonic:dict,非单调特征list
            finebin_monotonic:dict,单调特征list
        """        
        varbin_monotonic=[]
        varbin_nonmonotonic=[]
        for column in varbin.keys():
            #var_woe=varbin[column].query('bin!="missing"').woe
            varbin_col=varbin[column]
            var_woe=varbin_col[varbin_col['bin'].astype(str)!="missing"]['WOE']
            
            if var_woe.is_monotonic_decreasing or var_woe.is_monotonic_increasing:
                if varbin_col['bin_IV'].sum()>ivlimit:
                    varbin_monotonic.append(column)
            else:
                if varbin_col['bin_IV'].sum()>ivlimit:
                    varbin_nonmonotonic.append(column)
                    
        return(varbin_monotonic,varbin_nonmonotonic)
    
    
    

class optbinSelector(TransformerMixin):
    
    def __init__(self,method='dt',n_bins=10,min_samples=0.05,iv_limit=0.02,out_path=None,apply_dt=None,psi_base_mon='latest',break_list_adj=None):
        """ 
        最优分箱与交互分箱
        Parameters:
        ----------
            method
            n_bins
            min_samples
            iv_limit
            
        Attribute:
        ----------
            features_info:dict,每一步筛选的特征进入记录
        """
        self.method=method
        self.n_bins=n_bins
        self.min_samples=min_samples
        self.iv_limit=iv_limit
        self.out_path=out_path
        self.apply_dt=apply_dt
        self.psi_base_mon=psi_base_mon
        self.break_list_adj=break_list_adj
        
    def transform(self,X,y=None):
        
        return X[self.keep_col]
              
    def fit(self,X,y):
        """ 
        最优分箱
        """          
        self.X=X.copy()
        self.y=y.copy()      
        self.target=y.name
        
        
        #若分箱结果已知,可直接调用并进行分箱与后续woe编码
        if self.break_list_adj:
                     
            
            #只选择需要调整分箱的特征进行分箱
            self.keep_col=list(self.break_list_adj.keys())
            
            self.adjbin_woe=sc.woebin(self.X[self.keep_col].join(self.y),y=self.target,breaks_list=self.break_list_adj,check_cate_num=False)
            
            #是否输出报告
            if self.out_path:
                
                varReport(breaks_list_dict=self.break_list_adj,out_path=self.out_path,
                              sheet_name='_adj',apply_dt=None).fit(self.X[self.keep_col],self.y)
        
                if self.apply_dt is not None:
                    
                    varReport(breaks_list_dict=self.break_list_adj,
                                      out_path=self.out_path,apply_dt=self.apply_dt,
                                      psi_base_mon=self.psi_base_mon,
                                      sheet_name='_adj').fit(self.X[self.keep_col],self.y)  
                                            
        else:
        
            #使用toad进行最优分箱
            self.break_list_toad=toad.transform.Combiner().fit(self.X.join(self.y),y = self.target,\
                                                  method = self.method, n_bins=self.n_bins,min_samples=self.min_samples)            
                
            self.break_list_opt=self.get_Breaklist_sc(self.break_list_toad.export(),X,y)

            #最优分箱的特征iv统计值
            self.optimalbin=varReport(breaks_list_dict=self.break_list_opt,out_path=None).fit(self.X,self.y)
            
            self.iv_info=pd.concat(self.optimalbin.var_report_dict.values()).groupby('variable')['bin_IV'].sum().rename('iv')
    
            #IV筛选特征
            self.keep_col=self.iv_info[self.iv_info>=self.iv_limit].index.tolist()  
            
            #是否输出报告
            if self.out_path:
                
                varReport(breaks_list_dict=self.break_list_opt,out_path=self.out_path).fit(self.X[self.keep_col],self.y)

                if self.apply_dt is not None:
             
                    varReport(breaks_list_dict=self.break_list_opt,
                                          out_path=self.out_path,apply_dt=self.apply_dt,
                                          psi_base_mon=self.psi_base_mon,
                                          sheet_name='_opt').fit(self.X[self.keep_col],self.y)               
       
        return self

    def get_Breaklist_sc(self,break_list,X,y):
        
        """
        将toad的breaklist结构转化为scorecardpy可用的结构
        """      
        
        
        #判断break_list是sc格式还是toad格式
        count=0
        for var_list in list(break_list.values()):
            
            for value in var_list:
                if isinstance(value,list):
                    count=count+1           
                break
            
        columns=list(break_list.keys())
        
        #toad格式时转换为sc格式
        if count>0:
        
            cate_colname=X[columns].select_dtypes(exclude='number')
            num_colname=X[columns].select_dtypes(include='number')

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
        
    
    def checkMonotonicFeature(self,varbin):
        """
        检查细分箱后特征的分箱woe是否单调
        Parameters:
        --
        Attribute:
        --
            finebin_nonmonotonic:dict,非单调特征list
            finebin_monotonic:dict,单调特征list
        """        
        varbin_monotonic={}
        varbin_nonmonotonic={}
        for column in varbin.keys():
            #var_woe=varbin[column].query('bin!="missing"').woe
            varbin_col=varbin[column]
            var_woe=varbin_col[varbin_col['bin'].astype(str)!="missing"]['WOE']
            
            if var_woe.is_monotonic_decreasing or var_woe.is_monotonic_increasing:
                #if varbin_col['bin_IV'].sum()>ivlimit:
                #varbin_monotonic.append(column)
                varbin_monotonic[column]=varbin_col
            else:
                #if varbin_col['bin_IV'].sum()>ivlimit:
                #varbin_nonmonotonic.append(column)
                varbin_nonmonotonic[column]=varbin_col
                    
        return(varbin_monotonic,varbin_nonmonotonic)
    
    
    def fit_adjustBin(self,br_to_adjusted=None,only_nonmonotonic_var=True,out_path=None,apply_dt=None,psi_base_mon='latest',break_list_adj=None):
        """
        根据最优分箱结果调整分箱，需先运行fit
        plt.rcParams["figure.figsize"] = [10, 5]
        Parameters:
        --       
        br_to_adjusted:
        only_nonmonotonic_var
        
        Return:
        --
        adjbin_woe:
        
        """          
        
        #若自定义break_list存在则使用自定义分箱
        if br_to_adjusted:
            
            self.keep_col=list(br_to_adjusted.keys())
            
            
            bin_sc=sc.woebin(self.X[self.keep_col].join(self.y),
                      y=self.target,breaks_list=br_to_adjusted)         
            
            self.break_list_adj=sc.woebin_adj(
                dt=self.X[self.keep_col].join(self.y),
                y=self.target,
                adj_all_var=True,
                count_distr_limit=0.05,
                bins=bin_sc,
                method='chimerge'
            ) 
            
            self.adjbin_woe=sc.woebin(self.X[self.keep_col].join(self.y),
                                y=self.target,breaks_list=self.break_list_adj)
                        
            
        #否则将根据最优分箱结果调整    
        else:       
            
            bin_sc=sc.woebin(self.X[self.keep_col].join(self.y),
                      y=self.target,breaks_list=self.optimalbin.breaks_list_dict)
            
            #将单调与不单调特征进行分开
            self.optimalbin_monotonic,self.optimalbin_nonmonotonic=self.checkMonotonicFeature(bin_sc)              
            
            #只调整最优分箱中的非单调分箱
            if only_nonmonotonic_var is True:
                
                
                self.break_list_adj=sc.woebin_adj(
                    dt=self.X[self.keep_col].join(self.y),
                    y=self.target,
                    adj_all_var=True,
                    count_distr_limit=0.05,
                    bins=self.optimalbin_nonmonotonic,
                    method='chimerge'
                )
                
                adjustedbin_nonmonotonic=sc.woebin(self.X[self.keep_col].join(self.y),
                                           y=self.target,breaks_list=self.break_list_adj)
                
                self.adjbin_woe=self.optimalbin_monotonic.update(adjustedbin_nonmonotonic)
    
            #调整全量分箱
            else:
                
                self.break_list_adj=sc.woebin_adj(
                    dt=self.X[self.keep_col].join(self.y),
                    y=self.target,
                    adj_all_var=True,
                    count_distr_limit=0.05,
                    bins=bin_sc,
                    method='chimerge'
                )
                
                self.adjbin_woe=sc.woebin(self.X[self.keep_col].join(self.y),
                                    y=self.target,breaks_list=self.break_list_adj)
                 
        
        if out_path:
                
            varReport(breaks_list_dict=self.break_list_adj,out_path=out_path,
                              sheet_name='_adj',apply_dt=None).fit(self.X[self.keep_col],self.y)
        
            if apply_dt is not None:
                    
                varReport(breaks_list_dict=self.break_list_adj,
                           out_path=out_path,apply_dt=apply_dt,
                           psi_base_mon=psi_base_mon,
                           sheet_name='_adj').fit(self.X[self.keep_col],self.y)  
        
                      
        return self   