# -*- coding: utf-8 -*-
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
import scorecardpy as sc
import toad
from BDMtools.report_report import varReport
from BDMtools.selector_bin_fun import binAdjusterKmeans,binAdjusterChi
#from joblib import Parallel,delayed
#from pandas.api.types import is_numeric_dtype

class finbinSelector(TransformerMixin):
    
    def __init__(self,ivlimit=0.02,bin_num=20,bin_num_limit=5,special_values=[np.nan],method='freq',out_path=None,n_jobs=-1,verbose=0):
        """ 
        IV预筛选,适用于二分类模型
        Parameters:
        ----------
            ivlimit=0.02:float,IV阈值,IV低于该阈值特征将被剔除
            bin_num=20:int,连续特征等频分箱的分箱数,默认20,分类特征则将使用其原始类别
            method='freq',可选'freq'与'freq-kmeans'
                + 'freq':等频分箱,分箱数由bin_num指定
                + 'freq-kmeans':基于Kmeans，对等频分箱结果进行自动调整，以将badrate近似的箱进行合并
            bin_num_limit=10,method='freq-kmeans'时，合并分箱最低限制,bin_num_limit<bin_num时才有效果        
            special_values=[np.nan]:list,特殊值指代,列表,默认
            out_path:str,输出分箱结果报告至指定路径的模型文档中
            
                注意:finbinSelector所产生的特征分箱取决于fit中的in-sample数据,若希望基于in-sample信息对out-sample进行分箱并输出报告则可使用如下代码:
                    
                    breaks_list_insample=finbinSelector(bin_num=10).getBreakslistFinbin(X_train_1,y_train)

                    varReport(breaks_list_dict=breaks_list_insample,
                                  out_path='report',sheet_name='_oot').fit(X_oot,y_oot)
            
            n_jobs:int,输出特征分析报告和采用freq-kmeans分箱时并行计算job数,默认-1(使用全部的 CPU cores)
            verbose:int,并行计算信息输出等级
                
            
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
        self.method=method
        self.bin_num_limit=bin_num_limit
        self.special_values=special_values
        self.out_path=out_path
        self.n_jobs=n_jobs
        self.verbose=verbose
    
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
   
        if self.method=='freq-kmeans':
            
            self.breaks_list=self.getBreakslistFinbin(X,y,self.special_values)
            self.breaks_list=binAdjusterKmeans(breaks_list=self.breaks_list,
                                               bin_limit=self.bin_num_limit,
                                               special_values=self.special_values,
                                               n_jobs=self.n_jobs,
                                               verbose=self.verbose,
                                               ).fit(X,y).breaks_list_adj
            
        elif self.method=='freq':
            
            self.breaks_list=self.getBreakslistFinbin(X,y,self.special_values)
            
        else:
            raise IOError("method in ('freq','freq-kmeans')")
            
        
        bin_res=sc.woebin(X.join(y).replace(self.special_values,np.nan),
                              y=y.name,check_cate_num=False,count_distr_limit=0.01,
                              bin_num_limit=self.bin_num,breaks_list=self.breaks_list)
        
        #进行细分箱
        if self.out_path:

            varReport(breaks_list_dict=self.breaks_list,
                      out_path=self.out_path,
                      special_values=self.special_values,
                      tab_suffix='_fin',
                      n_jobs=self.n_jobs,
                      verbose=self.verbose,
                      apply_dt=None).fit(X,y)

        #加入ks
        bin_res_ks={}
        for col in bin_res:
            df_var=bin_res[col]
            good_distr=df_var['good'].div(df_var['good'].sum())
            bad_distr=df_var['bad'].div(df_var['bad'].sum())
            df_var['ks']=good_distr.sub(bad_distr).abs()
            bin_res_ks[col]=df_var
        
        finebindf=pd.concat(bin_res_ks.values())
                
        self.iv_info=pd.concat([finebindf.groupby('variable')['bin_iv'].sum().rename('total_iv')])        
        self.ks_info=pd.concat([finebindf.groupby('variable')['ks'].max().rename('ks_max')])        
        self.keep_col=self.iv_info[self.iv_info>=ivlimit].index.tolist()
        self.finebin={column:bin_res.get(column) for column in self.keep_col}
        self.breaks_list={column:self.breaks_list[column] for column in self.keep_col}
             
        return self
    
    def transform(self,X):
        return X[self.keep_col]
    
    def getBreakslistFinbin(self,X,y,special_values):

        """
        等频分箱产生sc.woebin可用的breaklist,用于细分箱
        Parameters:
        --
            X:特征数据,pd.DataFrame
            y:目标变量列,pd.Series,必须与X索引一致
        """
        
        CatCol=X.select_dtypes(include='object').columns.tolist() #分类列
        NumCol=X.select_dtypes(include='number').columns.tolist() #数值列

        breaklist={}
        
        X=X

        #bin_num下若特征分布过于集中,等频分箱将合并集中的分箱区间为最终分箱区间
        for numcol in NumCol:
            
            numcol_s=X[numcol].replace(special_values,np.nan)
            
            if numcol_s.dropna().size:
            
                _,breaks=pd.qcut(numcol_s.dropna(),self.bin_num,duplicates='drop',retbins=True,precision=3)
                
            else:
                
                breaks=[0]
            
            if len(breaks)==2:
                
                breaklist[numcol]=breaks.tolist()[:-1]
                
            elif len(breaks)==1:
                
                breaklist[numcol]=breaks.tolist()
                
            elif len(breaks)==0:
                
                breaklist[numcol]=numcol_s.dropna().unique().tolist()
                
            else:
                
                breaklist[numcol]=breaks.tolist()[1:-1]
            
        #分类变量则根据类别数直接分箱    
        for catcol in CatCol:
            
            catcol_s=X[catcol].replace(special_values,'missing')
            
            breaklist[catcol]=catcol_s.unique().tolist()
                
        return breaklist
        
    

class optbinSelector(TransformerMixin):
    
    def __init__(self,method='tree',n_bins=10,min_samples=0.05,special_values=[np.nan],iv_limit=0.02,
                 out_path=None,break_list_adj=None,n_jobs=-1,verbose=0):
        """ 
        最优分箱与交互分箱
        Parameters:
        ----------
            method:str,最优分箱方法
                + 'dt':决策树,递归分裂gini增益最高的切分点形成新分箱直到达到终止条件
                + 'chi':卡方,先等频预分箱,再递归合并低于卡方值(交叉表卡方检验的差异不显著)的分箱
                + 'chi_m':卡方单调,先等频预分箱,再合并低于卡方值(交叉表卡方检验的差异不显著)的分箱或不单调(badrate)的分箱    
                          注意chi_m只适用于数值列,字符列将在breaks_list中被剔除
            n_bins:int,method为'dt'和'chi'时代表分箱数,method为'chi_m'时代表卡方单调分箱的预分箱数
            min_samples,method为'dt'和'chi'时代表分箱最终箱样本占比限制
            special_values,list,特殊值指代值
            ivlimit=0.02:float,IV阈值,IV低于该阈值特征将被剔除
            out_path:str,输出分箱结果报告至指定路径的模型文档中              
            apply_dt:pd.Series,产生报告时使用,用于标示X的时期的字符型列且需要能转化为int
                + eg:pd.Series(['202001','202002‘...],name='apply_mon',index=X.index)
            psi_base_mon:str,当apply_dt非空时,psi计算的基准,可选earliest和latest，也可用户自定义
                + earliest:选择apply_dt中最早的时期的分布作为psi基准
                + latest:选择apply_dt中最晚的时期的分布作为psi基准
                + all:选择总分布作为psi基准            
            n_jobs:int,输出特征分析报告和采用chi2_m分箱时并行计算job数,默认-1(使用全部的 CPU cores)
            verbose:int,并行计算信息输出等级
        Attribute:
        ----------
            features_info:dict,每一步筛选的特征进入记录
        """
        self.method=method
        self.n_bins=n_bins
        self.min_samples=min_samples
        self.iv_limit=iv_limit
        self.special_values=special_values
        self.out_path=out_path
        self.break_list_adj=break_list_adj
        self.n_jobs=n_jobs
        self.verbose=verbose
        
    def transform(self,X,y=None):
        
        return X[self.keep_col]
              
    def fit(self,X,y):
        """ 
        最优分箱
        """              
        self.target=y.name
        
        
        #若给定分箱breaklist已知,可直接调用并进行分箱与后续woe编码
        if self.break_list_adj:
                     
            
            #只选择需要调整分箱的特征进行分箱
            self.keep_col=list(self.break_list_adj.keys())
            
            self.adjbin_woe=sc.woebin(X[self.keep_col].join(y).replace(self.special_values,np.nan),
                                      y=y.name,
                                      breaks_list=self.break_list_adj,check_cate_num=False)
            
            #是否输出报告
            if self.out_path:
                
                varReport(breaks_list_dict=self.break_list_adj,
                          special_values=self.special_values,
                          out_path=self.out_path,                          
                          n_jobs=self.n_jobs,
                          verbose=self.verbose,
                          tab_suffix='_adj',apply_dt=None).fit(X[self.keep_col],y)
                    
        #若不给定breaklist，则进行最优分箱                                   
        else:
            
            #卡方单调分箱
            if self.method == 'chi_m':
            
                #注意chi_m只适用于数值列,字符列将在breaks_list中被剔除
                self.break_list_opt=binAdjusterChi(bin_num=20,
                                                   chi2_p=0.1,
                                                   special_values=self.special_values,
                                                   n_jobs=self.n_jobs,
                                                   verbose=self.verbose).fit(X, y).breaks_list_chi2m
                                
                
            else:    
                #使用toad进行最优分箱
                self.break_list_toad=toad.transform.Combiner().fit(X.join(y),y = y.name,\
                                                      method = self.method, n_bins=self.n_bins,min_samples=self.min_samples)            
                    
                self.break_list_opt=self.get_Breaklist_sc(self.break_list_toad.export(),X,y)
            
            self.keep_col=list(self.break_list_opt.keys())
            #最优分箱的特征iv统计值
            
            bin_res=sc.woebin(X[self.keep_col].join(y).replace(self.special_values,np.nan),
                                      y=y.name,breaks_list=self.break_list_opt,
                                      check_cate_num=False)
            
            bin_res_ks={}
            for col in bin_res:
                df_var=bin_res[col]
                good_distr=df_var['good'].div(df_var['good'].sum())
                bad_distr=df_var['bad'].div(df_var['bad'].sum())
                df_var['ks']=good_distr.sub(bad_distr).abs()
                bin_res_ks[col]=df_var
            
            optbindf_ks=pd.concat(bin_res_ks.values())

            self.iv_info=optbindf_ks.groupby('variable')['bin_iv'].sum().rename('total_iv')
            self.ks_info=optbindf_ks.groupby('variable')['ks'].max().rename('total_iv')
            
            #IV筛选特征
            self.keep_col=self.iv_info[self.iv_info>=self.iv_limit].index.tolist()  
            self.optbin={column:bin_res.get(column) for column in self.keep_col}
            self.break_list_opt={column:self.break_list_opt.get(column) for column in self.keep_col}
 
            #是否输出报告
            if self.out_path:
                
                varReport(breaks_list_dict=self.break_list_opt,
                          special_values=self.special_values,
                          n_jobs=self.n_jobs,
                          verbose=self.verbose,
                          out_path=self.out_path,
                          apply_dt=None,
                          tab_suffix='_opt'
                          ).fit(X[self.keep_col],y)              
       
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
    
    
    def fit_adjustBin(self,X,y,br_to_adjusted=None,out_path=None):
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
        
        self.X=X
        self.y=y
        
        #若自定义break_list存在则使用自定义分箱
        if br_to_adjusted:
            
            self.keep_col=list(br_to_adjusted.keys())
            
            
            bin_sc=sc.woebin(self.X[self.keep_col].join(self.y).replace(self.special_values,np.nan),
                             y=self.target,
                             breaks_list=br_to_adjusted)
            
            breaks_str=sc.woebin_adj(
                dt=self.X[self.keep_col].join(self.y).replace(self.special_values,np.nan),
                y=self.target,
                adj_all_var=True,
                count_distr_limit=0.05,
                bins=bin_sc,
                method='chimerge'
            ) 
            
            exec('self.breaks_dict_raw='+breaks_str)

            self.break_list_adj={key:np.sort(self.breaks_dict_raw[key]).tolist() for key in list(self.breaks_dict_raw.keys())}
            
            del self.breaks_dict_raw
            
            self.adjbin_woe=sc.woebin(self.X[self.keep_col].join(self.y).replace(self.special_values,np.nan),
                                      y=self.target,breaks_list=self.break_list_adj)
                        
            
        #否则将根据最优分箱结果调整    
        else:       
            
            bin_sc=sc.woebin(self.X[self.keep_col].join(self.y).replace(self.special_values,np.nan),
                             y=self.target,breaks_list=self.optimalbin.breaks_list_dict)
            

                
            self.break_list_adj=sc.woebin_adj(
                    dt=self.X[self.keep_col].join(self.y),
                    y=self.target,
                    adj_all_var=True,
                    count_distr_limit=0.05,
                    bins=bin_sc,
                    method='chimerge'
                )
                
            
            self.adjbin_woe=sc.woebin(self.X[self.keep_col].join(self.y).replace(self.special_values,np.nan),
                                          y=self.target,breaks_list=self.break_list_adj)
                 
        
        return self   
