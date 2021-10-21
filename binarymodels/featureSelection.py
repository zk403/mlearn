from sklearn.base import TransformerMixin
from sklearn.feature_selection import VarianceThreshold,f_oneway,chi2
from lightgbm import LGBMClassifier
from category_encoders.ordinal import OrdinalEncoder
import numpy as np
import pandas as pd
import scorecardpy as sc
import toad

class selection_pre(TransformerMixin):
    
    def __init__(self,
                 na_pct=0.99,
                 unique_pct=0.99,
                 variance=0,
                 chi2_pvalue=0.05,
                 oneway_pvalue=0.05,
                 tree_imps=0,
                 tree_size=100
                 ):
        """ 
        预筛选,适用于二分类模型
        Parameters:
        ----------
            na_pct:float,(0,1),默认0.99,缺失率高于na_pct的列将被筛除，设定为None将跳过此步骤
            unique_pct:float,(0,1),默认0.99,唯一值占比高于unique_pct的列将被筛除,将忽略缺失值,unique_pct与variance需同时输入，设定为None将跳过此步骤
            variance:float,默认0,方差低于variance的列将被筛除,将忽略缺失值,unique_pct与variance需同时输入，设定为None将跳过此步骤
            chi2_pvalue:float,(0,1),默认0.05,大于chi2_pvalue的列将被剔除,缺失值将被视为单独一类,chi2_pvalue与oneway_pvalue需同时输入，设定为None将跳过此步骤
            oneway_pvalue:float,(0,1),默认0.05,缺失值将被填补为接近正负inf,将计算两次,两次结果中都较大于oneway_pvalue的列将被剔除,此外,该函数默认方差是齐性的,chi2_pvalue与oneway_pvalue需同时输入，设定为None将跳过此步骤
            tree_imps:float,lightgbm树的梯度gain小于等于tree_imps的列将被剔除,默认0，设定为None将跳过此步骤
            tree_size:int,lightgbm树个数,若数据量较大可降低树个数，若tree_imps为None时该参数将被忽略
            
        Attribute:
        ----------
            features_info:dict,每一步筛选的特征进入记录
        """
        self.na_pct=na_pct
        self.unique_pct=unique_pct #
        self.variance=variance
        self.chi2_pvalue=chi2_pvalue #
        self.oneway_pvalue=oneway_pvalue #
        self.tree_imps=tree_imps #
        self.tree_size=tree_size
        
    def transform(self,X,y=None):
        """ 
        变量筛选
        """ 
        return X[self.features_info[max(list( self.features_info.keys()))]]
       
    def fit(self,X,y):
        
        self.features_info={}
    
        #开始筛选
        self.features_info['1.orgin']=X.columns.tolist()
        print('1.start_____________________________________complete')
        
        if self.na_pct is None:
            pass
        elif self.na_pct<1 and self.na_pct>0:
            self.features_info['2.filterbyNA']=self.filterByNA(X[self.features_info[max(list(self.features_info.keys()))]])
            print('2.filterbyNA________________________________complete')
        else:
            raise IOError("na_pct in (0,1)")
        

        
        if self.variance is None or self.unique_pct is None:
            pass        
        elif self.variance>=0 and self.unique_pct>=0 and self.unique_pct<1:
            self.features_info['3.filterbyVariance']=self.filterByUnique(X[self.features_info[max(list(self.features_info.keys()))]])+self.fliterByVariance(X[self.features_info[max(list(self.features_info.keys()))]])
            print('3.filterbyVariance_________________________complete')
        else:
            raise IOError("variance in [0,inf) and unique_pct in [0,1)")                          
        
        
        if self.chi2_pvalue and self.oneway_pvalue:
            self.features_info['4.filterbyChi2Oneway']=self.filterByChisquare(X[self.features_info[max(list(self.features_info.keys()))]],y)+self.filterByOneway(X[self.features_info[max(list(self.features_info.keys()))]],y)         
            print('4.filterbyChi2Oneway______________________complete')
            
        if self.tree_imps is None:
            pass
        elif self.tree_imps>=0:            
            self.features_info['5.filterbyTrees']=self.filterByTrees(X[self.features_info[max(list( self.features_info.keys()))]],y)
            print('5.filterbyTrees_____________________________complete')
        else:
            raise IOError("tree_imps in [0,inf)")         

        print('Done_______________________________________________')        
        
        return self
    
    def filterByNA(self,X):
        
        """ 
        缺失值处理
        """ 
        NAreport=X.isnull().sum().div(X.shape[0])
        return NAreport[NAreport<=self.na_pct].index.tolist() #返回满足缺失率要求的列名
    
    def filterByUnique(self,X):
        """ 
        唯一值处理-分类变量
        """     
        X_categoty=X.select_dtypes(exclude='number')
        if X_categoty.columns.size:
            unique_pct=X_categoty.apply(lambda x:x.value_counts().div(X_categoty.shape[0]).max())    
            return unique_pct[unique_pct<self.unique_pct].index.tolist()
        else:
            return []
    
    def fliterByVariance(self,X):
        """ 
        方差处理-连续变量,缺失值将被忽略
        """     
        X_numeric=X.select_dtypes('number')
        if X_numeric.columns.size:
            support_vars=VarianceThreshold(threshold=self.variance).fit(X_numeric).get_support()        
            return X.select_dtypes('number').columns[support_vars].tolist()
        else:
            return []

    def filterByChisquare(self,X,y):
        
        """ 
        特征选择-分类变量:卡方值
        """
        
        X_categoty=X.select_dtypes(exclude='number')
        if X_categoty.columns.size:      
            X_categoty_encode=OrdinalEncoder().fit_transform(X_categoty.replace(np.nan,'NAN'))
            p_values=chi2(X_categoty_encode,y)[1]
            return  X_categoty_encode.columns[p_values<self.chi2_pvalue].tolist()#返回满足卡方值要求的列名
        else:
            return []
    
    def filterByOneway(self,X,y):
        
        """ 
        特征选择-连续变量:方差分析(假定方差齐性)
        """
        X_numeric=X.select_dtypes('number')
        if X_numeric.columns.size:
            p_values_pos=f_oneway(X_numeric.replace(np.nan,1e10),y)[1]
            p_values_neg=f_oneway(X_numeric.replace(np.nan,-1e10),y)[1]        
            return X_numeric.columns[(p_values_pos<self.oneway_pvalue) | (p_values_neg<self.oneway_pvalue)].tolist() #返回满足方差分析的列名
        else:
            return []

    def filterByTrees(self,X,y):
        """ 
        特征选择:树模型
        """
        X_numeric=X.select_dtypes('number')
        X_categoty=X.select_dtypes(exclude='number')
        
        if X_categoty.columns.size:
            X_categoty_encode=OrdinalEncoder().fit_transform(X_categoty).sub(1)
            X_new=pd.concat([X_numeric,X_categoty_encode],axis=1)

            lgb=LGBMClassifier(
                boosting_type='gbdt',
                objective = 'binary',
                learning_rate=0.1,
                n_estimators=self.tree_size,
                subsample=0.7,
                colsample_bytree=1,
            ).fit(X_new,y,categorical_feature=X_categoty_encode.columns.tolist())         

            lgb_imps=lgb.booster_.feature_importance(importance_type='gain')
        
            return X_new.columns[lgb_imps>self.tree_imps].tolist()
        
        elif X_numeric.columns.size:
            
            lgb=LGBMClassifier(
                boosting_type='gbdt',
                objective = 'binary',
                learning_rate=0.1,
                n_estimators=self.tree_size,
                subsample=0.7,
                colsample_bytree=1,
            ).fit(X_numeric,y)         

            lgb_imps=lgb.booster_.feature_importance(importance_type='gain')
        
            return X_numeric.columns[lgb_imps>self.tree_imps].tolist()
        
        else:
            return []

        
class selection_iv(TransformerMixin):
    
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
    

class selection_corr(TransformerMixin):
    
    def __init__(self,y='target',corr_limit=0.8):
        """ 
        相关系数筛选
        Parameters:
        ----------
            corr_limit:float,相关系数阈值,当两个特征相关性高于该阈值,将剔除掉IV较低的一个       
        Attribute:
        ----------
            features_info:dict,每一步筛选的特征进入记录
        """
        self.y=y
        self.corr_limit=corr_limit
        
    def transform(self,X,y=None):
        return X[self.var_keep]
          
    def fit(self,X,y):
        """ 
        变量筛选
        Parameters:
        ----------
            varbin:分箱结果,计算特征IV使用,由sc.woebin产生      
        """          
        self.filterByCorr(X,y)
        
        return self
    
    def filterByCorr(self,X,y):
        """
        特征共线性检查,将剔除共线性较强但iv较低的特征,保留共线性较强但iv较高的特征 
        Parameters:
        ----------      
        X:训练数据
        """        
#         #递归式剔除,速度较慢        
#         iv_t=toad.quality(X.join(y),target=self.y,iv_only=True)[['iv']]

#         self.var_keep=[]
#         count=0
#         while (X.corr().abs()>corr_limit).sum().sum()>X.columns.size or count==0: #相关系数矩阵中所有特征的相关系数需小于阈值时才停止迭代(非自身)
#             print(X.columns.size)
#             count=count+1
            
#             #相关性系数
#             corrtable=X.corr()
#             var_corr_max=corrtable.apply(lambda x:x[x.index!=x.name].abs().max())
#             var_highcorr=var_corr_max[var_corr_max>corr_limit].index.tolist()
#             #var_lowcorr=var_corr_max[var_corr_max<=corr_limit].index.tolist()

#             self.var_del=[]
            
#             for i in var_highcorr:                
#                 pairs=iv_t.join(corrtable[i][corrtable[i]>corr_limit],how='right').sort_values(i) #变量与其corr最高的变量
#                 #self.var_del.append(pairs[0:1].index[0]) #剔除IV较小的那个
#                 var_del=pairs[0:1].index[0]
            
#             X=X.drop(var_del,axis=1)
        
        #使用toad
        X_drop=toad.selection.drop_corr(X.join(y),target=self.y,threshold=self.corr_limit).drop(self.y,axis=1)
    
        self.var_keep=X_drop.columns.tolist()

        
        
class selection_optbin(TransformerMixin):
    
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
    
    

class getWOE(TransformerMixin):
    
    def __init__(self):
        pass
        
    def transform(self,X,y):
        """ 
        变量筛选
        """
        X_woe=pd.DataFrame(index=X.index).join(sc.woebin_ply(dt=X.join(y),bins=self.varbin,no_cores=None))
        return X_woe[X_woe.columns[X_woe.columns.str.contains('_woe')]]
          
    def fit(self,varbin):
        self.varbin=varbin
        return self       
        
    
    
# class getEncoding(TransformerMixin):
    
#     def __init__(self,varbin,combiner_toad,method='woe_toad'):
#         '''
#         进行数据编码       
#         Params:
#         ------
#         varbin:sc.woebin_bin产生的分箱信息dict
#         combiner_toad:toad分箱后产生的toad.transform.Combiner类        
#         method:可选"woe_toad","woe_sc"
#             + woe_toad:使用toad完成编码,注意toad的woe编码对不支持X中存在np.nan
#             + woe_sc:使用scorecardpy完成编码
 
#         Attributes
#         ------
#         X_bins_toad:method='woe_toad'时产生的toad的分箱中间数据
        
#         Examples
#         ------      
        
#         '''        
        
#         self.method=method
#         self.varbin_sc=varbin         
#         self.combiner_toad=combiner_toad
        
#     def transform(self,X,y):
#         """ 
#         变量筛选
#         """
#         if self.method=="woe_sc":
            
#             out=self.X_woe=[self.X_woe.columns[self.X_woe.columns.str.contains('_woe')]]
            
#         elif self.method=="woe_toad":
            
#             out=self.woe_ply_toad.transform(X,y)
            
#         else:
#             raise IOError('method in ("woe_toad","woe_sc")')
        
#         return out
          
#     def fit(self,X,y):  
        
#         if self.method=="woe_sc":
            
#             X_woe_raw=sc.woebin_ply(dt=X.join(y),bins=self.varbin,no_cores=None)
            
#             self.X_woe=pd.DataFrame(index=X.index).join(X_woe_raw)                                   
            
#         elif self.method=="woe_toad":
            
#             self.woe_ply_toad = toad.transform.WOETransformer()

#             self.X_bins_toad=self.varbin.transform(X.join(y),labels=False)
            
#             self.woe_ply_toad.fit(self.X_bins_toad,y)
        
#         else:
#             raise IOError('method in ("woe_toad","woe_sc")')
        
#         return self        
                