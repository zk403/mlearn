from sklearn.base import TransformerMixin
from sklearn.feature_selection import VarianceThreshold,f_oneway,chi2
from lightgbm import LGBMClassifier
from category_encoders.ordinal import OrdinalEncoder
import numpy as np
import pandas as pd
import scorecardpy as sc

class selection_pre(TransformerMixin):
    
    def __init__(self,
                 na_pct=0.99,
                 unique_pct=0.99,
                 variance=0,
                 chi2_pvalue=0.05,
                 oneway_pvalue=0.05,
                 tree_imps=0):
        """ 
        预筛选,适用于二分类模型
        Parameters:
        ----------
            na_pct:float,(0,1),默认0.99,缺失率高于na_pct的列将被筛除
            unique_pct:float,(0,1),默认0.99,唯一值占比高于unique_pct的列将被筛除,将忽略缺失值,unique_pct与variance需同时输入
            variance:float,默认0,方差低于variance的列将被筛除,将忽略缺失值,unique_pct与variance需同时输入
            chi2_pvalue:float,(0,1),默认0.05,大于chi2_pvalue的列将被剔除,缺失值将被视为单独一类,chi2_pvalue与oneway_pvalue需同时输入
            oneway_pvalue:float,(0,1),默认0.05,缺失值将被填补为接近正负inf,将计算两次,两次结果中都较大于oneway_pvalue的列将被剔除,此外,该函数默认方差是齐性的,chi2_pvalue与oneway_pvalue需同时输入
            tree_imps:float,lightgbm树的梯度gain小于等于tree_imps的列将被剔除,默认0
            
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
        if self.na_pct>0 and self.na_pct<1:
            self.features_info['2.filterbyNA']=self.filterByNA(X[self.features_info[max(list(self.features_info.keys()))]])
        print('2.filterbyNA________________________________complete')
        
        if self.variance>=0 and self.unique_pct>=0 and self.unique_pct<1:
            self.features_info['3.filterbyVariance']=self.filterByUnique(X[self.features_info[max(list(self.features_info.keys()))]])+self.fliterByVariance(X[self.features_info[max(list(self.features_info.keys()))]])
        print('3.filterbyVariance_________________________complete')
        
        if self.chi2_pvalue and self.oneway_pvalue:
            self.features_info['4.filterbyChi2Oneway']=self.filterByChisquare(X[self.features_info[max(list(self.features_info.keys()))]],y)+self.filterByOneway(X[self.features_info[max(list(self.features_info.keys()))]],y) 
        print('4.filterbyChi2Oneway______________________complete')
            
        if self.tree_imps>=0:
            self.features_info['5.filterbyTrees']=self.filterByTrees(X[self.features_info[max(list( self.features_info.keys()))]],y)
        print('5.filterbyTrees_____________________________complete')
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
                n_estimators=500,
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
                n_estimators=500,
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
        finebin=sc.woebin(df,y=self.y,check_cate_num=False,
                      breaks_list=self.breaks_list)

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
    

class selection_corr(TransformerMixin):
    
    def __init__(self,bins,corr_limit=0.8,corr_method='pearson'):
        """ 
        相关系数筛选
        Parameters:
        ----------
            bins:sc.woebin产生的分箱对象,用于计算IV
            corr_limit:float,相关系数阈值,当两个特征相关性高于该阈值,将剔除掉IV较低的一个
            corr_method:str,相关性计算方式,默认pearson,可选spearman        
        Attribute:
        ----------
            features_info:dict,每一步筛选的特征进入记录
        """
        self.bins=bins
        self.corr_limit=corr_limit
        self.corr_method=corr_method
        
    def transform(self,X,y=None):
        return X[self.var_keep]
          
    def fit(self,X):
        """ 
        变量筛选
        Parameters:
        ----------
            varbin:分箱结果,计算特征IV使用,由sc.woebin产生      
        """      
        self.filterByCorr(X)
        return self
    
    def filterByCorr(self,X):
        """
        特征共线性检查,将剔除共线性较强但iv较低的特征,保留共线性较强但iv较高的特征 
        Parameters:
        ----------      
        X:训练数据
        """
        corr_limit=self.corr_limit
        
        #计算IV
        finalbindf=pd.concat(self.bins)
        finalbindf['variable']=finalbindf.variable+'_woe'
        fimp=finalbindf.groupby('variable')[['bin_iv']].sum()
        
        self.var_keep=[]
        count=0
        while (X.corr().abs()>corr_limit).sum().sum()>X.columns.size or count==0: #相关系数矩阵中所有特征的相关系数需小于阈值时才停止迭代(非自身)
            count=count+1
            
            #相关性系数
            corrtable=X.corr(method=self.corr_method)
            var_corr_max=corrtable.apply(lambda x:x[x.index!=x.name].abs().max())
            var_highcorr=var_corr_max[var_corr_max>corr_limit].index.tolist()
            #var_lowcorr=var_corr_max[var_corr_max<=corr_limit].index.tolist()

            self.var_del=[]
            
            for i in var_highcorr:
                pairs=fimp.join(corrtable[i][corrtable[i]>corr_limit],how='right').sort_values(i,ascending=False).head(2) #变量与其corr最高的变量
                self.var_del.append(pairs[pairs.bin_iv.eq(pairs['bin_iv'].min())].index[0]) #选择IV较大的那个
            
            X=X.drop(self.var_del,axis=1)
        
        self.var_keep=X.columns.tolist()