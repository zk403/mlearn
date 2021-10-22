from sklearn.base import TransformerMixin
from sklearn.feature_selection import VarianceThreshold,f_oneway,chi2
from lightgbm import LGBMClassifier
from category_encoders.ordinal import OrdinalEncoder
import numpy as np
import pandas as pd
import toad

class preSelector(TransformerMixin):
    
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
    

class corrSelector(TransformerMixin):
    
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

        
        

    
    

