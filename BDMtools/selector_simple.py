from sklearn.base import TransformerMixin
from sklearn.feature_selection import VarianceThreshold,f_oneway,chi2
from lightgbm import LGBMClassifier
from category_encoders.ordinal import OrdinalEncoder
import numpy as np
import pandas as pd
import toad
import os
from glob import glob


class preSelector(TransformerMixin):
    
    """ 
    线性预筛选,适用于二分类模型
    Parameters:
    ----------
        na_pct:float or None,(0,1),默认0.99,缺失率高于na_pct的列将被筛除，设定为None将跳过此步骤
        unique_pct:float or None,(0,1),默认0.99,唯一值占比高于unique_pct的列将被筛除,将忽略缺失值,unique_pct与variance需同时输入，任一设定为None将跳过此步骤
        variance:float or None,默认0,方差低于variance的列将被筛除,将忽略缺失值,unique_pct与variance需同时输入，任一设定为None将跳过此步骤
        chif_pvalue:float or None,(0,1),默认0.05,大于chif_pvalue的列将被剔除,缺失值将被视为单独一类,为None将跳过此步骤
        tree_imps:float or None,lightgbm树的梯度gain小于等于tree_imps的列将被剔除,默认0，设定为None将跳过此步骤
        tree_size:int,lightgbm树个数,若数据量较大可降低树个数，若tree_imps为None时该参数将被忽略
        iv_limit:float or None使用toad.quality进行iv快速筛选的iv阈值
        out_path:str or None,模型报告路径,将预筛选过程每一步的筛选过程输出到模型报告中
        special_values:list,特殊值指代值列表,其将被替换为np.nan
        keep:list or None,需保留列的列名list
        
    Attribute:
    ----------
        features_info:dict,每一步筛选的特征进入记录
    """    
    
    
    def __init__(self,na_pct=0.99,unique_pct=0.99,variance=0,chif_pvalue=0.05,tree_imps=0,
                 tree_size=100,iv_limit=0.02,out_path="report",special_values=[np.nan],keep=None
                 ):

        self.na_pct=na_pct
        self.unique_pct=unique_pct #
        self.variance=variance
        self.chif_pvalue=chif_pvalue #
        self.tree_imps=tree_imps #
        self.tree_size=tree_size
        self.iv_limit=iv_limit
        self.out_path=out_path
        self.special_values=special_values
        self.keep=keep
        
    def transform(self,X,y=None):
        """ 
        变量筛选
        """  
        
        if self.keep:
        
            keep_col=self.features_info[max(list(self.features_info.keys()))] + self.keep  
            
        else:
            
            keep_col=self.features_info[max(list(self.features_info.keys()))]            
        
        return X[keep_col]
       
    def fit(self,X,y):
        
        X=X.replace(self.special_values,np.nan)
        
        X=X.drop(self.keep,axis=1) if self.keep else X
        
        self.features_info={}
    
        #开始筛选
        self.features_info['1.orgin']=X.columns.tolist()
        
        print('1.start______________________________________complete')
        
        if self.na_pct==None:
            
            pass
        
        elif self.na_pct<1 and self.na_pct>0:
            
            var_pre_na=self.features_info[max(list(self.features_info.keys()))]
            
            self.features_info['2.filterbyNA']=self.filterByNA(X[var_pre_na])
            
            print('2.filterbyNA_____________________________complete')
            
        else:
            
            raise IOError("na_pct in (0,1)")
                
        if self.variance==None or self.unique_pct==None:            
            
            pass        
        
        elif self.variance>=0 and self.unique_pct>=0 and self.unique_pct<1:
            
            var_pre_vari=self.features_info[max(list(self.features_info.keys()))]
            
            self.features_info['3.filterbyVariance']=self.filterByUnique(X[var_pre_vari])+self.fliterByVariance(X[var_pre_vari])
            
            print('3.filterbyVariance_______________________complete')
            
        else:
            
            raise IOError("variance in [0,inf) and unique_pct in [0,1)")                          
        
        
        if self.chif_pvalue==None:
            
            pass
        
        elif self.chif_pvalue>0 and self.chif_pvalue<=1: 
            
            var_pre_chi=self.features_info[max(list(self.features_info.keys()))]
            
            self.features_info['4.filterbyChi2Oneway']=self.filterByChisquare(X[var_pre_chi],y)+self.filterByOneway(X[var_pre_chi],y)         
            
            print('4.filterbyChi2Oneway_____________________complete')
        
        else:
            
            raise IOError("pvalue in (0,1]") 
            
        if self.tree_imps==None:
            
            pass
        
        elif self.tree_imps>=0:            
            
            var_pre_tree=self.features_info[max(list(self.features_info.keys()))]
            
            self.features_info['5.filterbyTrees']=self.filterByTrees(X[var_pre_tree],y)
            
            print('5.filterbyTrees__________________________complete')
            
        else:
            
            raise IOError("tree_imps in [0,inf)")         
   
        
        if self.iv_limit==None:   
            
            pass
        
        elif self.iv_limit>=0:
            
            var_pre_iv=self.features_info[max(list(self.features_info.keys()))]
            
            self.features_info['6.filterbyIV']=self.filterbyIV(X[var_pre_iv],y)
            
            print('6.filterbyIV_____________________________complete')
            
        else:
            
            raise IOError("iv_limit in [0,inf)")      
        
        print('Done_________________________________________________')  
        
        #打印筛选汇总信息
        for key in self.features_info:
            
            print('步骤{},保留的特征数:{}'.format(key,len(self.features_info[key])))
        
        #输出报告    
        if self.out_path: 
            
            self.preSelector_report=pd.concat([pd.Series(self.features_info[key],name=key) for key in self.features_info.keys()],axis=1)
                
            self.writeExcel() 
        
                    
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
        X_categoty=X.select_dtypes(include='object')
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
            return X_numeric.columns[support_vars].tolist()
        else:
            return []

    def filterByChisquare(self,X,y):
        
        """ 
        特征选择-分类变量:卡方值
        """
        
        X_categoty=X.select_dtypes(include='object')
        if X_categoty.columns.size:      
            X_categoty_encode=OrdinalEncoder().fit_transform(X_categoty.replace(np.nan,'NAN'))
            p_values=chi2(X_categoty_encode,y)[1]
            return  X_categoty_encode.columns[p_values<self.chif_pvalue].tolist()#返回满足卡方值要求的列名
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
            return X_numeric.columns[(p_values_pos<self.chif_pvalue) | (p_values_neg<self.chif_pvalue)].tolist() #返回满足方差分析的列名
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
        
    
    def filterbyIV(self,X,y):
        
        iv_t=toad.selection.IV(X,y,n_bins=20).T

        return iv_t[iv_t[0]>self.iv_limit].index.tolist()

    
    
    def writeExcel(self):
        
        if not glob(self.out_path):
            
            os.mkdir(self.out_path)
                
        if not glob(self.out_path+"/model_report.xlsx"):
            
            #print(self.out_path+"/model_report.xlsx")
            
            writer = pd.ExcelWriter(self.out_path+"/model_report.xlsx")       
            pd.DataFrame(None).to_excel(writer,sheet_name='summary')
            writer.save()                    
            
        writer=pd.ExcelWriter(self.out_path+'/model_report.xlsx',
                              mode='a',
                              if_sheet_exists='replace',
                              #engine_kwargs={'mode':'a','if_sheet_exists':'replace'},
                              engine='openpyxl')
        
        self.preSelector_report.to_excel(writer,sheet_name='3.preSelect')
            
        writer.save()     
        print('to_excel done') 
    

class corrSelector(TransformerMixin):
    
    """ 
    相关系数筛选
    Parameters:
    ----------
        corr_limit:float,相关系数阈值
        by:str or pd.Series,
            + ’IV‘:按照iv筛选
            + pd.Series:用户自定义权重,要求index为列名，value为权重值
        keep:需保留的列名list
        
    Attribute:
    ----------
        features_info:dict,每一步筛选的特征进入记录
    """    
    
    def __init__(self,corr_limit=0.8,by='IV',keep=None):

        self.corr_limit=corr_limit
        self.by=by
        self.keep=keep
        
    def transform(self,X,y=None):
        
        if self.keep and isinstance(self.keep,list):
            
            self.keep_col=list(set(self.keep_col+self.keep))
        
        return X[self.keep_col]
          
    def fit(self,X,y):
        """ 
        变量筛选
        Parameters:
        ----------
            varbin:分箱结果,计算特征IV使用,由sc.woebin产生      
        """          
        self.keep_col=self.filterByCorr(X,y)
        
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
        X_drop=toad.selection.drop_corr(X.join(y),target=y.name,threshold=self.corr_limit,by=self.by).drop(y.name,axis=1)
    
        keep_col=X_drop.columns.tolist()
        
        return keep_col

        
        

    
    

