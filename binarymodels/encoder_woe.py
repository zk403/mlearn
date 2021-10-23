# -*- coding: utf-8 -*-
from sklearn.base import TransformerMixin
#from category_encoders.ordinal import OrdinalEncoder
#import numpy as np
import pandas as pd
import scorecardpy as sc



class woeTransformer(TransformerMixin):
    
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
                