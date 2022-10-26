#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 23:08:38 2021

@author: zengke
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_array_like
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import pearsonr,spearmanr
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from BDMLtools.base import Base


class faSelector(Base,TransformerMixin):
    
    """
    变量聚类:基于sklearn.cluster.FeatureAgglomeration
    
        
    Parameters:
    --
        n_clusters=5:int,聚类数量
            + int,指定列聚类数量
            + 'auto',自动确定聚类列数量,在进行变量聚类前先进行相关性筛选，指定相关性阈值下留存的列个数将作为变量聚类的聚类数量
        corr_limit:n_cluster='auto'时,相关性筛选的相关性阈值,经验上corr_limit=0.6左右下的分群列特征值最大的前两个主成分累积解释占比在0.7左右               
        distance_threshold=None:距离阈值
        linkage='average':层次聚类连接方式     
        distance_metrics='pearson':距离衡量方式,可选pearson，spearman，r2
        scale=True:聚类前是否进行数据标准化,
        by='r2-ratio',聚类后的特征筛选方式
            + 'r2-ratio':与SAS一致，将筛选每一类特征集中r2-ratio最小的特征
            + pd.Series:用户自定义权重,要求index为列名，value为权重值，例如iv,ks等                  
        is_greater_better=True,若by参数内容为用户自定义权重,is_greater_better=True表示权重越高特征越重要,反之则越不重要
        keep=None,需保留的列名list
           
    Attribute:    
    --
        model:sklearn.cluster.FeatureAgglomeration object,变量聚类对象
        components_infos:聚类的类主成分特征值信息
        rsquare_infos:各个变量的R2信息
        
    """    
    
    def __init__(self,n_clusters=5,corr_limit=0.6,distance_threshold=None,linkage='average',
                 scale=True,distance_metrics='pearson',by='r2-ratio',is_greater_better=True,keep=None):
     
        
        self.distance_metrics=distance_metrics
        self.linkage=linkage
        self.n_clusters=n_clusters
        self.corr_limit=corr_limit
        self.distance_threshold=distance_threshold
        self.scale=scale
        self.by=by
        self.is_greater_better=is_greater_better
        self.keep=keep
        
        self._is_fitted=False
        
    def _distance(self):
        
        custom_distance=self.distance_metrics
        
        if custom_distance=='pearson':
            
            def pearson_distance(x, y):
                """
                pearson距离,相关系数绝对值的相反数
                """
                r,pvalue=pearsonr(x,y)
                return 1-np.abs(r)
            
            return pearson_distance
        
        elif custom_distance=='spearman':
            
            def spearman_distance(x, y):
                """
                spearman距离,相关系数绝对值的相反数
                """
                r,pvalue=spearmanr(x,y)
                return 1-np.abs(r) 
            
            return spearman_distance
            
        elif custom_distance=='r2':
        
            def r2_distance(x,y):
                """
                r2距离,原始r2的相反数
                """
                r2=np.corrcoef(x,y)[0, 1] ** 2
                return 1-r2
            
            return r2_distance
        
        else: 
            
            raise ValueError('distances support:r2,pearson,spearman')
            
  
    def fit(self,X,y):    
        
        self._check_data(X, y)
        
        #drop constant columns
        X = X.loc[:,X.apply(lambda col: False if col.unique().size==1 else True)]
        
        if X.columns.size == 0:
            
            raise ValueError("All columns in X are constant.")      
        
        if isinstance(self.n_clusters,int):
        
            self.model=self._featurecluster(X,self._distance(),self.linkage,self.n_clusters,self.distance_threshold)
            
        elif self.n_clusters=='auto':
            
            self.n_clusters=self._get_clusternum(X,threshold=self.corr_limit)
            
            print('n_clusters set to '+str(self.n_clusters))
            
            self.model=self._featurecluster(X,self._distance(),self.linkage,self.n_clusters,self.distance_threshold)
            
        else:
            
            raise ValueError("n_clusters in (int,'auto')")
            
        self.components_infos=self._getComponentsInfos(X,self.model.labels_)
        
        
        if isinstance(self.by,str):
        
            self.rsquare_infos=self._getRsquareInfos(X,self.model.labels_)
        
        elif is_array_like(self.by):

            self.rsquare_infos=self._getRsquareInfos(X,self.model.labels_).join(self.by)
        
        else:
            
            warnings.warn('by in (r2-ratio,pd.Series),use r2-ratio instead')  
        
            self.rsquare_infos=self._getRsquareInfos(X,self.model.labels_)
        
        self._is_fitted=True
        
        return self        
    
    def _get_clusternum(self,X,threshold = 0.7):
        
        """
        get columns number by correlation,rewrite from toad 0.1.0
        """
        cols = X.columns.copy()
    
        corr = X[cols].corr().abs()
    
        drops = []
    
        # get position who's corr greater than threshold
        ix, cn = np.where(np.triu(corr.values, 1) > threshold)
    
        # if has position
        if len(ix):
            # get the graph of relationship
            graph = np.hstack([ix.reshape((-1, 1)), cn.reshape((-1, 1))])
    
            uni, counts = np.unique(graph, return_counts = True)

            weights = np.ones(len(cols))
      
            while(True):
                # TODO deal with circle
    
                # get nodes with the most relationship
                nodes = uni[np.argwhere(counts == np.amax(counts))].flatten()
    
                # get node who has the min weights
                n = nodes[np.argsort(weights[nodes])[0]]
    
                # get nodes of 1 degree relationship of n
                i, c = np.where(graph == n)
                pairs = graph[(i, 1-c)]
    
                # if sum of 1 degree nodes greater than n
                # then delete n self
                # else delete all 1 degree nodes
                if weights[pairs].sum() > weights[n]:
                    dro = [n]
                else:
                    dro = pairs.tolist()
    
                # add nodes to drops list
                drops += dro
    
                # delete nodes from graph
                di, _ = np.where(np.isin(graph, dro))
                graph = np.delete(graph, di, axis = 0)
    
                # if graph is empty
                if len(graph) <= 0:
                    break
    
                # update nodes and counts
                uni, counts = np.unique(graph, return_counts = True)
    
        drop_list = corr.index[drops].values

        return len(cols)-len(drop_list)
    
    
    def transform(self,X):
        
        self._check_is_fitted()
        self._check_X(X)
        
        if isinstance(self.by,str):
        
            columns=self.rsquare_infos.sort_values(['Cluster','1-R2Ratio'],ascending=[True,True]).groupby('Cluster').head(1).index 
        
        elif is_array_like(self.by):

            name=self.by.name
            
            if self.is_greater_better:
                
                columns=self.rsquare_infos.sort_values(['Cluster',name],ascending=[True,False]).groupby('Cluster').head(1).index 
                
            else:
                 
                columns=self.rsquare_infos.sort_values(['Cluster',name],ascending=[True,True]).groupby('Cluster').head(1).index        

        else:
            
            warnings.warn('by in (r2-ratio,pd.Series),use r2-ratio instead')  
            
            columns=self.rsquare_infos.sort_values(['Cluster','1-R2Ratio'],ascending=[True,True]).groupby('Cluster').head(1).index   
            
        if self.keep and isinstance(self.keep,list):
            
            columns=list(set(columns.tolist()+self.keep))
      
        return X[columns]
    
    
    def _featurecluster(self,X,custom_distance,linkage,n_clusters,distance_threshold):
        #变量聚类
        #custom_distance=self.distance()
        #linkage=self.linkage
        #n_clusters=self.n_clusters
        #distance_threshold=self.distance_threshold
        
        # 生成距离矩阵
                
        if self.scale:
            
            X_t=np.transpose(StandardScaler().fit_transform(X))
            
        else:
            
            X_t=X.T
        
        m = pd.DataFrame(pairwise_distances(X_t,X_t,metric=custom_distance)) #距离衡量
        
        if distance_threshold:
            
            model = FeatureAgglomeration(distance_threshold=distance_threshold,n_clusters=None,affinity='precomputed',linkage=linkage).fit(m)
            
        elif n_clusters:
            
            model = FeatureAgglomeration(n_clusters=n_clusters,affinity='precomputed',linkage='average').fit(m)
            
        else:
            
            raise ValueError('set n_clusters or distance_threshold')

        if np.unique(model.labels_).size==1:
            
            raise ValueError('Only 1 cluster,reset the distance_threshold or n_clusters')
        
        return model
    
    def plot_dendrogram(self,X):
        
        custom_distance=self._distance()
        linkage=self.linkage

        def plot(model, **kwargs):
            # Create linkage matrix and then plot the dendrogram

            # create the counts of samples under each node
            counts = np.zeros(model.children_.shape[0])
            n_samples = len(model.labels_)
            for i, merge in enumerate(model.children_):
                current_count = 0
                for child_idx in merge:
                    if child_idx < n_samples:
                        current_count += 1  # leaf node
                    else:
                        current_count += counts[child_idx - n_samples]
                counts[i] = current_count

            linkage_matrix = np.column_stack([model.children_, model.distances_,counts]).astype(float)

            # Plot the corresponding dendrogram
            dendrogram(linkage_matrix, **kwargs)

        #m为预计算的距离矩阵
        if self.scale:
            
            X_t=np.transpose(StandardScaler().fit_transform(X))
            
        else:
            
            X_t=X.T
        
        m = pd.DataFrame(pairwise_distances(X_t, X_t, metric=custom_distance)) #使用相关系数距离衡量

        #affinity设定为'precomputed',linkage设定为非ward
        model = FeatureAgglomeration(distance_threshold=0,n_clusters=None,affinity='precomputed',linkage=linkage).fit(m)
        plt.title('Hierarchical Clustering Dendrogram')
        plot(model)
        plt.xlabel("Variable index")
        plt.ylabel("distance")
        
    def _getComponentsInfos(self,X,fclusters):
        
        #fclusters=self.model.labels_
        
        #提取各个类的主成分
        label_components={}
        components_infos=pd.DataFrame() #主成分报告
        for label in np.unique(fclusters):
            cluster_features=X.columns[fclusters==label]

            if len(cluster_features)>1:

                pca=PCA(n_components=None).fit(X[cluster_features])

                components_info=pd.DataFrame({
                    'cluster':[label],
                    'n_vars':[len(cluster_features)],
                    'eigval1':[pca.explained_variance_[0]], #第一主成分对应特征值
                    'eigval2':[pca.explained_variance_[1]], #第二主成分对应特征值
                    'explained_ratio':[pca.explained_variance_ratio_[0]] #第一大主成分累计解释方差比例
                })   

                label_components[label]=pd.Series(pca.transform(X[cluster_features])[:,0],name=label)

                components_infos=pd.concat([components_infos,components_info],ignore_index=True)
           
            #群内只有一个特征
            else:
                components_info=pd.DataFrame({
                    'cluster':[label],
                    'n_vars':[len(cluster_features)],
                    'eigval1':[1], #第一主成分对应特征值
                    'eigval2':[0], #第二主成分对应特征值
                    'explained_ratio':[1] #前两个主成分累计解释方差比例
                })   

                label_components[label]=pd.Series(X[cluster_features[0]].ravel())           
                components_infos=pd.concat([components_infos,components_info],ignore_index=True)
        
        self.label_components=label_components #所有类变量集合的第一主成分
        return components_infos
       
    def _getRsquareInfos(self,X,fclusters):
        
        #fclusters=self.model.labels_
        
        #提取各个类的主成分
        label_components={}
        for label in np.unique(fclusters):
            
            cluster_features=X.columns[fclusters==label]

            if len(cluster_features)>1:
                
                label_components[label]=pd.Series(PCA(n_components=None).fit_transform(X[cluster_features])[:,0],name=label)
           
            else:
                label_components[label]=pd.Series(X[cluster_features[0]].ravel())        
        
        label_components_df=pd.concat(label_components,axis=1,ignore_index=True)
        
        #计算类间,类内差异指标
        #label_neigbor={} #邻近类
        featrues_r2={} #特征的类内R方
        neigbors_r2={} #特征的类间R方(最邻近类)

        for label in np.unique(fclusters):
            
            cluster_features=X.columns[fclusters==label]
            #计算所有特征的类内,类间指标
            for feature in cluster_features:

                #计算类内的R方,类内所有特征与类主成分回归的R方

                r2_in=np.corrcoef(label_components_df[label],X[feature])[0, 1] ** 2                
                featrues_r2[feature]=r2_in

                #计算类间的R方,类内所有特征与最邻近类的主成分回归的R方
                r2_between=[]
                
                for components in label_components:
                    
                    if components !=label: #非自类主成分   
                    
                        r2_between.append(np.corrcoef(label_components_df[components],X[feature])[0, 1] ** 2) 
                        
                neigbors_r2[feature]=max(r2_between) if len(r2_between) > 0 else 0

        #汇总结果输出
        report=pd.concat(
                    [pd.Series(fclusters,index=X.columns,name='Cluster'),
                     pd.Series(featrues_r2,name='R2_Featrues'),
                     pd.Series(neigbors_r2,name='R2_Neigbor')],axis=1
            )
        report['1-R2Ratio']=(1-report['R2_Featrues'])/(1-report['R2_Neigbor']) 
        
        return report