#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:46:29 2020

@author: kezeng
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import pearsonr,spearmanr
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

class featureCluster(BaseEstimator):
    
    def __init__(self,n_clusters=5,distance_threshold=None,linkage='average',distance_metrics='pearson',scale=True):
        '''
        评分转换
        Parameters:
        --
            n_clusters=5:int,聚类数量
            distance_threshold=None:距离阈值
            linkage='average':层次聚类连接方式     
            distance_metrics='pearson':距离衡量方式
            scale=True:聚类前是否进行数据标准化
        Attribute:    
        --
            components_infos
            rsquare_infos
            
        '''      
        
        self.distance_metrics=distance_metrics
        self.linkage=linkage
        self.n_clusters=n_clusters
        self.distance_threshold=distance_threshold
        self.scale=scale
        
    def distance(self):
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
            raise ValueError('distances support:r2,pearson,spearman ')
  
    def fit(self,X,y=None):  
    
        self.X=X.copy()            
        self.model=self.featurecluster()
        self.components_infos=self.getComponentsInfos()
        self.rsquare_infos=self.getRsquareInfos()
        return self        
    
    def transform(self,X):
        columns=self.rsquare_infos.sort_values(['Cluster','1-R2Ratio']).groupby('Cluster').head(1).index 
        return X[columns]
    
    def featurecluster(self):
        #变量聚类
        custom_distance=self.distance()
        linkage=self.linkage
        n_clusters=self.n_clusters
        distance_threshold=self.distance_threshold
        
        # 生成距离矩阵
        m = pd.DataFrame(pairwise_distances(self.X.T, self.X.T, metric=custom_distance)) #距离衡量
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
        custom_distance=self.distance()
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
        m = pd.DataFrame(pairwise_distances(X.T, X.T, metric=custom_distance)) #使用相关系数距离衡量

        #affinity设定为'precomputed',linkage设定为非ward
        model = FeatureAgglomeration(distance_threshold=0,n_clusters=None,affinity='precomputed',linkage=linkage).fit(m)
        plt.title('Hierarchical Clustering Dendrogram')
        plot(model)
        plt.xlabel("Variable index")
        plt.ylabel("distance")
        
    def getComponentsInfos(self):
        
        X=self.X
        
        fclusters=self.model.labels_
        
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
       
    def getRsquareInfos(self):
        
        X=self.X
        
        fclusters=self.model.labels_
        
        #提取各个类的主成分
        label_components={}
        for label in np.unique(fclusters):
            cluster_features=X.columns[fclusters==label]

            if len(cluster_features)>1:
                label_components[label]=pd.Series(PCA(n_components=None).fit_transform(X[cluster_features])[:,0],name=label)
            else:
                label_components[label]=pd.Series(X[cluster_features[0]].ravel())        
        
        label_components_df=pd.concat(label_components,1,ignore_index=True)
        
        #计算类间,类内差异指标
        #label_neigbor={} #邻近类
        featrues_r2={} #特征的类内R方
        neigbors_r2={} #特征的类间R方(最邻近类)

        for label in np.unique(fclusters):
            cluster_features=X.columns[fclusters==label]
            #计算所有特征的类内,类间指标
            for feature in cluster_features:

                #计算类内的R方,类内所有特征与类主成分回归的R方
                #ols_in=LinearRegression().fit(label_components_df[[label]],X[feature])
                #r2_in=ols_in.score(label_components_df[[label]],X[feature])      
                r2_in=np.corrcoef(label_components_df[label],X[feature])[0, 1] ** 2
                featrues_r2[feature]=r2_in

                #计算类间的R方,类内所有特征与最邻近类的主成分回归的R方
                #ols_between=LinearRegression().fit(label_components_df[[label_neigbor[label]]],X[feature])
                #r2_between=ols_between.score(label_components_df[[label_neigbor[label]]],X[feature])
                r2_between=[]
                for components in label_components:
                    if components !=label: #非自类主成分   
                        r2_between.append(np.corrcoef(label_components_df[components],X[feature])[0, 1] ** 2) 
                neigbors_r2[feature]=max(r2_between) if len(r2_between) > 0 else 0

        #汇总结果输出
        report=pd.concat(
                    [pd.Series(fclusters,index=X.columns,name='Cluster'),
                     pd.Series(featrues_r2,name='R2_Featrues'),
                     pd.Series(neigbors_r2,name='R2_Neigbor')],1
            )
        report['1-R2Ratio']=(1-report['R2_Featrues'])/(1-report['R2_Neigbor']) 
        
        return report