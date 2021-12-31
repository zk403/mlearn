#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:19:07 2020

@author: kezeng
"""

from sklearn.base import TransformerMixin,BaseEstimator
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from sklearn.linear_model._logistic import LogisticRegression
from pandas.api.types import is_numeric_dtype,is_string_dtype
from BDMLtools.fun import raw_to_bin_sc,sp_replace
from joblib import Parallel,delayed


class stepLogit(BaseEstimator):
    
    '''
    逐步回归,请注意column name不能以数字开头
    Parameters:
    --
        custom_column=None:list,自定义列名,调整回归模型时使用,默认为None表示所有特征都会进行筛选
        no_stepwise=False,True时直接回归(不支持normalize=True)，不进行逐步回归筛选
        p_value_enter=.05:逐步法中特征进入的pvalue限制,默认0.05
        criterion='aic':逐步法筛选变量的准则,默认aic,可选bic
        normalize=False:是否进行数据标准化,默认False,若为True,则数据将先进行标准化,且不会拟合截距
        show_step=False:是否打印逐步回归过程
        max_iter=200,逐步回归最大迭代次数
        sample_weight=None,样本权重
        show_high_vif_only=False:True时仅输出vif大于10的特征,False时将输出所有特征的vif
    
    Attribute:    
    --
        logit_model:逐步回归的statsmodel结果对象,须先使用方法fit
        model_info: 回归结果报告,须先使用方法fit
        vif_info:pd.DataFrame,筛选后特征的方差膨胀系数,须先使用方法fit
    '''        
    
    def __init__(self,custom_column=None,no_stepwise=False,p_value_enter=.05,criterion='aic',
                 normalize=False,show_step=False,max_iter=200,sample_weight=None,
                 show_high_vif_only=False):
      
        self.custom_column=custom_column
        self.no_stepwise=no_stepwise
        self.p_value_enter=p_value_enter
        self.criterion=criterion
        self.normalize=normalize
        self.show_step=show_step
        self.max_iter=max_iter
        self.sample_weight=sample_weight
        self.show_high_vif_only=show_high_vif_only
        
    def predict_proba(self,X,y=None):
        '''
        模型预测,使用逐步回归模型预测,产生预测概率
        Parameters:
        --
        X:woe编码数据,pd.DataFrame对象,需与训练数据woe编码具有相同的特征
        '''      
        pred=self.logit_model.predict(X)
        return pred
    
    def transform(self,X,y=None,raw=True):     
        '''
        使用逐步回归进行特征筛选,返回逐步法筛选后的训练数据
        Parameters:
        --
        X:woe编码数据,pd.DataFrame对象,需与训练数据woe编码具有相同的特征
        '''        
        if raw:            
            return X[self.logit_model.params.index.tolist()[1:]]
        else:
            return X[[i[0:-4] for i in self.logit_model.params.index.tolist()[1:]]]
          
    def fit(self,X,y):
        '''
        拟合逐步回归
        Parameters:
        --
        X:woe编码训练数据,pd.DataFrame对象
        y:目标变量,pd.Series对象
        '''        
        if self.custom_column:
            
            if self.no_stepwise:
                
                formula = "{} ~ {} + 1".format(y.name,' + '.join(self.custom_column))
                
                self.logit_model=smf.glm(formula, data=X[self.custom_column].join(y),
                                         family=sm.families.Binomial(),
                                         freq_weights=self.sample_weight).fit(disp=0)
                
            else:
            
                self.logit_model=self.stepwise(X[self.custom_column].join(y),y.name,criterion=self.criterion,p_value_enter=self.p_value_enter,normalize=self.normalize,show_step=self.show_step,max_iter=self.max_iter) 
            
        else:
            
            if self.no_stepwise:
                
                formula = "{} ~ {} + 1".format(y.name,' + '.join(X.columns.tolist()))
                
                self.logit_model=smf.glm(formula, data=X.join(y),
                                         family=sm.families.Binomial(),
                                         freq_weights=self.sample_weight).fit(disp=0)                                                
            
            else:
                
                self.logit_model=self.stepwise(X.join(y),y.name,criterion=self.criterion,p_value_enter=self.p_value_enter,normalize=self.normalize,show_step=self.show_step,max_iter=self.max_iter) 
                    
        self.model_info=self.logit_model.summary()
        self.vif_info=self.vif(self.logit_model,X,show_high_vif_only=self.show_high_vif_only)
        
        return self
    
    def stepwise(self,df,response,intercept=True, normalize=False, criterion='aic', 
                      p_value_enter=.05, show_step=True,max_iter=200):
            '''
            逐步回归
            Parameters:
            --
                X:特征数据,pd.DataFrame
                y:目标变量列,pd.Series,必须与X索引一致
                df : dataframe
                    分析用数据框，response为第一列。
                response : str
                    回归分析相应变量。
                intercept : bool, 默认是True
                    模型是否有截距项。
                criterion : str, 默认是'aic',可选bic
                    逐步回归优化规则。
                p_value_enter : float, 默认是.05
                    移除变量的pvalue阈值。
                direction : str, 默认是'both'
                    逐步回归方向。
                show_step : bool, 默认是True
                    是否显示逐步回归过程。
                max_iter : int, 默认是200
                    逐步法最大迭代次数。
            '''
            criterion_list = ['bic', 'aic', 'bic_llf']
            if criterion not in criterion_list:
                raise ValueError('criterion must in', '\n', criterion_list)

            # 默认p_enter参数    
            p_enter = {'bic':0.0, 'aic':0.0}

            if normalize: # 如果需要标准化数据
                intercept = False  # 截距强制设置为0
                df_std = StandardScaler().fit_transform(df)
                df = pd.DataFrame(df_std, columns=df.columns, index=df.index)  

            remaining = list(df.columns)  # 自变量集合
            remaining.remove(response)
            selected = []  # 初始化选入模型的变量列表
            # 初始化当前评分,最优新评分
            if intercept: # 是否有截距
                formula = "{} ~ {} + 1".format(response, remaining[0])
            else:
                formula = "{} ~ {} - 1".format(response, remaining[0])

            result = smf.glm(formula, data=df,family=sm.families.Binomial(),freq_weights=self.sample_weight).fit(disp=0) # logit回归           
            current_score = eval('result.' + criterion)
            best_new_score = eval('result.' + criterion)

            if show_step:    
                print('\nstepwise starting:\n')
                
            # 当变量未剔除完，并且当前评分更新时进行循环
            iter_times = 0
            while remaining and (current_score == best_new_score) and (iter_times<max_iter):
                scores_with_candidates = []  # 初始化变量以及其评分列表
                for candidate in remaining:  # 在未剔除的变量中每次选择一个变量进入模型，如此循环
                    if intercept: # 是否有截距
                        formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
                    else:
                        formula = "{} ~ {} - 1".format(response, ' + '.join(selected + [candidate]))

                    result = smf.glm(formula, data=df,family=sm.families.Binomial(),freq_weights=self.sample_weight).fit(disp=0) # logit回归
                    llf = result.llf
                         
                    score = eval('result.' + criterion)                    
                    scores_with_candidates.append((score, candidate, llf)) # 记录此次循环的变量、评分列表

                if criterion in ['bic', 'aic']:  # 这几个指标取最小值进行优化
                    scores_with_candidates.sort(reverse=True)  # 对评分列表进行降序排序
                    best_new_score, best_candidate, best_new_llf = scores_with_candidates.pop()  # 提取最小分数及其对应变量
                    if (current_score - best_new_score) > p_enter[criterion]:  # 如果当前评分大于最新评分
                        remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优分对应的变量
                        selected.append(best_candidate)  # 将最新最优分对应的变量放入已选变量列表
                        current_score = best_new_score  # 更新当前评分
                        if show_step:  # 是否显示逐步回归过程
                            print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    elif (current_score - best_new_score) >= 0 and iter_times == 0: # 当评分差大于等于0，且为第一次迭代
                        remaining.remove(best_candidate)
                        selected.append(best_candidate)
                        current_score = best_new_score
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    elif iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
                        selected.append(remaining[0])
                        remaining.remove(remaining[0])
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))

                if intercept: # 是否有截距
                    formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
                else:
                    formula = "{} ~ {} - 1".format(response, ' + '.join(selected))                    

                result = smf.glm(formula, data=df,family=sm.families.Binomial(),freq_weights=self.sample_weight).fit(disp=0)  # 最优模型拟合                    
                if iter_times >= 1: # 当第二次循环时判断变量的pvalue是否达标
                
                    if result.pvalues.max() > p_value_enter:
                        var_removed = result.pvalues[result.pvalues == result.pvalues.max()].index[0]
                        p_value_removed = result.pvalues[result.pvalues == result.pvalues.max()].values[0]
                        selected.remove(result.pvalues[result.pvalues == result.pvalues.max()].index[0])
                        if show_step:  # 是否显示逐步回归过程                
                            print('Removing %s, Pvalue = %.3f' % (var_removed, p_value_removed))                            
                    
                iter_times += 1

            if intercept: # 是否有截距
                formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
            else:
                formula = "{} ~ {} - 1".format(response, ' + '.join(selected))

            stepwise_model = smf.glm(formula,data=df,family=sm.families.Binomial(),freq_weights=self.sample_weight).fit(disp=0)  # 最优模型拟合           
            if show_step:  # 是否显示逐步回归过程                
                print('\nLinear regression model:', '\n  ', stepwise_model.model.formula)
                print('\n', stepwise_model.summary())                

            return stepwise_model

    def vif(self,logit_model,X,show_high_vif_only=False):
        '''
        输出vif方差膨胀系数,大于10时说明存在共线性
        Parameters:
        --
            logit_model:stepwise产生的logit_model对象
            X:训练数据,pd.DataFrame
            show_high_vif_only=False:True时仅输出vif大于10的特征,False时将输出所有特征的vif
        '''
        vif = pd.DataFrame()
        variables_stepwise=logit_model.params.index.tolist()[1:]
        vif["VIF Factor"] = [variance_inflation_factor(X[variables_stepwise].values, i) for i in range(X[variables_stepwise].shape[1])]
        vif["features"] = variables_stepwise
        if show_high_vif_only:
            return(vif[vif['VIF Factor']>=10])
        else:
            return(vif)    


class cardScorer(TransformerMixin):
    
    '''
    评分转换
    Parameters:
    --
        logit_model:statsmodel/sklearn的logit回归模型对象
            + statsmodel.discrete.discrete_model.BinaryResultsWrapper类或statsmodels.genmod.generalized_linear_model类
            + sklearn.linear_model._logistic.LogisticRegression类
        varbin:BDMLtools.varReport(...).fit(...).var_report_dict,dict格式,woe编码参照此编码产生
        odds0=1/100:基准分对应的发生比(bad/good)
        pdo=50:int,评分翻番时间隔
        points0=600,int,基准分
        digit=0,评分卡打分保留的小数位数
        check_na,bool,为True时,若经打分后编码数据出现了缺失值，程序将报错终止   
                出现此类错误时多半是某箱样本量为1，或test或oot数据相应列的取值超出了train的范围，且该列是字符列的可能性极高
        special_values,缺失值指代值
            请特别注意:special_values必须与产生varbin的函数的special_values一致，否则缺失值的score将出现错误结果
            + None,保证数据默认
            + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
            + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换，字符被替换为'missing',数值被替换为np.nan
        n_jobs=1,并行数量,默认1(所有core),在数据量非常大，列非常多的情况下可提升效率但会增加内存占用，若数据量较少可设定为1    
        verbose=0,并行信息输出等级  
            
    Attribute:    
    --
        scorecard:dict,产生的评分卡,须先使用方法fit
        
    ''' 
    
    def __init__(self,logit_model,varbin,odds0=1/100,pdo=50,points0=600,digit=0,special_values=None,
                 check_na=True,n_jobs=1,verbose=0):
       
        self.logit_model=logit_model
        self.varbin=varbin
        self.odds0=odds0
        self.pdo=pdo
        self.points0=points0
        self.digit=digit
        self.special_values=special_values
        self.check_na=check_na
        self.n_jobs=n_jobs
        self.verbose=verbose

    def fit(self,X,y=None):        
                
        if isinstance(self.logit_model,(BinaryResultsWrapper,GLMResultsWrapper)):
            
            logit_model_coef=self.logit_model.params[1:].to_dict()
            logit_model_intercept=self.logit_model.params[0]
            self.columns=list(logit_model_coef.keys())
        
        elif isinstance(self.logit_model,LogisticRegression):  
            
            logit_model_coef=dict(zip(self.logit_model.feature_names_in_.tolist(),self.logit_model.coef_.tolist()[0]))
            logit_model_intercept=self.logit_model.intercept_[0]
            self.columns=self.logit_model.feature_names_in_.tolist()
            
        else:
            raise ValueError('type(logit_model) in (statsmodels..BinaryResultsWrapper;GLMResultsWrapper,sklearn.linear_model._logistic.LogisticRegression)')
            
        self.scorecard=self.getPoints(self.varbin,logit_model_coef,logit_model_intercept,self.digit)
        
        return self
    
    
    def transform(self,X,y=None):
        
        X=sp_replace(X,self.special_values)      

        p=Parallel(n_jobs=self.n_jobs,verbose=self.verbose)
            
        res=p(delayed(self.points_map)(X[key],self.scorecard[key],self.check_na) 
                              for key in self.columns)
            
        score=pd.concat({col:col_points for col,col_points in res},axis=1)
            
        score['score']=score.sum(axis=1).add(self.scorecard['intercept']['points'][0])
            
        return score  

    
    def getPoints(self,varbin,logit_model_coef,logit_model_intercept,digit):
        
        A,B=self.getAB(base=self.points0, ratio=self.odds0, PDO=self.pdo)
        
        bin_keep={col:varbin[col] for col in logit_model_coef.keys()}
        
        points_intercept=round(A-B*(logit_model_intercept),digit)

        points_all={}
        points_all['intercept']=pd.DataFrame({'variable':'intercept',
                                              'points':np.array(points_intercept)},index=['intercept'])
        
        for col in bin_keep:
            
            bin_points=bin_keep[col].join(
                        bin_keep[col]['woe'].mul(logit_model_coef[col]).mul(B).mul(-1).round(digit).rename('points') 
                    )[['variable','points','woe','breaks']]
            
            points_all[col]=bin_points
            
        return points_all
    

    def getAB(self,base=600, ratio=1/100, PDO=50):        
            
        b = PDO/np.log(2)
        a = base + b*np.log(ratio) 
        
        return a,b
    
    
    def points_map(self,col,bin_df,check_na=True):
    
        if is_numeric_dtype(col):
            
            bin_df_drop= bin_df[~bin_df['breaks'].isin([-np.inf,'missing',np.inf])]
            
            breaks=bin_df_drop['breaks'].astype('float64').tolist()
            
            points=bin_df[~bin_df['breaks'].eq('missing')]['points'].tolist()
            
            points_nan= bin_df[bin_df['breaks'].eq("missing")]['points'][0]
    
            col_points=pd.cut(col,[-np.inf]+breaks+[np.inf],labels=points,right=False,ordered=False).astype('float32')
            
            col_points=col_points.fillna(points_nan)
            
        elif is_string_dtype(col):
            
            breaks=bin_df.index.tolist();points=bin_df['points'].tolist()
           
            raw_to_breaks=raw_to_bin_sc(col.unique().tolist(),breaks)
            
            breaks_to_points=dict(zip(breaks,points))
            
            col_points=col.map(raw_to_breaks).map(breaks_to_points).astype('float32')
            
        else:
            
            raise ValueError(col.name+"‘s dtype not in ('number' or 'str')")
            
        if check_na:
            
            if col_points.isnull().sum()>0:
                
                raise ValueError(col.name+"_points contains nans")
            
        return col.name,col_points
 
    
    
    