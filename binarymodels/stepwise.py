#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:19:07 2020

@author: kezeng
"""
from sklearn.base import TransformerMixin,BaseEstimator
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import scorecardpy as sc


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

class Stepwise(BaseEstimator):
    
    def __init__(self,custom_column=None,target='target',p_value_enter=.05,criterion='aic',normalize=False,show_step=False,show_high_vif_only=False):
        '''
        逐步回归
        Parameters:
        --
            custom_column=None:list,自定义列名,调整回归模型时使用,默认为None表示所有特征都会进行筛选
            target='target':str,目标变量列列名
            p_value_enter=.05:逐步法中特征进入的pvalue限制,默认0.05
            criterion='aic':逐步法筛选变量的准则,默认aic,可选bic
            normalize=False:是否进行数据标准化,默认False,若为True,则数据将先进行标准化,且不会拟合截距
            show_step=False:是否打印逐步回归过程
            show_high_vif_only=False:True时仅输出vif大于10的特征,False时将输出所有特征的vif
        
        Attribute:    
        --
            logit_model:逐步回归的statsmodel结果对象,须先使用方法fit
            model_info: 回归结果报告,须先使用方法fit
            vif_info:pd.DataFrame,筛选后特征的方差膨胀系数,须先使用方法fit
        '''        
        self.custom_column=custom_column
        self.target=target
        self.p_value_enter=p_value_enter
        self.criterion=criterion
        self.normalize=normalize
        self.show_step=show_step
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
            
            self.logit_model=self.stepwise(X[self.custom_column].join(y),self.target,criterion=self.criterion,p_value_enter=self.p_value_enter,normalize=self.normalize,show_step=self.show_step) 
            
        else:
            
            self.logit_model=self.stepwise(X.join(y),self.target,criterion=self.criterion,p_value_enter=self.p_value_enter,normalize=self.normalize,show_step=self.show_step) 
            
        self.model_info=self.logit_model.summary()
        self.vif_info=self.vif(self.logit_model,X,show_high_vif_only=self.show_high_vif_only)
        
        return self
    
    def stepwise(self,df,response,intercept=True, normalize=False, criterion='aic', 
                     p_value_enter=.05, direction='both', show_step=True, 
                     criterion_enter=None,max_iter=200, **kw):
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
                    当选择derection=’both‘时，移除变量的pvalue阈值。
                direction : str, 默认是'both'
                    逐步回归方向。
                show_step : bool, 默认是True
                    是否显示逐步回归过程。
                criterion_enter : float, 默认是None
                    当选择derection=’both‘时，模型加入变量的相应的criterion阈值。
                max_iter : int, 默认是200
                    逐步法最大迭代次数。
            '''
            criterion_list = ['bic', 'aic']
            if criterion not in criterion_list:
                raise IOError('请输入正确的criterion, 必须是以下内容之一：', '\n', criterion_list)

            direction_list = ['both']
            if direction not in direction_list:
                raise IOError('请输入正确的direction, 必须是以下内容之一：', '\n', direction_list)

            # 默认p_enter参数    
            p_enter = {'bic':0.0, 'aic':0.0}
            if criterion_enter:  # 如果函数中对p_remove相应key传参，则变更该参数
                p_enter[criterion] = criterion_enter

            if normalize: # 如果需要标准化数据
                intercept = False  # 截距强制设置为0
                df_std = StandardScaler().fit_transform(df)
                df = pd.DataFrame(df_std, columns=df.columns, index=df.index)  

            ''' both '''
            if direction == 'both':
                remaining = list(df.columns)  # 自变量集合
                remaining.remove(response)
                selected = []  # 初始化选入模型的变量列表
                # 初始化当前评分,最优新评分
                if intercept: # 是否有截距
                    formula = "{} ~ {} + 1".format(response, remaining[0])
                else:
                    formula = "{} ~ {} - 1".format(response, remaining[0])

                result = smf.logit(formula, df).fit(disp=0) # logit回归           
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

                        result = smf.logit(formula, df).fit(disp=0) # logit回归
                        llf = result.llf
                        llr_pvalue = result.llr_pvalue               
                        score = eval('result.' + criterion)                    
                        scores_with_candidates.append((score, candidate, llf, llr_pvalue)) # 记录此次循环的变量、评分列表

                    if criterion in ['bic', 'aic']:  # 这几个指标取最小值进行优化
                        scores_with_candidates.sort(reverse=True)  # 对评分列表进行降序排序
                        best_new_score, best_candidate, best_new_llf, best_new_llr_pvalues = scores_with_candidates.pop()  # 提取最小分数及其对应变量
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

                    result = smf.logit(formula, df).fit(disp=0)  # 最优模型拟合                    
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

                stepwise_model = smf.logit(formula, df).fit(disp=0)  # 最优模型拟合           
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

class getCreditScore(TransformerMixin):
    
    def __init__(self,logit_model,varbin,odds0=1/100,pdo=50,points0=600):
        '''
        评分转换
        Parameters:
        --
            logit_model:stepwise后的statsmodel的logit回归模型对象
            varbin:sc.woebin产生的分箱信息,dict格式,woe编码参照此编码产生
            odds0=1/100:基准分对应的发生比(bad/good)
            pdo=20:int,评分翻番时间隔
            points0=600,int,基准分
        
        Attribute:    
        --
            scorecard:dict,产生的评分卡,须先使用方法fit
        '''        
        self.logit_model=logit_model
        self.varbin=varbin
        self.odds0=odds0
        self.pdo=pdo
        self.points0=points0
        
    def fit(self):        
        self.logit_model.coef_=self.logit_model.params[1:]
        self.logit_model.intercept_=[self.logit_model.params[0]]
        self.scorecard=sc.scorecard(self.varbin,
                           self.logit_model,
                           self.logit_model.params[1:].index.tolist(),
                           odds0=self.odds0,
                           points0=self.points0,
                           pdo=self.pdo)
        return self
    
    def transform(self,X):
        score=sc.scorecard_ply(X,self.scorecard,only_total_score=False)
        return score