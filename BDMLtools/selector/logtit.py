#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:19:07 2020

@author: kezeng
"""

from sklearn.base import TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import SET_USE_BIC_LLF
import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from sklearn.linear_model._logistic import LogisticRegression
from pandas.api.types import is_numeric_dtype,is_string_dtype
from BDMLtools.fun import raw_to_bin_sc,Specials
from joblib import Parallel,delayed,effective_n_jobs
from BDMLtools.base import Base
import warnings


class stepLogit(Base,TransformerMixin):
    
    '''
    逐步回归,仅支持数值变量、有截距的逐步法logistic,列名应能够被pasty识别
    
    逐步回归过程:
    向前法(forward):      
        首先尝试加入:
            +从潜在特征中尝试所有特征并选择出使指标(aic,bic)优化的特征进入
        循环上述步骤直到
            +无潜在特征可用
            +无潜在特征可使aic或bic优化 
            +迭代次数达到最大            
    向后法(backward):       
        首先尝试剔除:
            +从潜在特征中尝试移除某个特征并选择出使指标(aic,bic)优化的特征进入

        循环上述步骤直到
            +无潜在特征可用
            +无潜在特征可使aic或bic优化 
            +迭代次数达到最大        
    向前向后法(both):
        首先尝试加入:
            +从潜在特征中尝试所有特征并选择出使指标(aic,bic)优化的特征进入
        再进行剔除:
            +剔除模型中p值最高的特征(大于p_value_enter)
        循环上述步骤直到
            +无潜在特征可用
            +无潜在特征可使aic或bic优化 
            +迭代次数达到最大
    
    Parameters:
    --
        custom_column=None:list,自定义列名,调整回归模型时使用,默认为None表示所有特征都会进行筛选
        method='no_stepwise',str
            no_stepwise:不进行逐步回归直接以custom_column或全部变量进行回归
            forward:向前法逐步回归
            backward:向后法逐步回归
            both:向前向后法
        p_value_enter=.05:向前向后法中特征进入的pvalue限制,默认0.05
        criterion='aic':逐步法筛选变量的准则,默认aic,可选bic
        show_step=False:是否打印逐步回归过程
        max_iter=200,逐步回归最大迭代次数
        show_high_vif_only=False:True时仅输出vif大于10的特征,False时将输出所有特征的vif
    
    Attribute:    
    --
        logit_model:逐步回归的statsmodel结果对象,须先使用方法fit
        model_info: 回归结果报告,须先使用方法fit
        vif_info:pd.DataFrame,筛选后特征的方差膨胀系数,须先使用方法fit
        coefficients_sd: 标准化回归系数(b_i_std=b_i*sd_i),须先使用方法fit
    '''        
    
    def __init__(self,custom_column=None,
                 method='no_stepwise',
                 p_value_enter=.05,
                 criterion='aic',
                 show_step=False,
                 max_iter=200,
                 show_high_vif_only=False):
      
        self.custom_column=custom_column
        self.method=method
        self.p_value_enter=p_value_enter
        self.criterion=criterion
        self.show_step=show_step
        self.max_iter=max_iter
        self.show_high_vif_only=show_high_vif_only
        
        self._is_fitted=False
        
    def predict_proba(self,X,y=None):
        '''
        模型预测,使用逐步回归模型预测,产生预测概率
        Parameters:
        --
        X:woe编码数据,pd.DataFrame对象,需与训练数据woe编码具有相同的特征
        '''      
        self._check_is_fitted()
        self._check_X(X)
        
        pred=self.logit_model.predict(X)
        
        return pd.Series(pred,index=X.index,name='pred')
    
    def transform(self,X,y=None):     
        '''
        使用逐步回归进行特征筛选,返回逐步法筛选后的训练数据
        Parameters:
        --
        X:woe编码数据,pd.DataFrame对象,需与训练数据woe编码具有相同的特征
        '''        
        
        self._check_is_fitted()
        self._check_X(X)
        
        return X[self.logit_model.params.index.tolist()[1:]]
    
    def _check_colnum(self,X):
        
        if self.custom_column:
            
            if len(set(self.custom_column))<3:
                
                warnings.warn('at least 3 features required in stepwise,method will be reset to “no_stepwise”')
                
                self.method='no_stepwise'
                
        else:
            
            if X.columns.size<3:
                
                warnings.warn('at least 3 features required in stepwise,method will be reset to “no_stepwise”')
                
                self.method='no_stepwise'

          
    def fit(self,X,y,sample_weight=None):
        '''
        拟合逐步回归
        Parameters:
        --
        X:woe编码训练数据,pd.DataFrame对象
        y:目标变量,pd.Series对象
        '''        
        self._check_data(X, y)     
        self._check_colnum(X)
        
        if self.custom_column:
            
            if self.method=='no_stepwise':
                
                formula = "{} ~ {} + 1".format(y.name,' + '.join(self.custom_column))
                
                self.logit_model=smf.glm(formula, data=X[self.custom_column].join(y),
                                         family=sm.families.Binomial(),
                                         freq_weights=sample_weight).fit(disp=0)
            
            elif self.method=='forward':
            
                self.logit_model=self._stepwise_forward(X[self.custom_column].join(y),y.name,criterion=self.criterion,show_step=self.show_step,max_iter=self.max_iter,sample_weight=sample_weight) 
            
            elif self.method=='backward':
            
                self.logit_model=self._stepwise_backward(X[self.custom_column].join(y),y.name,criterion=self.criterion,show_step=self.show_step,max_iter=self.max_iter,sample_weight=sample_weight)               
                
            elif self.method=='both':
            
                self.logit_model=self._stepwise_both(X[self.custom_column].join(y),y.name,criterion=self.criterion,p_value_enter=self.p_value_enter,show_step=self.show_step,max_iter=self.max_iter,sample_weight=sample_weight) 
                
            else:
                
                raise ValueError("method in ('no_stepwise','forward','backward','both')")
            
        else:
            
            if self.method=='no_stepwise':
                
                formula = "{} ~ {} + 1".format(y.name,' + '.join(X.columns.tolist()))
                
                self.logit_model=smf.glm(formula, data=X.join(y),
                                         family=sm.families.Binomial(),
                                         freq_weights=sample_weight).fit(disp=0)                                                

            elif self.method=='forward':
            
                self.logit_model=self._stepwise_forward(X.join(y),y.name,criterion=self.criterion,show_step=self.show_step,max_iter=self.max_iter,sample_weight=sample_weight) 
            
            elif self.method=='backward':
            
                self.logit_model=self._stepwise_backward(X.join(y),y.name,criterion=self.criterion,show_step=self.show_step,max_iter=self.max_iter,sample_weight=sample_weight) 
            
            elif self.method=='both':
                
                self.logit_model=self._stepwise_both(X.join(y),y.name,criterion=self.criterion,p_value_enter=self.p_value_enter,show_step=self.show_step,max_iter=self.max_iter,sample_weight=sample_weight) 
                                 
            else:
                
                raise ValueError("method in ('no_stepwise','forward','backward','both')")                     
                    
        self.model_info=self.logit_model.summary()
        self.vif_info=self._vif(self.logit_model,X,show_high_vif_only=self.show_high_vif_only)
        self.coefficients_sd=self._sdcoeff(self.logit_model,X)
        
        self._is_fitted=True
        
        return self
    
    def _stepwise_forward(self,df,response,criterion='aic',show_step=True,max_iter=200,sample_weight=None):
        
        SET_USE_BIC_LLF(True)

        criterion_list = ['bic', 'aic']
        remaining = df.columns.tolist()
        remaining.remove(response)
        selected = []
        iter_times=0
        
        if criterion not in criterion_list:
        
            raise ValueError('criterion must in', '\n', criterion_list)
        
        current_score = np.inf
        best_new_score = np.inf
        
        if show_step:       
            
            print('\nstepwise-foward starting:\n')
                
        while remaining and current_score==best_new_score and (iter_times<max_iter):
            
            scores_with_candidates = []
        
            for candidate in remaining:
                
                formula = "{} ~ {} + 1".format(response,' + '.join(selected + [candidate]))
                result = smf.glm(formula, data=df,family=sm.families.Binomial(),freq_weights=sample_weight).fit(disp=0)
                score = eval('result.'+criterion)
                scores_with_candidates.append((score, candidate))
                
            scores_with_candidates.sort(reverse=True) #the lower aic/bic the better
            best_new_score, best_candidate = scores_with_candidates.pop()
            
            if current_score > best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
                
                if show_step:    
                    
                    print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    
            iter_times+=1
        
                
        formula = "{} ~ {} + 1".format(response,' + '.join(selected))
        stepwise_model = smf.glm(formula, data=df,family=sm.families.Binomial(),freq_weights=sample_weight).fit(disp=0)
        
        if show_step:         
            
            print('\nLinear regression model:', '\n  ', stepwise_model.model.formula)
            print('\n', stepwise_model.summary()) 
            
        return stepwise_model
    
    
    def _stepwise_backward(self,df,response,criterion='aic',show_step=True,max_iter=200,sample_weight=None):
    
        SET_USE_BIC_LLF(True)
    
        criterion_list = ['bic', 'aic']
        remaining = df.columns.tolist()
        remaining.remove(response)
        selected = []
        iter_times=0
    
        if criterion not in criterion_list:
    
            raise ValueError('criterion must in', '\n', criterion_list)
    
        current_score = np.inf
        best_new_score = np.inf
        
        if show_step: 
            
            print('\nstepwise-backward starting:\n')
        
        
        while remaining and current_score==best_new_score and (iter_times<max_iter):
            
            scores_with_candidates = []
    
            for candidate in remaining:
    
                selected=remaining.copy()
                selected.remove(candidate)
                formula = "{} ~ {} + 1".format(response,' + '.join(selected))
                result = smf.glm(formula, data=df,family=sm.families.Binomial(),freq_weights=sample_weight).fit(disp=0)
                score = eval('result.'+criterion)
                scores_with_candidates.append((score, candidate))
                
            scores_with_candidates.sort(reverse=True) #the lower aic/bic the better
            best_new_score, best_candidate = scores_with_candidates.pop()
            
            if current_score > best_new_score:
                remaining.remove(best_candidate)
                selected=remaining.copy()
                current_score = best_new_score
                
                if show_step:    
                    
                    print('Removing %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    
            iter_times+=1
    
        
        formula = "{} ~ {} + 1".format(response,' + '.join(selected))
        stepwise_model = smf.glm(formula, data=df,family=sm.families.Binomial(),freq_weights=sample_weight).fit(disp=0)
        
        if show_step:         
            
            print('\nLinear regression model:', '\n  ', stepwise_model.model.formula)
            print('\n', stepwise_model.summary()) 
            
        return stepwise_model
        
    
    def _stepwise_both(self,df,response,criterion='aic',p_value_enter=.05,show_step=True,max_iter=200,sample_weight=None):

        SET_USE_BIC_LLF(True)
        
        criterion_list = ['bic', 'aic']
        
        if criterion not in criterion_list:
            
            raise ValueError('criterion must in', '\n', criterion_list)



        remaining = list(df.columns)  # variables set
        remaining.remove(response)
        selected = []  # selected variables set
        # initializing
            
        current_score = np.inf
        best_new_score = np.inf

        if show_step:    
            print('\nstepwise-both starting:\n')
            
        # loop when current_score keeps updating 
        iter_times = 0
        
        while remaining and (current_score == best_new_score) and (iter_times<max_iter):
            
            scores_with_candidates = []  
            
            for candidate in remaining:  
                
                formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
                result = smf.glm(formula, data=df,family=sm.families.Binomial(),freq_weights=sample_weight).fit(disp=0) 
                llf = result.llf
                     
                score = eval('result.' + criterion)                    
                scores_with_candidates.append((score, candidate, llf))
                
            
            if criterion in ['bic', 'aic']:  
                
                #sort aic/bic decscending and pop the minimal aic/bic element(the best score)
                scores_with_candidates.sort(key=lambda x:x[0],reverse=True)                      
                best_new_score, best_candidate, best_new_llf = scores_with_candidates.pop() 
                
                #print(current_score,best_new_score)
                
                if (current_score - best_new_score) > 0:  
                
                    remaining.remove(best_candidate)  
                    
                    selected.append(best_candidate) 
                    
                    current_score = best_new_score  
                    
                    if show_step: 
                    
                        print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))

                #when no aic/bic updating at first selection then continuing the process        
                elif iter_times == 0:  
                
                    selected.append(remaining[0])
                    
                    remaining.remove(remaining[0])
                    
                    if show_step:       
                    
                        print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
            
            #print(current_score,best_new_score)
            
  
            formula = "{} ~ {} + 1".format(response, ' + '.join(selected))                
            result = smf.glm(formula, data=df,family=sm.families.Binomial(),freq_weights=sample_weight).fit(disp=0)  
               
            if iter_times >= 1: #remove variables i selected set when its pvalue too high 
            
                if result.pvalues.max() > p_value_enter:
                    
                    var_removed = result.pvalues[result.pvalues == result.pvalues.max()].index[0]
                    
                    p_value_removed = result.pvalues[result.pvalues == result.pvalues.max()].values[0]
                    
                    selected.remove(result.pvalues[result.pvalues == result.pvalues.max()].index[0])
                    
                    if show_step:          
                    
                        print('Removing %s, Pvalue = %.3f' % (var_removed, p_value_removed))                            
                
            iter_times += 1


        formula = "{} ~ {} + 1".format(response, ' + '.join(selected))

        
        #modeling with selected vars    
        stepwise_model = smf.glm(formula,data=df,family=sm.families.Binomial(),freq_weights=sample_weight).fit(disp=0)  
        
        if show_step:                 
            print('\nLinear regression model:', '\n  ', stepwise_model.model.formula)
            print('\n', stepwise_model.summary())                

        return stepwise_model
        

    def _vif(self,logit_model,X,show_high_vif_only=False):
        
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
        
        if len(variables_stepwise)<=1:
            
            return(None)
            
        else:
        
            vif["VIF Factor"] = [variance_inflation_factor(X[variables_stepwise].values, i) for i in range(X[variables_stepwise].shape[1])]
            vif["features"] = variables_stepwise
        
            if show_high_vif_only:
                
                return(vif[vif['VIF Factor']>=10])
            
            else:
                
                return(vif)
            
    def _sdcoeff(self,logit_model,X,method='Agresti'):
            
        '''
        Partial standardization coefficients
        
        参考文献:
        http://web.pdx.edu/~newsomj/mvclass/ho_logistic.pdf
        https://think-lab.github.io/d/205/
        
        标准化回归系数(Agresti):
        b_i_std=b_i*sd_i=b_i*sd_i 
        
        标准化回归系数(SAS):        
        b_i_std=b_i*sd_i/(pi/sqrt(3))=b_i*sd_i*0.5513   
        
        标准化回归系数(Long):
        b_i_std=b_i*sd_i=b_i*sd_i /(pi/sqrt(3)+1)=b_i*sd_i*0.3554
  
        '''

        params=logit_model.params[1:]
        
        if method=='Agresti':
        
            coeff_std=params*X[params.index].std()
            
        elif method=='SAS':
            
            coeff_std=params*X[params.index].std()*0.5513
            
        elif method=='Long':
            
            coeff_std=params*X[params.index].std()*0.3554
            
        else:
            
            coeff_std=params*X[params.index].std()
        
        return coeff_std



class cardScorer(Base,Specials,TransformerMixin):
    
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
        special_values,特殊值指代值,若数据中某些值或某列某些值需特殊对待(这些值不是np.nan)时设定
            请特别注意,special_values必须与binSelector的special_values一致,否则score的special行会产生错误结果
            + None,保证数据默认
            + list=[value1,value2,...],数据中所有列的值在[value1,value2,...]中都会被替换为special
            + dict={col_name1:[value1,value2,...],...},数据中指定列替换，被指定的列的值在[value1,value2,...]中都会被替换为special
        n_jobs=1,并行数量 
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
        
        self._is_fitted=False

    def fit(self,X=None,y=None):     
                
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
            
        self.scorecard=self._getPoints(self.varbin,logit_model_coef,logit_model_intercept,self.digit)
        
        self._is_fitted=True
        
        return self
    
    
    def transform(self,X,y=None):
        
        self._check_is_fitted()
        self._check_X(X)
        
        n_jobs=effective_n_jobs(self.n_jobs)

        p=Parallel(n_jobs=n_jobs,verbose=self.verbose)
            
        res=p(delayed(self._points_map)(X[key],self.scorecard[key],self.check_na,self.special_values) 
                              for key in self.columns)
            
        score=pd.concat({col:col_points for col,col_points in res},axis=1)
            
        score['score']=score.sum(axis=1).add(self.scorecard['intercept']['points'][0])
            
        return score  

    
    def _getPoints(self,varbin,logit_model_coef,logit_model_intercept,digit):
        
        A,B=self._getAB(base=self.points0, ratio=self.odds0, PDO=self.pdo)
        
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
    

    def _getAB(self,base=600, ratio=1/100, PDO=50):        
            
        b = PDO/np.log(2)
        a = base + b*np.log(ratio) 
        
        return a,b
    
    
    def _points_map(self,col,bin_df,check_na=True,special_values=None):
        
        col=self._sp_replace_single(col,self._check_spvalues(col.name,special_values),fill_num=np.finfo(np.float32).max,fill_str='special')
    
        if is_numeric_dtype(col):
            
            bin_df_drop= bin_df[~bin_df['breaks'].isin([-np.inf,'missing','special',np.inf])]
            
            breaks=bin_df_drop['breaks'].astype('float64').tolist()
            
            points=bin_df[~bin_df['breaks'].isin(['missing','special'])]['points'].tolist()
            
            points_nan= bin_df[bin_df['breaks'].eq("missing")]['points'][0]
            
            points_sp= bin_df[bin_df['breaks'].eq("special")]['points'][0]
            
            if special_values:
                
                breaks_cut=breaks+[np.finfo(np.float32).max]
                
                col_points=pd.cut(col,[-np.inf]+breaks_cut+[np.inf],labels=points+[points_sp],right=False,ordered=False).astype('float64')
                
                col_points=col_points.fillna(points_nan)                

            else:
                
                breaks_cut=breaks
    
                col_points=pd.cut(col,[-np.inf]+breaks_cut+[np.inf],labels=points,right=False,ordered=False).astype('float64')
                
                col_points=col_points.fillna(points_nan)                
            
        elif is_string_dtype(col):
            
            points_nan= bin_df[bin_df['breaks'].eq("missing")]['points'][0]
        
            points_sp= bin_df[bin_df['breaks'].eq("special")]['points'][0]
        
            breaks=bin_df[~bin_df['breaks'].isin(['missing','special'])].index.tolist()
        
            points=bin_df[~bin_df['breaks'].isin(['missing','special'])]['points'].tolist()
    
    
            if all(list(map(self._is_no_sp,[i.split('%,%') for i in breaks]))):
                
                breaks.append('special')
                points.append(points_sp)
                
            if all(list(map(self._is_no_na,[i.split('%,%') for i in breaks]))):
                
                breaks.append('missing')
                points.append(points_nan)    

           
            raw_to_breaks=raw_to_bin_sc(col.unique().tolist(),breaks)
            
            breaks_to_points=dict(zip(breaks,points))
            
            col_points=col.map(raw_to_breaks).map(breaks_to_points).astype('float64')
            
        else:
            
            raise ValueError(col.name+"‘s dtype not in ('number' or 'str')")
            
        if check_na:
            
            if col_points.isnull().sum()>0:
                
                raise ValueError(col.name+"_points contains nans")
            
        return col.name,col_points
    
    
    def _is_no_sp(self,strings):    
    
        if 'special' in strings:
            
            return False
        
        else:
            
            return True
    
    def _is_no_na(self,strings):
        
        if 'missing' in strings:
            
            return False
        
        else:
            
            return True
 
    
    
    