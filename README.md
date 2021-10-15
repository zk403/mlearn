# binarymodels

适用于常见商业数据分析数据挖掘场景下，中小数据量(建议n_sample<10w,n_cols<5000)的二分类模型的建模工具,将流行的二分类建模工具进行了集成,使之更加适合实际项目场景与工程化,集成的工具包括sklearn,xgboost,lightgbm,statsmodels,scorecardpy等基本涵盖了数据清洗、数据探索、特征工程、评分卡、模型评估、统计学逐步回归、机器学习模型及其参数优化等内容,具体包括:

- binarymodels.getColmuns:规范化原始数据的数据类型
- binarymodels.getReport:EDA阶段的数据质量报告
- binarymodels.outliersTransformer:异常值处理
- binarymodels.selection_pre:集成了多种方法的特征粗筛
- binarymodels.selection_iv:特征分箱与IV筛选
- binarymodels.selection_corr:使用相关性分析进行特征筛选
- binarymodels.featureCluster:类似于SAS的变量聚类
- binarymodels.getWOE:WOE编码
- binarymodels.stepwise:基于经典统计学(statsmodels)的逐步回归
- binarymodels.getCreditScore:基于回归制作评分卡
- binarymodels.searchGirdCV:网格优化与随机优化的xgb或lgbm
- binarymodels.searchBayesianXGB:贝叶斯优化的xgboost
- binarymodels.searchBayesianXGB:贝叶斯优化的lightgbm
    
所有的功能都基于sklearn.base的TransformerMixin与BaseEstimator进行构建,因此可以进行sklearn的pipline操作

安装: 

```
pip install git+https://github.com/zk403/mlearn.git
```



