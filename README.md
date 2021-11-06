# BDMtools-0.0.7

BDMtools是适用于常见商业数据分析数据挖掘场景下，中小数据量(建议n_sample<10w,n_cols<5000)的二分类模型的建模工具包。
本模组将集成商业分析场景中二分类模型中常用的模型，并使之能够兼顾开发效率、报告制作与模型自动化。

+ 涵盖数据清洗、数据探索、特征工程、评分卡、模型评估、统计学逐步回归、机器学习模型及其参数优化等内容
+ 在准确性和运行效率的前提下集成了诸多流行的三方库，包括sklearn,xgboost,lightgbm,statsmodels,scorecardpy,toad等,
+ 部分功能都基于sklearn.base的TransformerMixin与BaseEstimator进行构建，可支持pipeline
+ 对模块中的部分函数进行了并行优化
+ 工具包目前还在开发中

功能介绍:

+ 数据清洗与数据预处理

    - BDMtools.dtpypeAllocator:特征规范化
    - BDMtools.EDAReport:数据质量报告
    - BDMtools.businessReport:业务分析报告
    - BDMtools.nanTransformer:缺失值处理
    - BDMtools.outliersTransformer:异常值处理

+ 特征工程
    - BDMtools.preSelector:特征预筛
    - BDMtools.finbinSelector:细分箱筛选器
    - BDMtools.optbinSelector:最优分箱筛选器
    - BDMtools.corrSelector:相关性分析筛选器
    - BDMtools.faSelector:变量聚类筛选器
    - BDMtools.RFECVSelector:递归式特征消除筛选器
    - BDMtools.stepLogit:逐步回归筛选器
    - BDMtools.lassoSelector:lasso筛选器
    - BDMtools.sequentialSelector:sequential筛选器
    - BDMtools.woeTransformer:WOE编码
    
+ 报告
    - BDMtools.varReport:全量及分月的特征分析报告
    - BDMtools.varGroupsReport:分群特征分析报告

+ 评分卡
    - BDMtools.stepLogit:统计逐步回归
    - BDMtools.cardScorer:制作评分卡

+ 机器学习
    - BDMtools.girdTuner:网格优化与随机优化的xgb或lgbm
    - BDMtools.BayesianXGBTuner:贝叶斯优化的xgboost
    - BDMtools.BayesianLgbmTuner:贝叶斯优化的lightgbm

安装: 

```
pip install git+git://github.com/zk403/mlearn.git
```

卸载: 

```
pip uninstall binarymodels
```

示例:

```
待补充
```





