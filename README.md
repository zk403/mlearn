# BDMtools-0.1.2

BDMtools是适用于常见商业数据分析数据挖掘场景下，中小数据量(建议n_sample<10w,n_cols<5000)的二分类模型的建模工具包。
本模组将集成商业分析场景中二分类模型中常用的模型，并使之能够兼顾开发效率、报告制作与模型自动化。

+ 涵盖数据清洗、数据探索、特征工程、评分卡、模型评估、统计学逐步回归、机器学习模型及其参数优化等内容
+ 在准确性和运行效率的前提下集成了诸多流行的三方库，包括sklearn,xgboost,lightgbm,statsmodels等
+ 本模组的评分卡开发步骤参考了scorecardpy,并对其在底层上进行了重写
    - 功能上几乎与其一致并增加了更多的拓展
    - 可同时支持sklearn、statsmodel的logit回归模型
    - 在分箱算法、报告制作、评分转换上的运行效率上更优    
+ 部分功能都基于sklearn.base的TransformerMixin与BaseEstimator进行构建，可支持pipeline
+ 模块中对列的数据处理进行了并行优化
+ 工具包目前还在开发中

功能介绍:

+ 数据清洗与数据预处理

    - BDMtools.dtStandardization:数据规范化，解决实体重复、内存占用,索引等问题
    - BDMtools.dtypeAllocator:特征类型分配，将原始数据中的列类型转换为适合进行分析与建模的数据类型
    - BDMtools.nanTransformer:缺失值处理，缺失值填补
    - BDMtools.outliersTransformer:异常值处理
    - BDMtools.prefitModel:预拟合数据，在不进行任何特征工程的前提下使用全量特征预拟合数据以预估模型表现

+ 特征工程
    - BDMtools.preSelector:特征预筛，按照缺失值、唯一值、方差、卡方值、f值、树模型、iv等进行特征预筛
    - BDMtools.binSelector:分箱筛选器，提供等频、决策树、卡方、单调等分箱算法并依据分箱结果进行特征筛选
    - BDMtools.corrSelector:相关性分析筛选器，按照相关性筛选特征
    - BDMtools.faSelector:变量聚类筛选器，先进行变量聚类，再依据r2ratio等指标筛选特征
    - BDMtools.RFECVSelector:递归式特征消除筛选器，开发中
    - BDMtools.stepLogit:逐步回归筛选器，提供基于aic/bic与statsmodel的logit-reg的逐步法进行筛选特征与模型构建
    - BDMtools.lassoSelector:lasso筛选器，基于logit-lasso算法进行特征筛选
    - BDMtools.sequentialSelector:sequential筛选器，开发中
    - BDMtools.woeTransformer:WOE编码，对原始数据进行woe编码
    
+ 报告
    - BDMtools.EDAReport:数据质量报告
    - BDMtools.businessReport:业务分析报告   
    - BDMtools.varReportSinge:单特征分析报告 
    - BDMtools.varReport:特征分析报告
    - BDMtools.varGroupsReport:分群特征分析报告

+ 评分卡
    - BDMtools.stepLogit:统计逐步回归
    - BDMtools.cardScorer:制作评分卡

+ 机器学习-分类算法
    - BDMtools.girdTuner:网格优化与随机优化的xgb或lgbm
    - BDMtools.BayesianXGBTuner:贝叶斯优化的xgboost
    - BDMtools.BayesianLgbmTuner:贝叶斯优化的lightgbm
    - BDMtools.hgirdTuner:scucessive halving优化的xgb或lgbm

+ 机器学习-异常发现算法
    - 待补充
    
安装: 

```
pip install git+git://github.com/zk403/mlearn.git
```

卸载: 

```
pip uninstall BDMtools
```

示例:

```
待补充
```





