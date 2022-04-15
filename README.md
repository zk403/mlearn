# BDMLtools-0.2.2

BDMLtools是适用于常见商业数据分析数据挖掘场景下，中小数据量的二分类模型的机器学习建模工具包。
本模组将集成商业分析场景中二分类模型中常用的机器学习模型，并使之能够兼顾模型开发效率、报告制作与建模流程标准化。

+ 涵盖数据清洗、数据探索、特征工程、评分卡制作、模型评估、统计学逐步回归、机器学习模型及其参数优化等内容
+ 在准确性和运行效率的前提下集成了诸多流行的三方库，包括sklearn,xgboost,lightgbm,statsmodels等
+ 本模块的评分卡开发部分参考了谢士晨博士的R::scorecard(https://github.com/ShichenXie/scorecard)
    - 功能上几乎与其一致并增加了更多的拓展，例如单调最优分箱算法
    - 支持sklearn、statsmodel的logit回归模型的评分卡制作
    - 在分箱算法、报告制作、评分转换上的本模组运行效率更优  
+ 部分功能都基于sklearn.base的TransformerMixin与BaseEstimator进行构建，可支持pipeline
+ 模块中对列的数据处理进行了并行优化
+ 工具包总体上还在开发中

功能介绍:

+ 数据清洗与数据预处理

    - BDMLtools.dtStandardization:数据规范化，解决实体重复、内存占用,索引等问题
    - BDMLtools.dtypeAllocator:特征类型分配，将原始数据中的列类型转换为适合进行分析与建模的数据类型
    - BDMLtools.nanTransformer:缺失值处理，缺失值填补
    - BDMLtools.outliersTransformer:异常值处理
    - BDMLtools.prefitModel:预拟合数据，在不进行任何特征工程的前提下使用全量特征预拟合数据以预估模型表现

+ 特征工程/评分卡

    - BDMLtools.preSelector:特征预筛，按照缺失值、唯一值、方差、卡方值、f值、树模型、iv等进行特征预筛
    - BDMLtools.faSelector:变量聚类筛选器
    - BDMLtools.woeTransformer:WOE编码，对原始数据进行woe编码
    - BDMLtools.binSelector:分箱筛选器，提供等频、决策树、卡方、单调等分箱算法并依据分箱结果进行特征筛选
    - BDMLtools.binAdjuster:分箱调整器,支持单特征、组特征的交互式分箱及分箱调整
    - BDMLtools.stepLogit:统计逐步回归筛选器，提供基于aic/bic与statsmodel的logit-reg的逐步法进行筛选特征与模型构建
    - BDMLtools.cardScorer:制作评分卡
    - BDMLtools.LgbmSeqSelector:Lgbm逐步式特征选择筛选器
    - BDMLtools.LgbmShapRFECVSelector:Lgbm递归式特征消除筛选器
    - BDMLtools.LgbmPISelector:Lgbm组合重要性除筛选器
    
+ 报告

    - BDMLtools.EDAReport:数据质量报告
    - BDMLtools.varReportSinge:单特征分析报告 
    - BDMLtools.varReport:特征分析报告
    - BDMLtools.varGroupsReport:分群特征分析报告


+ 机器学习-分类算法

    - BDMLtools.gridTuner:网格优化与随机优化的xgb、lgbm、catboost
    - BDMLtools.BayesianCVTuner:贝叶斯优化的xgb、lgbm、catboost
    - BDMLtools.hgridTuner:scucessive halving优化的xgb、lgbm、catboost

+ 机器学习-异常发现算法
    - 待补充
    
    
+ 模型评估 

    - BDMLtools.perfEval:二分类排序模型评估，包含绘制图ks、roc、gini、gain、lorenz、f1、pr、density的功能
    
安装: 

+ github

```
pip install git+git://github.com/zk403/mlearn.git
```

+ pypi

```
pip install BDMLtools
```

卸载: 

```
pip uninstall BDMLtools
```

示例:

```
见demo
```


更新日志:

```
v0.2.2
1.加入特征筛选方法LgbmShapRFESelector和LgbmPISelector
2.优化了preSelector代码
3.移除特征筛选方法_fliterByShuffle,LgbmRFESelector
4.移除BayesianXGBTuner,BayesianLGBMTuner,BayesianCBTuner
5.修复了网格参数优化、减半参数优化、贝叶斯优化中early_stopping的bug
6.相关文档维护
```


```
v0.2.1
1.在网格参数优化、减半参数优化、贝叶斯优化中加入对Catboost的支持
2.在网格参数优化、减半参数优化、贝叶斯优化中加入early_stopping参数
3.优化贝叶斯优化代码使之能够支持设定随机数实现模型复现
4.优化贝叶斯优化代码使之能够支持更多参数选择，并加入固定参数
5.LgbmRFESelector更名为LgbmRFECVSelector
6.dtypeAllocator中的number类型以float和int进行了进一步的区分
7.修复了outliersTransformer中的后非number列丢失问题
8.修复了减半参数优化中模型无法通过随机数复现的bug
9.相关文档维护
```

```
v0.2.0
1.加入特征筛选方法LgbmRFESelector和LgbmSeqSelector
2.删除了corrSelector
3.优化代码
4.文档维护
```


