# BDMLtools

[![PyPI version](https://img.shields.io/pypi/pyversions/BDMLtools.svg)](https://pypi.python.org/pypi/BDMLtools)
[![License](https://img.shields.io/github/license/zk403/mlearn)](https://github.com/zk403/mlearn/blob/main/LICENSE)
[![Build Status](https://github.com/zk403/mlearn/actions/workflows/python-test.yml/badge.svg)](https://github.com/zk403/mlearn/actions/workflows/python-test.yml)
[![codecov](https://codecov.io/gh/zk403/mlearn/main/graphs/badge.svg)](https://app.codecov.io/gh/zk403/mlearn)
[![PyPI release](https://img.shields.io/pypi/v/BDMLtools.svg)](https://pypi.python.org/pypi/BDMLtools)

BDMLtools是适用于常见商业数据分析数据挖掘场景下，中小数据量的二分类模型的机器学习建模工具包。
本模组将集成商业分析场景中二分类模型中常用的机器学习模型，并使之能够兼顾模型开发效率、报告制作与建模流程标准化。
本模组涵盖数据清洗、数据探索、特征工程、评分卡制作、模型评估、统计学逐步回归、机器学习模型及其参数优化等内容

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

更新
```
v0.4.5
1.移除tunner中有关模型校准(calibration)内容
2.修复varGroupsPlot的bug
3.修改binSelector中参数iv_limit默认值
4.全面替换贝叶斯优化基础库由scikit-optimize为optuna
5.LgbmShapRFECVSelector中替换贝叶斯优化的scikit-optimize为随机网格搜索
6.移除tunner中的评估函数ks与lift，auc与logloss修改为sklearn.metrics默认
7.重构gridsearch、FLgridTuner、halvingcv代码，使其更加合理
8.更新单元测试脚本，更新部分代码说明
9.更新example
```

