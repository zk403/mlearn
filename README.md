# BDMLtools-0.3.7

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
v0.3.8
1.报告模块中加入auc指标，加入参数show_metrics以控制auc、ks、lift的输出
2.修复了binAdjuster中组变量加入breaks_list时导致终止的bug
3.binSelector中将不再输出ks
4.调整FLLGBMSklearn中与梯度、正则相关的树参数的默认范围，避免产生决策树无法生长的问题
5.其他bug修复
6.更新部分依赖库的版本
7.更新单元测试脚本,更新部分代码说明
8.更新example
```