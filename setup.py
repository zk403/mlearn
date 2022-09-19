# -*- coding: utf-8 -*-

"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

https://python-packaging.readthedocs.io/
https://packaging.python.org/tutorials/distributing-packages/
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import re

base_pkgs=[       'numpy>=1.18,<=1.22',#https://github.com/numpy/numpy
                  'matplotlib>=3.5.0,<=3.5.3',
                  #'lofo-importance>=0.3.1',#https://github.com/aerdem4/lofo-importance
                  'fastparquet>=0.7.1',#https://github.com/dask/fastparquet
                  'pandas>=1.3.5',#https://github.com/pandas-dev/pandas
                  #'statsmodels>=0.13.2',#https://github.com/statsmodels/statsmodels
                  'plotnine>=0.9.0',#https://github.com/has2k1/plotnine
                  'scikit-learn>=1.0',#https://github.com/scikit-learn/scikit-learn
                  'xgboost>=1.4.2',#https://github.com/dmlc/xgboost
                  'catboost>=1.0.4',#https://github.com/catboost/catboost
                  'category_encoders>=2.3.0',#https://github.com/scikit-learn-contrib/category_encoders
                  'lightgbm>=3.3.0',#https://github.com/microsoft/LightGBM 
                  'probatus>=1.8.9',#https://github.com/ing-bank/probatus
                  'mlxtend>=0.19.0',#https://github.com/rasbt/mlxtend
                  'scikit-optimize>=0.9.0',#https://github.com/scikit-optimize/scikit-optimize
                  'openpyxl'
                 ]

dev_dep = [
    "pytest>=6.0.0",
    "pytest-cov>=2.10.0",
    "IPython",
    "mock"
]

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the version from __init__
with open(path.join(here, 'BDMLtools/__init__.py'), encoding='utf-8') as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

setup(
    name='BDMLtools',  # Required
    version=__version__,  # Required
    description='Ml learning tools for busniess data mining',  # Required
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    #url='http://github.com/zengke403/binarymodels',  # Optional
    author='曾珂',  # Optional
    author_email='zengke403@163.com',  # Optional
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        # 'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    #keywords='credit scorecard',  # Optional
    packages=find_packages(),  # Required
    install_requires=base_pkgs,  # Optional
    extras_require={
        "all": base_pkgs + dev_dep,
    },
    #package_data={'scorecardpy': ['data/*.csv']},
    # data_files=[('scorecardpy': ['data/*.csv'])],  # Optional
)
