# -*- coding: utf-8 -*-

"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

https://python-packaging.readthedocs.io/
https://packaging.python.org/tutorials/distributing-packages/
"""

# Always prefer setuptools over distutils
from setuptools import setup#, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import re

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the version from __init__
with open(path.join(here, 'binarymodels/__init__.py'), encoding='utf-8') as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

setup(
    name='binarymodels',  # Required
    version=__version__,  # Required
    description='binarymodels for busniess usage',  # Required
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='http://github.com/zengke403/binarymodels',  # Optional
    author='zengke403',  # Optional
    author_email='',  # Optional
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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='credit scorecard, woe binning, performace evaluation',  # Optional
    packages=['binarymodels'],  # Required
    install_requires=['numpy','pandas>=0.25.0',
                      'matplotlib','scikit-learn>=0.22.0','' 
                      'statsmodels', 'patsy'],  # Optional
    #package_data={'scorecardpy': ['data/*.csv']},
    # data_files=[('scorecardpy': ['data/*.csv'])],  # Optional
    #project_urls={  # Optional
    #    'Bug Reports': 'https://github.com/shichenxie/scorecardpy/issues',
    #    'Source': 'https://github.com/shichenxie/scorecardpy/',
    #},
)
