#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:10:05 2022

@author: zengke
"""

class DataDtypesError(ValueError):
    """Exception class to raise if data's dtypes not in ('number' or 'object')

    """
    
class DataTypeError(ValueError):
    
    """Exception class to raise if data is not pandas Dataframe or Series

    """ 
    
class XyIndexError(IndexError):
    
    """Exception class to raise if X's index not equal to y's index

    """ 
    
class yValueError(ValueError):
    
    """Exception class to raise if y values not equal to number 0 or number 1

    """ 
    
    