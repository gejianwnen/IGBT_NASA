# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:06:43 2020
@author: admin
specification
import语句会首先检查something是不是__init__.py的变量，
            然后检查是不是subpackage，
            再检查是不是module，
            最后抛出ImportError

"""


__all__ = ["myteststationary", "pdplot"]
