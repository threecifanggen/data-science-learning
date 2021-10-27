"""线性回归相关工具
"""
from statsmodels import api
import pandas as pd

def model(X, y, model_func=api.OLS, add_const=True):
    """方便地增加常数的模型
    """
    if add_const:
        m = model_func(y, api.add_constant(X))
    else:
        m = model_func(y, X)
    return m.fit()
    

def vif_col(X, y, col_name):
    """计算vif

    计算具体一个column的vif，
    一般阈值在5或者10，超过这个数字则表明有
    共线性。

    Attributes:
        X (pd.DataFrame): 自变量
        y (pd.Series): 因变量
        col_name (str): 需要判断的列

    References:

        James, Gareth, Daniela Witten, Trevor Hastie, and Robert Tibshirani.
        An Introduction to Statistical Learning. pp. 112, Vol. 112: Springer, 2013.
    """
    r_square_minus = model(X.loc[:, X.columns != col_name].values, y).rsquared
    return 1 / (1 - r_square_minus)


def vif(X, y):
    """计算一个df里每一列的vif
    """
    return dict(
        (col_name, vif_col(X, y, col_name)) 
        for col_name in X
    )


