#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
    
    
def RMSE(y_true, y_predicted):
    """Obtain RMSE between true y's and predicted y's"""
    return (np.mean((y_true-y_predicted)**2))**0.5


def ThreeD1(axa, plt):
    plt.rcParams["figure.figsize"] = (6,4) 
    plt.rcParams["figure.dpi"] = 150
    axa.tick_params(axis='y', labelsize=10) 
    axa.tick_params(axis='x', labelsize=10) 
    axa.set_xlabel(" Clay" , fontweight='bold', fontsize=12) 
    axa.set_ylabel(" F1mass" , fontweight='bold', fontsize=12) 
    axa.set_zlabel(" CEC [meq/100g]" , fontweight='bold', fontsize=12) 
    axa.set_title("  " , fontweight='bold', fontsize=16) 
    axa.set_zlim(0, 80) 