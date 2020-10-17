#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd 
import warnings

from dt_help import Helper
from dt_pdr import HistData
from pandas.plotting import register_matplotlib_converters

warnings.filterwarnings('ignore',category=FutureWarning)
pd.options.mode.chained_assignment = None 
register_matplotlib_converters()

if __name__ == '__main__':
    obj_helper = Helper('data_in','conf_help.yml')
    obj_helper.read_prm()
    
    fontsize = obj_helper.conf['font_size']
    matplotlib.rcParams['axes.labelsize'] = fontsize
    matplotlib.rcParams['xtick.labelsize'] = fontsize
    matplotlib.rcParams['ytick.labelsize'] = fontsize
    matplotlib.rcParams['legend.fontsize'] = fontsize
    matplotlib.rcParams['axes.titlesize'] = fontsize
    matplotlib.rcParams['text.color'] = 'k'

    data_obj = HistData('data_in','data_out','conf_pdr.yml')
    data_obj.read_prm()
    data_obj.process()
    data_obj.visualize()
