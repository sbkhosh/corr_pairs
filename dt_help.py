#!/usr/bin/python3

import bs4 as bs
import csv
import inspect
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import requests
import time
import yaml

from functools import wraps
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
from statsmodels.tsa.stattools import coint

class Helper():
    def __init__(self, input_directory, input_prm_file):
        self.input_directory = input_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, input parameter file  = {}'.format(self.input_directory, self.input_prm_file))

    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
        self.rr_symb1 = self.conf.get('rr_symb1')
        self.rr_symb2 = self.conf.get('rr_symb2')
            
    @staticmethod
    def timing(f):
        """Decorator for timing functions
        Usage:
        @timing
        def function(a):
        pass
        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            end = time.time()
            print('function:%r took: %2.2f sec' % (f.__name__,  end - start))
            return(result)
        return wrapper

    @staticmethod
    def get_delim(filename):
        with open(filename, 'r') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
        return(dialect.delimiter)

    @staticmethod
    def get_class_membrs(clss):
        res = inspect.getmembers(clss, lambda a:not(inspect.isroutine(a)))
        return(res)

    @staticmethod
    def check_missing_data(data):
        print(data.isnull().sum().sort_values(ascending=False))

    @staticmethod
    def missing_values_table(dff):
        res = round((dff.isnull().sum() * 100/ len(dff)),2).sort_values(ascending=False)

        if(isinstance(self.dt_select.columns,pd.MultiIndex) == False):
            df_res = pd.DataFrame()
            df_res['Missing Value Tickers'] = res.index
            df_res['Total number of samples'] = len(dff)
            df_res['Percentage of missing values'] = res.values
            df_res['Number of missing values'] = len(dff) * df_res['Percentage of missing values'] // 100.0
            df_res = df_res[df_res['Percentage of missing values'] > 0.0]
        else:
            pass
        return(df_res)

    @staticmethod
    def view_data(data):
        print(data.head())

    @staticmethod
    def get_user_agents():
        software_names = [SoftwareName.CHROME.value]
        operating_systems = [OperatingSystem.LINUX.value]   

        user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)
        user_agents = user_agent_rotator.get_user_agents()

        user_agent = user_agent_rotator.get_random_user_agent()
        headers = {'userAgent': 'python 3.7.5', 'platform': user_agent}
        return(headers)
        
    @staticmethod
    def nasdaq100_tickers(url):
        df = pd.read_html(url)[3]
        df.rename(columns={'Ticker':'tickers','Company':'names'},inplace=True)
        df.set_index('tickers',inplace=True)
        return(df)
    
    @staticmethod
    def sp500_tickers(url):
        df = pd.read_html(url)[0]
        df = df[['Symbol','Security']]
        df.rename(columns={'Symbol':'tickers','Security':'names'},inplace=True)
        df.set_index('tickers',inplace=True)
        return(df)

    @staticmethod
    def plot_dataframe(df):
        df.plot(subplots=True,figsize=(32,20))
        plt.show()
