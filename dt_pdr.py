#!/usr/bin/python3

import itertools
import matplotlib.pyplot as plt
import os
import numpy as np
import operator
import pandas as pd
import pandas_datareader as pdr
import requests_cache
import yaml

from datetime import datetime, timedelta
from dt_help import Helper
from sqlalchemy import create_engine

class HistData():
    def __init__(self,input_directory, output_directory, input_prm_file):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.output_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, output directory = {}, input parameter file  = {}'.\
               format(self.input_directory, self.output_directory, self.input_prm_file))
        
    @Helper.timing
    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
        self.start_date = self.conf.get('start_date')
        self.end_date = self.conf.get('end_date')
        self.ohlc = self.conf.get('ohlc')
        self.urls = self.conf.get('urls')
        self.index = self.conf.get('index')
        self.new_db_tickers = self.conf.get('new_db_tickers')
        self.new_db_raw_cap = self.conf.get('new_db_raw_cap')
        self.new_db_raw_prices = self.conf.get('new_db_raw_prices')
        self.top_vals = self.conf.get('top_vals')
        
    @Helper.timing
    def process(self):
        # get the tickers for the selected index
        HistData.get_tickers(self)
        
        # get raw prices
        HistData.get_raw_prices(self)
        
        # compute returns as (P_t-P_{t-1})/P_{t-1}
        self.dt_select_ret = self.dt_select.apply(lambda x: x.pct_change().fillna(0))
        self.dt_select_ret.columns = pd.MultiIndex.from_product([[el + '_Return' for el in self.ohlc],self.symbs],names=('Attributes', 'Symbols'))
        self.all_pairs = dict(zip([el[0]+'_'+el[1]  for el in list(itertools.combinations(self.tickers_names.index,2))],
                                  list(itertools.combinations(self.tickers_names.index,2))))

        corr_matrix = self.dt_select_ret.corr()
        res = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)).stack())
        res.reset_index(level=0, drop=True, inplace=True)
        res.columns = ['corr']
        df = pd.DataFrame(res.to_dict()['corr'],index=[0])
        self.dct_pairs = {v[0]:k[0]+'-'+k[1] for k,v in df.to_dict().items()}
        
    @Helper.timing
    def visualize(self):
      dct = {k: self.dct_pairs[k] for k in list(self.dct_pairs)[:len(self.dct_pairs)]}
      dct_sorted = {el[0]: el[1] for el in sorted(dct.items(), key=operator.itemgetter(0))}
      data_y = list(dct_sorted.keys())
      data_x = [dct_sorted[i] for i in data_y]
      fig, ax = plt.subplots(figsize=(32,20))
      plt.title('Top correlated Nasdaq 100 pairs')
      ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
      plt.xticks(rotation=90)
      ax.scatter(data_x[-self.top_vals:], data_y[-self.top_vals:], c=data_y[-self.top_vals:], cmap='RdYlGn', vmin='-1', vmax='1')
      ax.set_ylabel('Correlation')
      ax.set_xlabel('Pairs')
      plt.grid(color='#C0C0C0', linestyle='--', linewidth=0.5)
      plt.show()

    @Helper.timing
    def get_tickers(self):
        if(self.index == 0):
            tickers_names = Helper.nasdaq100_tickers(self.urls[self.index])
            tickers_names = tickers_names[:96]
        elif(self.index == 1):
            tickers_names = Helper.sp500_tickers(self.urls[self.index])

        if(self.new_db_tickers):
                engine = create_engine("sqlite:///" + self.output_directory + "/tickers_names.db", echo=False)
                tickers_names.to_sql(
                    'tickers_names',
                    engine,
                    if_exists='replace',
                    index=True,
                    )
        else:
            engine = create_engine("sqlite:///" + self.output_directory + "/tickers_names.db", echo=False)

        self.tickers_names = pd.read_sql_table(
            'tickers_names',
            con=engine
            ).set_index('tickers')
        
    @Helper.timing
    def get_raw_prices(self):
        self.attrs = ['Adj Close','Close','High','Open','Low','Volume']
        self.symbs = self.tickers_names.index
        self.midx = pd.MultiIndex.from_product([self.attrs,self.symbs],names=('Attributes', 'Symbols'))
        
        if(self.new_db_raw_prices):
            dt_raw_prices = pdr.DataReader(self.tickers_names,'yahoo',self.start_date,self.end_date)
            dt_raw_prices['Dates'] = pd.to_datetime(dt_raw_prices.index,format='%Y-%m-%d')
            
            engine = create_engine("sqlite:///" + self.output_directory + "/dt_raw_prices.db", echo=False)
            dt_raw_prices.to_sql(
                'dt_raw_prices',
                engine,
                if_exists='replace',
                index=False
                )
        else:
            engine = create_engine("sqlite:///" + self.output_directory + "/dt_raw_prices.db", echo=False)

        self.dt_raw_prices = pd.read_sql_table(
            'dt_raw_prices',
            con=engine,
            parse_dates={'Dates': {'format': '%Y-%m-%d'}}
            )

        self.dt_raw_prices.rename(columns={"('Dates', '')":'Dates'},inplace=True)
        self.dt_raw_prices.set_index('Dates',inplace=True)
        self.dt_raw_prices.columns = self.midx
        
        # select the tickers based on ohlc parameters
        self.dt_select = self.dt_raw_prices.loc[:,self.dt_raw_prices.columns.get_level_values(0).isin(self.ohlc)]
        

        
