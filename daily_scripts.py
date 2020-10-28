# -*- coding: utf-8 -*-
"""
----daily scan----

scan for low float stocks
    -take float / outstanding... [10]/[9]
scan for high short %
    -[15]; % short of float, [16] % short of outstanding
check change in float/short% values, which stocks anewly crossed threshold, 
    which stocks had significant changes in these values
calculate greeks and prices for all options on specified dates/ tickers

"""
import numpy as np
import stock_options
import datetime as dt
from yahoo_fin import options
from yahoo_fin import stock_info as si
import pandas as pd
import os
import datetime
from openpyxl import load_workbook

#xl = pd.ExcelFile('foo.xls')
#xl.sheet_names  # see all sheet names
#xl.parse(sheet_name)  # read a specific sheet to DataFrame

def get_options_stored(file, tab):
    header = pd.read_excel(file,tab).columns.to_list()
    options = [i for i in header if 'Unnamed' not in i]
    options.remove('Rate')
    options.remove('Price Today')
    return options

def calc_existing_contracts(t, d, contracts,years_to_maturity, rate, stock_price,header1, header2, row):
    all_contracts= options.get_options_chain(t, d)
    for c in contracts:
        
        if stock_options.option_breakdown(c)['option']=='C':
            iv = all_contracts['calls'].loc[all_contracts['calls']['Contract Name']==c]['Implied Volatility'].to_list()[0][0:-1]
            if ',' in iv:
                iv = iv.replace(',', '')
            implied_volatility = float(iv)/100
            strike_price = all_contracts['calls'].loc[all_contracts['calls']['Contract Name']==c]['Strike'].to_list()[0]
            calc = stock_options.option_greeks(stock_price, strike_price, years_to_maturity, implied_volatility, rate)['call']
            calculated_price = stock_options.option_price(stock_price, strike_price, years_to_maturity, implied_volatility, rate)['call']
        else:
            iv = all_contracts['puts'].loc[all_contracts['puts']['Contract Name']==c]['Implied Volatility'].to_list()[0][0:-1]

            if ',' in iv:
                iv = iv.replace(',', '')
            implied_volatility = float(iv)/100
            strike_price = all_contracts['puts'].loc[all_contracts['puts']['Contract Name']==c]['Strike'].to_list()[0]
            calc = stock_options.option_greeks(stock_price, strike_price, years_to_maturity, implied_volatility, rate)['put']
            calculated_price = stock_options.option_price(stock_price, strike_price, years_to_maturity, implied_volatility, rate)['put']
        delta = calc['delta']
        gamma = calc['gamma']
        theta = calc['theta']
        vega = calc['vega']
        rho = calc['rho']
        
        row.append(calculated_price)
        row.append(implied_volatility)
        row.append(delta)
        row.append(gamma)
        row.append(theta)
        row.append(vega)
        row.append(rho)
        #extend header
        header1 = header1 + [c] * 7
        header2 = header2 + ['Option Price', 'Implied Volatility', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
    return header1, header2, row

def existing_dates_calc(existing_dates, stock_price, date_today, rate,historical_loc, t):
    writer = pd.ExcelWriter(historical_loc + t + '.xlsx', engine = 'openpyxl')
    writer.book = load_workbook(historical_loc + t + '.xlsx')
    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    reader = pd.read_excel(r''+historical_loc+t+'.xlsx')
    
    for d in existing_dates:
        print('updating date tab {} for {}'.format(d, t))
        years_to_maturity = stock_options.years_to_maturity_calc(d)
        
        #get list of contract values previously collected
        contracts = get_options_stored(historical_loc+t+'.xlsx', d) #should include both calls and puts
        header1 = ['Date', 'Price Today', 'Rate']
        header2 = ['', '', '']
        row = [date_today, stock_price, rate]
        print('updating existing contract values')
        header1, header2, row = calc_existing_contracts(t, d, contracts, years_to_maturity, rate, stock_price, header1, header2, row)
        
        #look through contracts not already listed; if any new contracts for the given date, calculate greeks for the new contracts and expand 'contract names' list
        all_contracts = options.get_options_chain(t, d)
        lst1 = all_contracts['calls']['Contract Name'].to_list()
        lst2 = all_contracts['puts']['Contract Name'].to_list()
        all_contracts = lst1+lst2
        new_contracts = [c for c in all_contracts if c not in contracts]
        print('adding {} new contracts to existing date {} for {}'.format(len(new_contracts), d, t))
        header1, header2, row = calc_existing_contracts(t, d, new_contracts, years_to_maturity, rate, stock_price, header1, header2, row)
        
        col_names = pd.MultiIndex.from_tuples(list(zip(*[header1, header2])))
        df = pd.DataFrame( data = [row], columns = col_names)
        df = df.set_index('Date')
        
        
        #
        
        
        #new row added
        df.to_excel(writer,d, header = False, startrow = len(reader) + 1)
        #update header
    writer.save()
    return 
def new_dates_calc(new_dates, stock_price, date_today, rate,historical_loc, t):
    #creates new file if ticker hasnt been tracked yet
    writer = pd.ExcelWriter(historical_loc + t+'.xlsx', engine = 'openpyxl')
    
    for d in new_dates:
        print('creating new date tab for {}, date {}'.format(t, d))
        years_to_maturity = stock_options.years_to_maturity_calc(d)
        
        #get list of contract values previously collected
        #contracts = get_options_stored(historical_loc+t+'.xlsx', d) #should include both calls and puts
        header1 = ['Date', 'Price Today', 'Rate']
        header2 = ['', '', '']
        row = [date_today, stock_price, rate]
        
        #header1, header2, row = calc_existing_contracts(t, d, contracts, 
        #                                                years_to_maturity, rate, stock_price, header1, header2, row)
        
        #look through contracts not already listed; if any new contracts for the given date, calculate greeks for the new contracts and expand 'contract names' list
        try:
            all_contracts = options.get_options_chain(t, d)
            lst1 = all_contracts['calls']['Contract Name'].to_list()
            lst2 = all_contracts['puts']['Contract Name'].to_list() 
            all_contracts = lst1+lst2
            header1, header2, row = calc_existing_contracts(t, d, all_contracts, years_to_maturity, rate, stock_price, header1, header2, row)
        except ValueError:
            all_contracts = []
            print('no current contracts for {}'.format(d))
        
        col_names = pd.MultiIndex.from_tuples(list(zip(*[header1, header2])))
        df = pd.DataFrame( data = [row], columns = col_names)
        df = df.set_index('Date')
        
        df.to_excel(writer, d)
    writer.save()
    #for each date, create new tab and write df to historical_loc + t
    return
def daily_options_calc(target_folder, tickers):
    #first change all tickers to uppercase
    tickers = [i.upper() for i in tickers]
    #first check to make sure all tickers are valid:
    for t in tickers:
        try:
            si.get_live_price(t)
        except:
            print('')
        
    #each ticker will have a separate excel file/collection of databases corresponding with a ticker
    historical_loc = target_folder
    #FIRST GET LIST OF TICKERS WITH DATA ALREADY COLLECTED, INTERSECT WITH LIST OF TICKERS REQUESTED
    existing_tickers = [i[0:i.index('.')] for i in os.listdir(historical_loc)]
    
    new_tickers = [i for i in tickers if i not in existing_tickers]    
    rate = stock_options.get_current_risk_free_interest_rate()/100
    date_today = datetime.datetime.today()
    date_today = '{}-{}-{}'.format(date_today.month, date_today.day, date_today.year)
    for t in existing_tickers:
        print('updating {}'.format(t))
        #stock price for today
        stock_price = si.get_live_price(t)
        #all expiration dates
        all_dates = options.get_expiration_dates(t)
        
        existing_dates = pd.ExcelFile(historical_loc+t+'.xlsx').sheet_names
        
        new_dates = [i for i in all_dates if i not in existing_dates]
        #first get list of tabs available
        #each date will correspond with a differnt tab in excel sheet/ different table/ different database
        print('updating options information on existing dates')
        existing_dates_calc(existing_dates, stock_price, date_today, rate,historical_loc, t)
        if len(new_dates)>0:
            print('getting values for new options on new dates')        
            new_dates_calc(new_dates, stock_price, date_today,rate,historical_loc, t)
    
    for t in new_tickers:
        print('new file for {}'.format(t))
        stock_price = si.get_live_price(t)
        all_dates = options.get_expiration_dates(t)
        new_dates_calc(all_dates, stock_price, date_today, rate, historical_loc, t)
    
    return 
target_folder = 'C:\\Users\\clin4\\Documents\\py\\stocks\\stock_dash_platform\\historical_collection\\'
daily_options_calc(target_folder, ['cvx', 'amd', 'aapl', 'wmt', 'mcd', 'fnko'])
#daily_options_calc(target_folder, ['fnko'])