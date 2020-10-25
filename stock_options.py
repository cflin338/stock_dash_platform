 
"""
    def get_price_by_binomial_tree(self):
        
        This function will make the same calculation but by Binomial Tree
        
        n=30
        deltaT=self.t/n
        u = math.exp(self.vol*math.sqrt(deltaT))
        d=1.0/u
        # Initialize our f_{i,j} tree with zeros
        fs = [[0.0 for j in xrange(i+1)] for i in xrange(n+1)]
        a = math.exp(self.rf*deltaT)
        p = (a-d)/(u-d)
        oneMinusP = 1.0-p 
        # Compute the leaves, f_{N,j}
        for j in xrange(i+1):
            fs[n][j]=max(self.s * u**j * d**(n-j) - self.k, 0.0)
        print( fs)
 
        for i in xrange(n-1, -1, -1):
            for j in xrange(i+1):
                fs[i][j]=math.exp(-self.rf * deltaT) * (p * fs[i+1][j+1] +
                                                        oneMinusP * fs[i+1][j])
        print( fs)
 
        return fs[0][0]

"""

import datetime
from scipy.stats import norm
import numpy as np
import datetime as dt
from yahoo_fin import options
from pytrends.request import TrendReq
from yahoo_fin import stock_info as si
import requests
from bs4 import BeautifulSoup
import re #for regex

    

def get_current_risk_free_interest_rate():
    url= 'https://ycharts.com/indicators/10_year_treasury_rate'
    resp = requests.get(url)
    if resp.status_code==200:
        soup = BeautifulSoup(resp.text, 'html.parser')
        txt = soup.findAll('div', {'class': 'key-stat-title'})[0].get_text('self')
        val = re.search('\d.*%', txt).__getitem__(0)
        rate = float(val[:-1])
        return rate
    else:
        return 0.7

def years_to_maturity_calc(date):
    month_int = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November','December']
    month_today = month_int.index(date[0:date.index(' ')])+1
    date = date[date.index(' ')+1:]
    day_today = int(date[0:date.index(',')])
    year_today = int(date[-4:])
    strike_date_datetime = datetime.date(year_today, month_today, day_today)
    tod = datetime.date(year = datetime.datetime.today().year, month = datetime.datetime.today().month, day = datetime.datetime.today().day)
    years_to_maturity = (strike_date_datetime-tod).days/365
    
    return years_to_maturity

def option_breakdown(string):
    #symbol (max 6 char), Yr (YY), Mo (MM), Day (DD), C/P, Strike Price (_____.___)
    strike = string[-8:]
    strike = int(strike[0:5]) + .001*int(strike[5:])
    string = string[0:-8]
    option_type = string[-1]
    string = string[0:-1]
    date = string[-6:]
    date = dt.date(2000+int(date[0:2]),int(date[2:4]),int(date[4:]))
    symbol = string[0:-6]
    return {'symbol':symbol, 'date': date, 'option': option_type,'strike': strike}

def option_naming(ticker, price, date, option_type):
    #assume naming convention is 'Month Day, Year'
    month_int = ['January','February','March','April','May','June','July','August','September','October','November','December']
    month = str(month_int.index(date[0:date.index(' ')])+1)
    if len(month)<2:
        month = '0'+month
    date = date[date.index(' ')+1:]
    day = date[0:date.index(',')]
    if len(day)<2:
        day = '0'+day
    year = date[-2:]
    
    if len(ticker)>5:
        ticker = ticker[0:6]
    contract_name = ticker.upper() + year + month + day
    if option_type == 'Call':
        contract_name = contract_name + 'C'
    else:
        contract_name = contract_name + 'P'
    price = str(int(price*100))
    if len(price)<8:
        price = '0'*(8-len(price))+price
    return contract_name + price
    
    

def option_price(stock_price, strike_price, years_to_maturity,  IV, rate):
    #
    #solving black-scholes equation
    #
    A = np.exp(-rate * years_to_maturity)
    
    d1 = 1/(IV*np.sqrt(years_to_maturity)) * (np.log(stock_price/(strike_price*A)) + (.5*(IV**2)*years_to_maturity))
    d2 = 1/(IV*np.sqrt(years_to_maturity)) * (np.log(stock_price/(strike_price*A)) - (.5*(IV**2)*years_to_maturity))
    
    call = norm.cdf(d1)*stock_price - strike_price * A * norm.cdf(d2)
    put = -norm.cdf(-d1)*stock_price + strike_price * A *norm.cdf(-d2)
    
    return {'call': call, 'put': put}

def option_greeks(stock_price, strike_price, years_to_maturity, volatility, rate):
    
    #implied volatility: 1 standard deviation; 68% change stock will move v% in either direction within 1 year
    #       IE if IV = 90%, 68% chance (1 stdev) taht stock will increase or decrease 90% in one year
    #
    #as the VIX drops, implied volatility tends to be low in equities
    #
    d1 = (np.log(stock_price/strike_price) + (rate + .5*volatility**2)*years_to_maturity) / (volatility*np.sqrt(years_to_maturity))
    d2 = (np.log(stock_price/strike_price) + (rate - .5*volatility**2)*years_to_maturity) / (volatility*np.sqrt(years_to_maturity))
    
    b = np.exp(-rate*years_to_maturity)
    
    a = volatility * np.sqrt(years_to_maturity)
    
    delta_call = round(norm.cdf(d1),4)
    delta_put = -round(norm.cdf(-d1),4)
    
    gamma = round(norm.pdf(d1)/(stock_price * a),4)
    
    
    """
    call = -self.underlyingPrice * norm.pdf(self._d1_) * self.volatility / \
				(2 * self.daysToExpiration**0.5) - self.interestRate * \
				self.strikePrice * _b_ * norm.cdf(self._d2_)
		put = -self.underlyingPrice * norm.pdf(self._d1_) * self.volatility / \
				(2 * self.daysToExpiration**0.5) + self.interestRate * \
				self.strikePrice * _b_ * norm.cdf(-self._d2_)
    """                
    theta_call = round((-stock_price * norm.pdf(d1) * volatility/(2*np.sqrt(years_to_maturity)) + rate*strike_price*b * norm.cdf(d2))/365,4)
    theta_put = round((-stock_price * norm.pdf(-d1) * volatility / (2*np.sqrt(years_to_maturity)) + rate * strike_price * b * norm.cdf(-d2))/365,4)
    
    vega = round(stock_price * norm.pdf(d1) * years_to_maturity**.5/100, 4)
    
    rho_call = round(stock_price * years_to_maturity * b * norm.cdf(d2)/100,4)
    rho_put = -round(stock_price * years_to_maturity * b * norm.cdf(-d2)/100,4)
    
    call = {'delta': delta_call, 'gamma': gamma, 'theta': theta_call, 'vega': vega,'rho':rho_call}
    put = {'delta': delta_put, 'gamma': gamma, 'theta': theta_put, 'vega': vega,'rho':rho_put}
    
    return {'call': call, 'put': put}



def calculate_profit(type, strike_price, current_price, option_premium, purchase_premium, option_count):
    profits = {'execute':-option_count*purchase_premium*100, 'sell': (option_premium - purchase_premium)*option_count*100}
    
    if type=='call':
        profits['execute'] = 100*option_count*(current_price - strike_price) + profits['execute']
    elif type == 'put':
        profits['execute'] = 100*option_count*(strike_price - current_price) + profits['execute']
    return profits


def bulk_calc(all_dates,all_strikes, current_price, rate, IV_list):
    bulk_options = [option_greeks(current_price, strike_price, years_to_maturity, volatility, rate) for 
     strike_price, years_to_maturity, volatility in zip(all_strikes, all_dates, IV_list)]
    all_call_greeks = [i['call'] for i in bulk_options]
    all_put_greeks = [i['put'] for i in bulk_options]
    
    bulk_prices = [option_price(current_price, strike_price, years_to_maturity, volatility, rate) for 
     strike_price, years_to_maturity, volatility in zip(all_strikes, all_dates, IV_list)]
    all_call_prices = [i['call'] for i in bulk_prices]
    all_put_prices = [i['put'] for i in bulk_prices]
    
    return {'call':{'price': all_call_prices, 'greeks': all_call_greeks},
            'put':{'price': all_put_prices, 'greeks': all_put_greeks}}

def optimal_options_combination(ticker, date, price_today,risk_free_rate):
    all_options = options.get_options_chain(ticker, date)
    calls = all_options['calls'].to_dict('record')
    puts = all_options['puts'].to_dict('record')
    
    years_to_maturity = years_to_maturity_calc(date)
    
    good_combinations = []
    for c in calls:
        for p in puts:
            
            #####
            call_iv = float(c['Implied Volatility'][0:-1].replace(',',''))/100
            put_iv = float(p['Implied Volatility'][0:-1].replace(',',''))/100
            
            
            call_strike_price = c['Strike']
            put_strike_price = p['Strike']
            
            call_option_price = round(option_price(price_today, 
                                              call_strike_price, years_to_maturity, 
                                              call_iv,
                                              risk_free_rate)['call'],2)
            
            #call_option_price = np.max([c['Last Price'], c['Bid'], c['Ask']])
            
            put_option_price = round(option_price(price_today, 
                                              put_strike_price, years_to_maturity, 
                                              put_iv,
                                              risk_free_rate)['put'],2)
            
            #put_option_price = np.max([p['Last Price'], p['Bid'], p['Ask']])
            
            if (c==20) & (p == 25):
                print([call_option_price, put_option_price])
            
            total_option_cost = call_option_price + put_option_price
            
#            c_xval = [0, call_strike_price, call_strike_price + call_option_price, call_strike_price * 1.5]
#            p_xval = [0, max(0, put_strike_price - put_option_price), put_strike_price, put_strike_price*1.5]
            
            combined_xval = np.sort(np.unique([0, call_strike_price, call_strike_price+call_option_price, max(call_strike_price *1.5, (call_strike_price+call_option_price)*1.2),
                                     max(0,put_strike_price-put_option_price), put_strike_price, put_strike_price*1.5]))
            
#            combined_xval = np.sort(np.unique(np.array(c_xval+p_xval)))
            overall_profit = []
            
            for v in combined_xval:
                
                temp_prof = 0
                if v<=call_strike_price:
                    temp_prof-=call_strike_price
                else:
                    temp_prof+=(v-call_strike_price)
                if v>=put_strike_price:
                    temp_prof-=put_strike_price
                else:
                    temp_prof+=(put_strike_price-v)
                overall_profit.append(max(v-call_strike_price,0) + max(put_strike_price-v,0) - total_option_cost)
            
            if np.min(overall_profit)>1:
                good_combinations.append({'profit': [combined_xval, overall_profit], 'call': c, 'put': p})
    return good_combinations

#print(optimal_options_combination('amd', 'November 20, 2020',83.02, .0069))
           
def daily_historical_trends(ticker):
    #returns total daily searches for last 10 years for the ticker name, ticker name + "news", ticker name + "stocks"
    pytrends = TrendReq(hl = 'en-US', tz = 360) 
    #time zone = 360 = us central, hl = host language (always use end-US)
    kw = [ticker, ticker + ' news', ticker + ' stock']
    pytrends.build_payload(kw, cat=0,)
    dat = pytrends.interest_over_time()
    return dat

def moving_average(series, avg_type, length):
    tmp_vals = series[0:length]
    averages = []
    if avg_type == 'standard':
        averages.append(np.mean(tmp_vals))
        for k in range(length,len(series)):
            tmp_vals.pop(0)
            tmp_vals.append(series[k])
            averages.append(np.mean(tmp_vals))
            
    elif avg_type ==  'exponential':
        alpha= .95
        
        alpha_list = [alpha**i for i in range(length)]
        alpha_sum = np.sum(alpha_list)
        tmp_vals.reverse()
        averages.append(np.mean([i*j for (i,j) in zip(tmp_vals,alpha_list)])/(1-alpha))
        tmp_vals.reverse()
        for k in range(length, len(series)):
            tmp_vals.pop(0)
            tmp_vals.append(series[k])
            tmp_vals.reverse()
            averages.append(np.mean([i*j for (i,j) in zip(tmp_vals, alpha_list)])/(1-alpha))
            tmp_vals.reverse()
            
    elif avg_type == 'weighted':
        alpha_list = list(range(1,length+1))
        wma_sum = sum(alpha_list)
        averages.append(sum([i*j for (i,j) in zip(alpha_list, tmp_vals)])/wma_sum)
        for k in range(length, len(series)):
            tmp_list.pop(0)
            tmp_list.append(series[k])
            averages.append(sum([i*j for (i,j) in zip(alpha_list, tmp_vals)])/wma_sum)
        
        
    return averages
        




class stock_custom():
    def __init__(self):
        self.ticker = ''
        self.current_price = 0
        self.stats = [{'Attribute':'', 'Value':0}]
        self.historical =0
        self.risk_free_rate = 0
        self.prices_today = []
        self.todays_times = []
        self.option_dates = []
        self.observed_options = {}
        
    def update(self,new_ticker, rate):
        old_ticker = self.ticker
        self.risk_free_rate = rate/100
        self.ticker = new_ticker.upper()
        if (self.ticker!='') & (new_ticker!=old_ticker):
            self.current_price = si.get_live_price(self.ticker)
            self.todays_times = [datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")[11:]]
            self.prices_today = [si.get_live_price(self.ticker)]
            self.historical = si.get_data(self.ticker)
            self.option_dates = options.get_expiration_dates(new_ticker)
            self.observed_options = {}
            try:
                self.stats = si.get_stats(self.ticker).to_dict('records')
            except ValueError:
                self.stats = [{'Attribute':'none available', 'Value':'none available'}]
            
    def update_price(self):
        if self.ticker!='':
            self.todays_times.append(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")[11:])
            self.prices_today.append(si.get_live_price(self.ticker))
            self.current_price = self.prices_today[-1]
    
    def update_options_list(self, option_type, strike, date):
        empty = {'Last Trade Date': '', 
                 'Strike':'',
                 'Last Price': '',
                 'Bid': '',
                 'Ask':'',
                 'Change':'',
                 '% Change':'',
                 'Volume':'',
                 'Open Interest':'',
                 'Implied Volatility': ''}  
        contract_name = option_naming(self.ticker, strike, date, option_type)
        orig_date = date
        if contract_name not in self.observed_options:
            #breakdown date string    
            """
            month_int = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November','December']
            month_today = month_int.index(date[0:date.index(' ')])+1
            date = date[date.index(' ')+1:]
            day_today = int(date[0:date.index(',')])
            year_today = int(date[-4:])
            strike_date_datetime = datetime.date(year_today, month_today, day_today)
            tod = datetime.date(year = datetime.datetime.today().year, month = datetime.datetime.today().month, day = datetime.datetime.today().day)
            years_to_maturity = (strike_date_datetime-tod).days/365
            """
            years_to_maturity = years_to_maturity_calc(date)
            #get implied volatility
            if option_type.lower()=='call':
                df = options.get_calls(self.ticker, orig_date)
            else:
                df = options.get_puts(self.ticker, orig_date)
            iv = float(df.loc[df['Strike']==strike]['Implied Volatility'].to_list()[0][0:-1].replace(',',''))/100
    
            #calculate greeks        
            grks = option_greeks(self.prices_today[-1], 
                                              strike, years_to_maturity, 
                                              iv,
                                              self.risk_free_rate)[option_type.lower()]
            
            #calculate prices
            price = round(option_price(self.prices_today[-1], 
                                              strike, years_to_maturity, 
                                              iv,
                                              self.risk_free_rate)[option_type.lower()],2)
            
            opt = {'Expir Date': orig_date,
                   'Strike Price': '${}'.format(strike)}
            
            opt['Value'] = '${}'.format(price)
            opt['Type'] = option_type
            opt['Implied Vol'] = '{}%'.format(np.round(iv*100,4))
            opt.update(grks)
            self.observed_options[contract_name] = opt
        return self.observed_options
    
                
    """
    ###
    #currently unused
    ###
    def add_observed_option(self, option_type, strike, date, greeks):
        new_opt = {'type': option_type,
                    'strike':strike,
                    'date': date, 
                    'greeks': greeks}
        if new_opt not in self.observed_options:
            self.observed_options.append(new_opt)
            
    def remove_observed_option(self, option_type, strike, date, greeks):
        new_opt = {'type': option_type,
                    'strike':strike,
                    'date': date, 
                    'greeks': greeks}
        if new_opt in self.observed_options:
            self.observed_options.remove(new_opt)
    """
    

    
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

"""
greeks

delta
-change in price of option for every dollar change of stock price
-estimates probability of expiring in the money; delta% of expiring ITM

gamma
-rate of change of delta
-after first dollar change of stock, how much does delta increase or decrease

theta
-decay of option price with each day
-additive
-against option buy, for option seller
-increases as it gets closer to expiring date
    -what's rate of change for theta? is it exponential? linear?

vega
-how much option price changes with change in implied volatility

rho
-interest rate change's impact
-small impact on options pricing

implied volatility
-1 year standard deviation
-gives range of value
-increases in bearish market, decreases in bullish market
- * sqrt(day/365) for time period other than 1 year



"""




"""
strategies
:straddle, same strike price, same expiration date
:strangle, diff strike price, same expiration date
"""

