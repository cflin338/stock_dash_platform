import numpy as np
from yahoo_fin import stock_info as si
from plotly.subplots import make_subplots
import stock_options
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table
import plotly.graph_objs as go
import scipy.stats as stats
from dash.dependencies import Input, Output, State
import datetime
from plotly.graph_objs.scatter.marker import Line

import json
from yahoo_fin import options

from stock_options import stock_custom

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
   
"""
option_price(stock_price, strike_price, years_to_maturity,  IV, rate)
option_breakdown(string)

option_greeks(stock_price, strike_price, years_to_maturity, volatility, rate)
bulk_calc(all_dates,all_strikes, current_price, rate, IV_list)
    
intrinsic value: max(0, (current_price - strike_price)*(-1)**(option_type == 'put'))
time value: option_price - intrinsic value
"""
    
examined_stock = stock_custom()

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div(children = [
    html.H3("Ticker Name and Risk Free Interest Rate"),
    
    html.Div(children = [
        dcc.Input(id = 'ticker-input', value = '', type = 'text'),
        dcc.Input(id = 'rate-input', value = stock_options.get_current_risk_free_interest_rate(), type = 'number'),
        html.Button(id = 'update-ticker-info', n_clicks = 0, children = 'Start Tracking Live Price'),
        html.Button(id = 'fill-ticker-table-info', n_clicks = 0, children = 'Fill Table'),
        ],
        className = 'row'),
    
    html.Div(id='ticker-output', className = 'row'),
    
    html.Div(children = [
        html.Div(children = [
            dash_table.DataTable(id = 'stockoverview-table', 
                             columns = [{'name': 'Attribute', 'id': 'Attribute'}, {'name': 'Value', 'id': 'Value'}], 
                             data = [{'Attribute':'', 'Value':0}],
                             page_action='none',
#                             style_table={'height': '300px', 'width': '500px','overflowY': 'auto'},
                             fixed_rows={'headers': True}) ], 
            className = 'six columns'),
        html.Div(children = [
            dcc.Graph(id='live-price-graph',
#                  style = {'width': 800}
                  )], 
            className = 'six columns')
        
        ], className = 'row'),#style = {'display': 'inline-block'}, 
    
    dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
        ),
    html.Div(children = [
        dcc.Input(id = 'prev-date-select', value = datetime.datetime.today().strftime("%d/%m/%Y %H:%M:%S")[0:10], type = 'text'),
        html.Button(id = 'display-candle', n_clicks = 0, children = 'Update'),
        ], className = 'row'),
    
    html.Br(),
    
    html.Div(children = [
        dcc.Graph(id = 'historical-candlestick-graph')],className = 'row' ),
    
    html.Div(children = [
        dcc.Graph(id = 'macd-overlay-graph')], className = 'row'),
    
    html.H4('Options Details'),
    
    html.Div(children = [
        html.Div(children = [dcc.Dropdown(id = 'contract-dates', 
                                          options = [{'label':'','value':''}], 
#                                          value = '',
                                          placeholder = 'Select Strike Date'),], className = 'two columns'), #'height': '400px', 
        html.Div(children = [html.Button(id = 'show-options-info', n_clicks = 0, children = 'Display Options Info')], className = 'two columns'),
        ],
        className = 'row'),
    
    html.Div(children = [
        html.Div(children = [
            html.H5('Call Options'),
            dash_table.DataTable(id = 'options-calls-table',
                            columns = [{'name':'Contract Name', 'id':'Contract Name'}, 
                                       {'name':'Last Trade Date', 'id':'Last Trade Date'}, 
                                       {'name':'Strike', 'id':'Strike'}, 
                                       {'name':'Last Price', 'id':'Last Price'},
                                       {'name':'Bid','id':'Bid'},
                                       {'name':'Ask','id':'Ask'}, 
                                       {'name':'Change','id':'Change'}, 
                                       {'name':'% Change','id':'% Change'}, 
                                       {'name':'Volume','id':'Volume'}, 
                                       {'name':'Open Interest','id':'Open Interest'},
                                       {'name':'Implied Volatility','id':'Implied Volatility'}],
                            data = [{'Contract Name':'', 
                                     'Last Trade Date': '', 
                                     'Strike':'',
                                     'Last Price': '',
                                     'Bid': '',
                                     'Ask':'',
                                     'Change':'',
                                     '% Change':'',
                                     'Volume':'',
                                     'Open Interest':'',
                                     'Implied Volatility': ''}],
                            style_table={'overflowY': 'auto'},#'height': '400px', 
                            fixed_rows={'headers': True}),
            ], className = 'six columns'),
        html.Div(children = [
            html.H5('Put Options'),
            dash_table.DataTable(id = 'options-puts-table',
                            columns = [{'name':'Contract Name', 'id':'Contract Name'}, 
                                       {'name':'Last Trade Date', 'id':'Last Trade Date'}, 
                                       {'name':'Strike', 'id':'Strike'}, 
                                       {'name':'Last Price', 'id':'Last Price'},
                                       {'name':'Bid','id':'Bid'},
                                       {'name':'Ask','id':'Ask'}, 
                                       {'name':'Change','id':'Change'}, 
                                       {'name':'% Change','id':'% Change'}, 
                                       {'name':'Volume','id':'Volume'}, 
                                       {'name':'Open Interest','id':'Open Interest'},
                                       {'name':'Implied Volatility','id':'Implied Volatility'}],
                            data = [{'Contract Name':'', 
                                     'Last Trade Date': '', 
                                     'Strike':'',
                                     'Last Price': '',
                                     'Bid': '',
                                     'Ask':'',
                                     'Change':'',
                                     '% Change':'',
                                     'Volume':'',
                                     'Open Interest':'',
                                     'Implied Volatility': ''}],
                            style_table={'overflowY': 'auto'},#'height': '400px', 
                            fixed_rows={'headers': True}),
            ], className = 'six columns'),
        ], className = 'row'),
    
    html.Br(),
    
    html.Div(children = [
        html.H3("Options Greeks"),
        html.Div(children = [
            html.Div(children = [dcc.Dropdown(id = 'analysis1-contract-type-selection', style = {'width': 250}, options = [{'label':'Call','value':'Call'}, {'label':'Put','value':'Put'}], placeholder = 'Select Option Type' ),],
                     className = 'two columns'),
            html.Div(children = [dcc.Dropdown(id = 'analysis1-contract-date-selection', style = {'width': 250}, options = [{'label':'','value':''},], placeholder = 'Select Strike Date' ),],
                     className = 'two columns'),
            html.Div(children = [html.Button(id = 'analysis1-fill-options-prices-button', n_clicks = 0, children = 'Fill Strike Prices')], 
                     className = 'two columns'),
            html.Div(children = [dcc.Dropdown(id = 'analysis1-contract-strike-selection', style = {'width': 250}, options = [{'label':'','value':''}], placeholder = 'Select Strike Price', ),],
                     className = 'two columns'),
            html.Div(children = [html.Button(id = 'analysis1-display-options-greeks-button', n_clicks = 0, children = 'Calc Options Info')], 
                     className = 'two columns'),
            ],className = 'row', ),
        
        ],
        ),
    
    html.Br(),
    
    html.Div(children = [
        html.Div(children = [dash_table.DataTable(id = 'options-greeks-table',
                            columns = [{'name': 'Contract Name', 'id': 'Contract Name'},
                                       {'name': 'Expir Date', 'id': 'Expir Date'},
                                       {'name': 'Strike Price', 'id': 'Strike Price'},
                                       {'name':'Value', 'id': 'Value'},
                                       {'name':'Type', 'id': 'Type'},
                                       {'name': 'Implied Vol', 'id': 'Implied Vol'},
                                       {'name':'delta', 'id':'delta'}, 
                                       {'name':'gamma', 'id':'gamma'}, 
                                       {'name':'theta', 'id':'theta'}, 
                                       {'name':'vega', 'id':'vega'},
                                       {'name':'rho', 'id':'rho'}],
                            data = [{'Contract Name': '',
                                     'Expir Date': '',
                                     'Strike Price': '',
                                     'Value': '',
                                     'Type': '',
                                     'Implied Vol': '',
                                     'delta':'', 
                                     'gamma': '', 
                                     'theta':'',
                                     'vega':'',
                                     'rho': '',}],
                            fixed_rows={'headers': True}),], className = 'six columns'),
        ],
        className = 'row'),
    html.Br(),
    #up to 3 lines, one for overall profit, others are profits from up to two contracts
    html.Div(children = [
        html.Div(children = [dcc.Dropdown(id = 'options-to-display-dropdown',placeholder = 'Select Options To Plot', 
                     multi = True, clearable = True)], className = 'three columns'),
        html.Button(id = 'plot-options-possibilities-button', n_clicks = 0, children = 'Plot Projections'),
        ],
        className = 'row'),
    html.Div(id = 'possible-radio-items-listing', #children = ['Buy/Sell Selection Displayed Here After Options Selected'
#        dcc.RadioItems(options = [{'label': 'Buy', 'value': 'Buy'},{'label':'Sell', 'value': 'Sell'}], labelStyle = {'display': 'inline-block'}),
#       will fill childrens list with radio items das more options selected to be plotted
        #],
        className = 'row'),
    html.Div(children = [
        html.Div(id = 'projected-pl-display'),
        html.Div(children = [dcc.Graph(id = 'options-plot-overlay' )], 
                 className = 'row'),],
        className = 'row'
        ),
    html.Div(children = [
        html.Div(children = [
            html.Button(id = 'search-contract-combinations',children = 'Check possible C/P combinations for Profit', n_clicks = 0),
            html.Div(id = 'optimal-search-results',)
            ])],
        className = 'row'),


])

@app.callback(Output(component_id = 'possible-radio-items-listing', component_property = 'children'),
              Input(component_id = 'options-to-display-dropdown', component_property = 'value'))
def show_options_radioitems_buy_call(dropdown_items):
    if dropdown_items is None:
        return ['Buy/Sell Selection Displayed Here After Options Selected']
#    print(len(dropdown_items))
    if len(dropdown_items)==0:
        return ['no options selected']
    radio_list = []
    for i in dropdown_items:
        radio_list.append(dcc.RadioItems(options = [{'label': 'Buy '+i, 'value': 'Buy '+i}, {'label': 'Sell '+i, 'value': 'Sell '+i}], 
                           labelStyle = {'display': 'inline-block'}))
    return radio_list

@app.callback(Output(component_id = 'optimal-search-results', component_property = 'children'),
              Input(component_id = 'search-contract-combinations', component_property = 'n_clicks'))
def eval_all_combinations_for_profits(clicks):
    if clicks>0:
        #loop through dates
        good_combinations = 0
        combination_list = {}
        print('checking for possible profitable combinations...')
        print(len(examined_stock.option_dates))
        for d in examined_stock.option_dates:
            opt = stock_options.optimal_options_combination(examined_stock.ticker, d,examined_stock.current_price, examined_stock.risk_free_rate)
            if len(opt)>0:
                good_combination+=len(opt)
                combination_list[d] = opt
        return 'number of good combinations: {} over {} dates'.format(good_combinations, len(examined_stock.option_dates))
    else:
        return 'n/a'

#
#-----------------------------------------------------------------------------------
#
#function: display option profitability graph
#
#initiation: press 'plot options possiblities' button option(s) selected
#
#-----------------------------------------------------------------------------------
@app.callback([Output(component_id = 'options-plot-overlay', component_property = 'figure'),
               Output(component_id = 'projected-pl-display', component_property = 'children')],
              Input(component_id = 'plot-options-possibilities-button', component_property = 'n_clicks'),
              [State(component_id = 'options-to-display-dropdown', component_property = 'value'),
               State(component_id = 'possible-radio-items-listing', component_property = 'children')])

def display_options_profitabilities_plot(clicks, values, buy_sell):
    selections = {}
    if buy_sell is not None:
        for i in buy_sell:
            temp_str = i['props']['value']
            selections[temp_str[temp_str.index(' ')+1:]] = temp_str[0:temp_str.index(' ')]
            
    fig = make_subplots(rows=1, cols=2, x_title = 'Stock Price($)',y_title = 'Profit($)',
                    subplot_titles = ['Contract(s) Profit/Loss($)', 'Overall Profit/Loss($)'],
                    specs = [[{"secondary_y": True}, {"secondary_y": True}],])
    if (values is None) | (values==[]):
        fig.add_scatter(x = [-.05,1], y = [0, 0], mode = 'lines', marker = dict(color = 'Grey'),
                        row = 1, col = 1, showlegend=False)
        fig.add_scatter(x = [0,0], y = [-.05, 1], mode = 'lines', marker = dict(color = 'Grey'), 
                        row = 1, col = 1, showlegend=False)
        fig.add_scatter(x = [-.05,1], y = [0, 0], mode = 'lines', marker = dict(color = 'Grey'), 
                        row = 1, col = 2, showlegend=False)
        fig.add_scatter(x = [0,0], y = [-.05, 1], mode = 'lines', marker = dict(color = 'Grey'), 
                        row = 1, col = 2, showlegend=False)
        return fig,''
    else:
        #plot individual options
        total_option_cost = 0
        
        xlist = [0]
        ylist = []
        typelist = []
        pricelist = []
        strikepricelist = []
        
        sigma_list = []
        
        for v in values:
            dat = examined_stock.observed_options[v]
            date = dat['Expir Date']
            
            years_to_maturity = stock_options.years_to_maturity_calc(date)
            sigma_list.append(float(dat['Implied Vol'][0:-1])/100*examined_stock.current_price*np.sqrt(years_to_maturity))

            option_price = float(dat['Value'][1:])
            pricelist.append(option_price)
            total_option_cost+=option_price
            option_type = dat['Type']
            typelist.append(option_type)
            
            option_strike_price = float(dat['Strike Price'][1:])
            strikepricelist.append(option_strike_price)
            
            
            xlist.append(option_strike_price+option_price)
            xlist.append(option_strike_price*1.5)
            xlist.append((option_strike_price+option_price)*1.1)
            xlist.append(examined_stock.current_price+2*float(dat['Implied Vol'][0:-1])/100*examined_stock.current_price*np.sqrt(years_to_maturity))
        
        minX = np.min(np.array(xlist))
        maxX = np.max(np.array(xlist))
        
        xvals_for_profits = np.linspace(minX,maxX, 200)
        for p,s,t,v in zip(pricelist, strikepricelist,typelist,values):
            if t.lower()=='call':
                ylist.append([(-1)**(selections[v]=='Sell') * max(-p, x - s - p) for x in xvals_for_profits])
            else:
                ylist.append([(-1)**(selections[v]=='Sell') * max(-p, s - x - p) for x in xvals_for_profits])
                
        for ind in range(len(values)):
            fig.add_scatter(x =xvals_for_profits ,y=ylist[ind], mode="lines",
                            marker=dict(size=5, color="LightSeaGreen"),name = values[ind],row=1, col=1)

        #calculate profits
        profits = sum([np.array(i) for i in ylist])
        
        minY = min(np.min(profits),np.min(np.array(ylist).flatten()))
        maxY = max(np.max(np.array(ylist).flatten()),np.max(profits))
        
        #plot profit
        fig.add_scatter(x = xvals_for_profits, y = profits, mode="lines",
                        marker=dict(size=5, color="MediumPurple"),name="P/L", row=1, col=2, secondary_y = False)
 
        #plot axis
        fig.add_scatter(x = [minX-.1,maxX+.1], y = [0, 0], mode = 'lines', marker = dict(color = 'Black'),
                        row = 1, col = 1, showlegend=False)
        fig.add_scatter(x = [minX-.1,minX-.1], y = [minY-.1, maxY+.1], mode = 'lines', marker = dict(color = 'Black'), 
                        row = 1, col = 1, showlegend=False)
        fig.add_scatter(x = [minX-.1,maxX+.1], y = [0, 0], mode = 'lines', marker = dict(color = 'Black'), 
                        row = 1, col = 2, showlegend=False, secondary_y = False)
        fig.add_scatter(x = [minX-.1,minX-.1], y = [minY-.1, maxY+.1], mode = 'lines', marker = dict(color = 'Black'), 
                        row = 1, col = 2, showlegend=False, secondary_y = False)
        
        #plot normal distribution on second plot, but different y-axis
        mu = examined_stock.current_price
        sigma = np.mean(sigma_list)
        
        norm_distr_x = xvals_for_profits
        norm_distr_y = stats.norm.cdf(norm_distr_x,mu,sigma)
        
        fig.add_scatter(x = norm_distr_x, y = norm_distr_y, mode = 'lines', marker = dict(color = 'Grey'),
                        row = 1, col = 2, showlegend = True, secondary_y = True, name = 'Cumulative Normal Distr')
          
        #how should standard deviation of multiple options be calculated? 
        #what makes the most sense?
        #for now, will simply use average of the contracts
        norm_distr_pdf_y = stats.norm.pdf(norm_distr_x,mu,sigma)
        
        fig.add_scatter(x = norm_distr_x, y = norm_distr_pdf_y, mode = 'lines', marker = dict(color = 'Grey'),
                        row = 1, col = 2, showlegend = True, secondary_y = True, name = 'Normal Distr')
        
        estimated_profit = np.sum([i*j for i,j in zip(norm_distr_pdf_y, profits)])
        
        return fig, 'Estimated Profit/Loss: ${}'.format(np.round(estimated_profit,2))
    
#
#-----------------------------------------------------------------------------------
#
#function: display option greeks in table and option price
#
#initiation: press 'show options greeks' button AFTER price and option type selected
#
#-----------------------------------------------------------------------------------
#currently only buying options; implement selling options as well; simply involves flip over x-axis
@app.callback(
    [Output(component_id = 'options-to-display-dropdown', component_property = 'options'),
    Output(component_id = 'options-greeks-table', component_property = 'data')],
    Input(component_id = 'analysis1-display-options-greeks-button', component_property = 'n_clicks'),
    [State(component_id = 'analysis1-contract-strike-selection', component_property = 'value'),
     State(component_id = 'analysis1-contract-type-selection', component_property = 'value'), 
     State(component_id = 'analysis1-contract-date-selection', component_property = 'value')])

def display_option_greeks_table(clicks, strike,  option_type,date):

    if date is None:
        return [{'label': '', 'value': ''}],[{'Contract Name': '','Expir Date': '','Strike Price': '','Value': '',
                 'Type': '', 'Implied Vol': '','delta':'','gamma': '', 'theta':'',
                 'vega':'','rho': '',}]
    else:
        dat = examined_stock.update_options_list(option_type, strike, date)
        names = [i for i in dat]
        info = [dat[i] for i in dat]
        
        opt = []
        for i in range(len(names)):
            tmp = {'Contract Name': names[i]}
            tmp.update(info[i])
            opt.append(tmp)

        return [{'label': i, 'value': i} for i in names],opt

#-----------------------------------------------------------------------------------
#
#function: fill options dropdown with list of strike prices for available options
#
#initiation: press 'fill strike prices' button AFTER price and option type selected
#
#-----------------------------------------------------------------------------------

@app.callback(
    Output(component_id = 'analysis1-contract-strike-selection', component_property = 'options'),
    Input(component_id = 'analysis1-fill-options-prices-button', component_property = 'n_clicks'),
    [State(component_id = 'analysis1-contract-date-selection', component_property = 'value'),
     State(component_id = 'analysis1-contract-type-selection', component_property = 'value')])

def update_options_strike_dropdown(clicks, date, option_type):
    if examined_stock.ticker!='':
        strike_prices = options.get_options_chain(examined_stock.ticker,date)[option_type.lower()+'s'].Strike.to_numpy()
    
        return [{'label':float(i), 'value':float(i)} for i in strike_prices]
    else:
        return [{'label':'', 'value':''}]


#-----------------------------------------------------------------------------------
#
#function: fill call and put tables 
#
#initiation: press 'show options info' button AFTER expiration date has been selected
#
#-----------------------------------------------------------------------------------

@app.callback(
    [Output(component_id = 'options-calls-table', component_property = 'data'),
     Output(component_id = 'options-puts-table', component_property = 'data')],
    Input(component_id = 'show-options-info', component_property = 'n_clicks'),
    State(component_id = 'contract-dates', component_property = 'value'))
def display_calls_table(clicks, date):
    if date is None:
        return[{'Contract Name':'', 'Last Trade Date': '', 'Strike':'',
                'Last Price': '','Bid': '','Ask':'','Change':'',
                '% Change':'','Volume':'','Open Interest':'',
                'Implied Volatility': ''}], [{'Contract Name':'', 
                                     'Last Trade Date': '', 
                                     'Strike':'',
                                     'Last Price': '',
                                     'Bid': '',
                                     'Ask':'',
                                     'Change':'',
                                     '% Change':'',
                                     'Volume':'',
                                     'Open Interest':'',
                                     'Implied Volatility': ''}]
    else:        
        return options.get_options_chain(examined_stock.ticker,date)['calls'].to_dict('records'), options.get_options_chain(examined_stock.ticker,date)['puts'].to_dict('records')


#-----------------------------------------------------------------------------------
#
#function: display candlestick historical graph, will show at most 10 years prior
#
#initiation: press 'display candlestick' button AFTER ticker ahs been selected
#   use 'previous date' selection to limit further limit display
#-----------------------------------------------------------------------------------

@app.callback(
    [Output(component_id = 'historical-candlestick-graph', component_property = 'figure'),
     Output(component_id = 'macd-overlay-graph', component_property = 'figure')],
    [Input(component_id = 'display-candle', component_property = 'n_clicks')],
    [State(component_id = 'prev-date-select', component_property = 'value')])

def update_candlestick(click, date):
    #add moving average plots, length 12 and 26 as traces to historical candlestick graph
    #add macd (12 - 26) and exp mov avg of macd (9) to macd overlay graph, will have 2 lines + 1 bar graph
    #for moving averages, use closing price
    df = examined_stock.historical
    yrsbefore = datetime.datetime.today() - datetime.timedelta(days = 365*10)
    
    if len(np.shape(df))>0:
        fig = make_subplots(specs=[[{"secondary_y": True}]], subplot_titles = ['Trends'], x_title = 'Date')
        macd_fig = make_subplots(specs=[[{"secondary_y": True}]], subplot_titles = ['MACD'], x_title = 'Date')
        dates = df.index.to_list()#df[i for i in df.to_dict()['open']]
        earliest = dates[0].to_pydatetime()
        latest = max(yrsbefore, earliest)
        google_interest = stock_options.daily_historical_trends(examined_stock.ticker)
        trends_trace1 = go.Scatter(x = google_interest.index,y = google_interest[examined_stock.ticker].to_numpy(), 
                                  mode = 'lines', showlegend = True, name = 'Searches for {}'.format(examined_stock.ticker),
                                  marker = dict(color = 'Teal'))
        trends_trace2 = go.Scatter(x = google_interest.index,y = google_interest[examined_stock.ticker + ' news'].to_numpy(), 
                                  mode = 'lines', showlegend = True, name = 'Searches for {} news'.format(examined_stock.ticker),
                                  marker = dict(color = 'Teal'))
        trends_trace3 = go.Scatter(x = google_interest.index,y = google_interest[examined_stock.ticker + ' stock'].to_numpy(), 
                                  mode = 'lines', showlegend = True, name = 'Searches for {} stock'.format(examined_stock.ticker),
                                  marker = dict(color = 'Teal'))
        trends_trace_total = go.Scatter(x = google_interest.index,y = google_interest[examined_stock.ticker].to_numpy() + google_interest[examined_stock.ticker + ' news'].to_numpy() + google_interest[examined_stock.ticker + ' stock'].to_numpy(), 
                                  mode = 'lines', showlegend = True, name = 'Total searches for {}/news/stock'.format(examined_stock.ticker),
                                  marker = dict(color = 'Teal'))

        if latest ==earliest:
            trace1 = go.Candlestick(x= dates, 
                                    open = df['open'],
                                    high = df['high'],
                                    low = df['low'],
                                    close = df['close'],
                                    name = 'Historical Candlestick')
            seq = df['close']
            seq_x = dates
        else:
            latest = datetime.datetime(year = latest.year, month = latest.month, day = latest.day)
            while latest not in dates:
                latest = latest + datetime.timedelta(days = 1)
            ind = dates.index(latest)
            trace1 = go.Candlestick(x= dates[ind:], 
                                    open = df['open'][ind:],
                                    high = df['high'][ind:],
                                    low = df['low'][ind:],
                                    close = df['close'][ind:],
                                    name = 'Historical Candlestick')
            seq = df['close'][ind:]
            seq_x = dates[ind:]
        
        m1 = stock_options.moving_average(list(seq), 'exponential',12)
        trace_12 = go.Scatter(x = seq_x[12:], y = m1, mode = 'lines', showlegend = True, name = '12 day exp moving average')
        m2 = stock_options.moving_average(list(seq), 'exponential',26)
        trace_26 = go.Scatter(x = seq_x[26:], y = m2, mode = 'lines', showlegend = True, name = '26 day exp moving average')
        fig.add_trace(trace_12, secondary_y = False)
        fig.add_trace(trace_26, secondary_y = False)
        #overlay m1, m2 onto candlestick (2 additional traces)
        macd = np.array(m1[14:])-np.array(m2)
        trace_macd = go.Scatter(x = seq_x[26:], y = macd, mode = 'lines', showlegend = True, name = 'MACD')
        m3 = stock_options.moving_average(list(macd), 'exponential',9)
        trace_macd_signal = go.Scatter(x = seq_x[35:], y = m3, mode = 'lines', showlegend = True, name = 'MACD Signal')
        trace_macd_diff = go.Bar(x = seq_x[35:], y = macd[8:] - m3 ,showlegend = True, name = 'MACD Histogram', marker_color = 'black')
        
        macd_fig.add_trace(trace_macd)
        macd_fig.add_trace(trace_macd_signal)
        macd_fig.add_trace(trace_macd_diff)
        #overlay macd, m3, and macd-m3 onto macd plot (3 total traces)
        #mac-m3 will be a bar graph
        
        fig.add_trace(trace1,secondary_y=False,)
        fig.add_trace(trends_trace1,secondary_y = True)
        fig.add_trace(trends_trace2,secondary_y = True)
        fig.add_trace(trends_trace3,secondary_y = True)
        fig.add_trace(trends_trace_total, secondary_y = True)
        
        fig.update_yaxes(title_text="$", secondary_y=False)
        fig.update_yaxes(title_text="Searches", secondary_y=True)
    else:
        fig = go.Figure(data=[go.Candlestick(x= [0], 
                                             open =[0],
                                             high = [0],
                                             low = [0],
                                             close = [0])], 
                        layout= {'title': 'Input ticker then click update'},)
        macd_fig = go.Figure(data = [],layout = {'title': 'No MACD to Display'})
        
    return fig,macd_fig

#-----------------------------------------------------------------------------------
#
#function: begin tracking live price; tracks every 1 second, displays at most 600 seconds
#
#initiation: n/a
#
#-----------------------------------------------------------------------------------

@app.callback(
    Output(component_id = 'live-price-graph', component_property = 'figure'),
    [Input(component_id = 'interval-component', component_property = 'n_intervals')],)

def update_live_price(interval):
    #updated every 1 second;
    #do we only want to look at previous... 60*10= 600 seconds?
    #implement kalman filter projection of live price
    examined_stock.update_price()
    p = examined_stock.prices_today
    t = examined_stock.todays_times
    if len(p)>600:
        p = p[-600:]
        t = t[-600:]
    fig = go.Figure(data=[go.Scatter(x= t, y=p)], 
                    layout= {'title': 'Today\'s Price'}
            )

    return fig
            
#-----------------------------------------------------------------------------------
#
#function: display ticker selected; will internally update stock tracked to be the ticker 
#
#initiation: press 'update ticker info' button
#
#-----------------------------------------------------------------------------------

@app.callback(
    [Output(component_id = 'contract-dates', component_property = 'options'),
     Output(component_id = 'analysis1-contract-date-selection', component_property = 'options'),
    Output(component_id='ticker-output', component_property='children')] ,
    [Input(component_id ='update-ticker-info', component_property = 'n_clicks'),],
    [State(component_id='ticker-input', component_property='value'),
     State(component_id='rate-input', component_property='value'),]
)
def update_output_div(n_clicks, i1, i2):
    if i1!='':
        output = i1.upper()+' with interest rate {}%'.format(i2)
    else:
        output = 'Select Ticker'
    examined_stock.update(i1, i2)
    dates = [{'label': i, 'value': i} for i in examined_stock.option_dates]
    return dates,dates,output


#-----------------------------------------------------------------------------------
#
#function: fill stock info tables 
#
#initiation: press 'show ticker info' button AFTER selecting stock ticker and updating internal system
#
#-----------------------------------------------------------------------------------

@app.callback( Output(component_id = 'stockoverview-table', component_property = 'data'),
              Input(component_id = 'fill-ticker-table-info', component_property = 'n_clicks'),
              State(component_id = 'ticker-input', component_property = 'value'))
def update_general_info_table(tick, col):
    return examined_stock.stats
                

if __name__ == '__main__':
    app.run_server(debug=True)
    