# Stock Dash Platform

## Usage
- Input single ticker
- Change risk-free interest rate if necessary (input as %)

## Functions
- Tracks live price
- Displays stock information
- Displays historical candle-stick plot
- Displays various options available for specified date
- Calculates and displays option price, option greeks for specific contract
- Plots specified options and possible profit outcomes along with normal distribution for both buying and selling
- Profit assumption is that option will be exercised (or left alone) at/until expiration date and no earlier
- Displays expected profit margin utilizing normal distribution
- Pulls historical/daily mentions via google, overlaid onto candle-stock plot
- Calculates/ Plots PDF and CDF onto profits plot

## To-dos:
- FIX HEROKU DEPLOYMEN; currently has issue with importing numpy, fixing requirements has not helped
- Incorparate mentions into analysis
- Include individual contract analysis; detailing potential price changes with utilizing slide bars for (stock price, IV, days till expir)
- Incorporate predictive analysis on daily/ live price plot using kalman filtering (just idea), predicting using (splines or other functions)
- Utilize different contracts with same strike price but different dates to anticipate price changes, calculating % chance a the option price will be a specific value on a certain date