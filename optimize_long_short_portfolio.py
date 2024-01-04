import pandas as pd
import numpy as np
import logging
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




def main():

    # Get holdings and their direction (L/S) in a dataframe from an Excel file in the work directory and sort by Tickers
    logging.info("Reading portfolio holdings...")
    df_holdings = pd.read_excel(f'config/Portfolio_Details.xlsx', sheet_name='Portfolio', header=0).sort_values(by='Tickers')

    # Get Avg Volume and Sector for the holdings
    list_holdings = df_holdings['Tickers'].tolist()
    list_5_pct_avg_daily_volume = get_5_pct_avg_daily_volume(list_holdings)
    df_holdings['Volume'] = list_5_pct_avg_daily_volume
    list_sectors = get_sectors(list_holdings)
    df_holdings['Sector'] = list_sectors

    # Get sector exposures for the S&P500 and portfolio holdings
    # snp500_tickers = get_sp500_tickers()
    # dict_snp500_sector_exposures = get_sector_exposures(snp500_tickers)
    dict_snp500_sector_exposures = {'Industrials': 0.08, 'Healthcare': 0.12, 'Technology': 0.28, 'Utilities': 0.02, 'Financial Services': 0.11, 'Basic Materials': 0.02, 'Consumer Cyclical': 0.11, 'Real Estate': 0.02, 'Communication Services': 0.13, 'Consumer Defensive': 0.06, 'Energy': 0.04}

    # Get holdings and their direction (L/S) in a dataframe from an Excel file in the work directory and sort by Tickers
    df_portfolio_thresholds = pd.read_excel(f'config/Portfolio_Details.xlsx', sheet_name='Thresholds', header=0)
    logging.info("Portfolio thresholds:")
    logging.info(f"\n{df_portfolio_thresholds.to_string(index=False)}")

    # create a dictionary of portfolio thresholds with Restriction_Type as key and Value as value
    dict_portfolio_thresholds = dict(zip(df_portfolio_thresholds['Restriction_Type'], df_portfolio_thresholds['Value']))

    # Get the price history dataframe
    price_history_days = dict_portfolio_thresholds['price_history_days']
    df_price_history = get_stocks_price_history(list_holdings, price_history_days)

    # Calculate the optimal portfolio
    calculate_optimal_portfolio(df_price_history, df_holdings, dict_snp500_sector_exposures, dict_portfolio_thresholds)


#################################### Portfolio Optimization Functions ####################################


# Function to calculate the portfolio volatility given the weights, returns, and number of trading days
def calculate_portfolio_volatility(list_weights, df_cov_matrix_scaled):
    # Calculate the weighted covariance matrix
    df_weighted_cov_matrix = np.dot(df_cov_matrix_scaled, list_weights)
    # Calculate the portfolio variance
    float_portfolio_variance = np.dot(list_weights.T, df_weighted_cov_matrix)
    # Calculate the portfolio volatility as the square root of the variance
    float_portfolio_volatility = np.sqrt(float_portfolio_variance)
    return float_portfolio_volatility


# Function to initialize arrays to store the weights, returns, volatilities, and Sharpe ratios for each portfolio
def initialize_arrays(iterations, num_assets):
    list_all_weights = np.zeros((iterations, num_assets))
    list_return_array = np.zeros(iterations)
    list_volatility_array = np.zeros(iterations)
    list_sharpe_array = np.zeros(iterations)
    return list_all_weights, list_return_array, list_volatility_array, list_sharpe_array


# Plot a scatter plot of the portfolio returns vs. volatility and plot the efficient frontier and the optimal portfolio and show weights on hovering over the points
def plot_portfolio_scatter(list_volatility_array, list_return_array, list_sharpe_array, max_sharpe_ind):
    logging.info("Plotting efficient frontier and optimal portfolio...")
    plt.figure(figsize=(12, 8))
    plt.scatter(list_volatility_array, list_return_array, c=list_sharpe_array, cmap='viridis')
    plt.colorbar(label='Sharpe ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.scatter(list_volatility_array[max_sharpe_ind], list_return_array[max_sharpe_ind], c='red', s=50) # red dot
    plt.savefig('output/scatter_plot.png')  # Save the scatter plot as an image
    plt.close()  # Close the plot to release memory
    logging.info(f"Scatter plot saved to - 'output/scatter_plot.png'")


def store_optimal_portfolio_details(list_holdings, list_direction, optimal_weights, list_total_returns, optimal_portfolio_dtc, max_no_of_days_to_cover, optimal_portfolio_return, list_portfolio_sectors, list_optimal_portfolio_sector_exposures, dict_snp500_sector_exposures, max_sharpe_ratio, max_length):
    # Create a dataframe to store the optimal portfolio details
    df_optimal_portfolio = pd.DataFrame({
        'Tickers': list_holdings + [''] * (max_length - len(list_holdings)),
        'Direction': list_direction + [''] * (max_length - len(list_direction)),
        'Optimal Weights': optimal_weights.tolist() + [''] * (max_length - len(optimal_weights)),
        'Total Returns': list_total_returns + [''] * (max_length - len(list_total_returns)),
        'Days to Cover': optimal_portfolio_dtc + [''] * (max_length - len(optimal_portfolio_dtc)),
        'Max Days to Cover': [max_no_of_days_to_cover] * max_length,
        'Portfolio Sectors': list_portfolio_sectors + [''] * (max_length - len(list_portfolio_sectors)),
        'Portfolio Sector Exposures': list_optimal_portfolio_sector_exposures + [''] * (max_length - len(list_optimal_portfolio_sector_exposures)),
        'S&P500 Sector Exposures': list(dict_snp500_sector_exposures.values()) + [''] * (max_length - len(dict_snp500_sector_exposures)),
        'Max Sharpe Ratio': [max_sharpe_ratio] + [''] * (max_length - 1),
        'Optimal Portfolio Return': [optimal_portfolio_return] + [''] * (max_length - 1)
    })

    # Save the optimal portfolio details to an Excel file
    df_optimal_portfolio.to_excel(f'output/Optimal_Portfolio_Details.xlsx', sheet_name='Sheet1', index=False)
    logging.info(f"Optimal portfolio details saved to - 'output/Optimal_Portfolio_Details.xlsx'")
    return df_optimal_portfolio


def read_portfolio_thresholds(dict_portfolio_thresholds):
    iterations = int(dict_portfolio_thresholds['portfolio_iterations'])
    log_every_n = int(dict_portfolio_thresholds['log_every_n'])
    weight_limit = dict_portfolio_thresholds['single_stock_weight_limit'] # not more than this value either in long or short direction
    port_net_weight_lower_limit = dict_portfolio_thresholds['port_net_weight_lower_limit'] # not less than this value
    port_net_weight_upper_limit = dict_portfolio_thresholds['port_net_weight_upper_limit'] # not more than this value
    net_weight_range = (port_net_weight_lower_limit, port_net_weight_upper_limit)
    short_weight_limit = dict_portfolio_thresholds['port_short_weight_limit'] # not less than this value
    long_weight_limit = dict_portfolio_thresholds['port_long_weight_limit'] # not less than this value
    sector_exposure_threshold = dict_portfolio_thresholds['sector_exposure_threshold'] # does not variate from benchmark sector exposure by more than this value in either direction
    start_of_period_portfolio_value = dict_portfolio_thresholds['start_of_period_portfolio_value'] # start of period portfolio value
    max_no_of_days_to_cover = int(dict_portfolio_thresholds['max_no_of_days_to_cover']) # Set the maximum number of days to cover/sell a position
    return iterations, log_every_n, weight_limit, net_weight_range, short_weight_limit, long_weight_limit, sector_exposure_threshold, start_of_period_portfolio_value, max_no_of_days_to_cover


def calculate_total_returns(df_price_history):
    # Calculate the total returns for each asset
    df_returns = df_price_history.pct_change()
    df_total_returns = (1 + df_returns).prod() - 1
    list_total_returns = df_total_returns.tolist()
    return list_total_returns, df_returns


def covariance_matrix(df_returns):
    # Calculate the covariance matrix of the returns
    df_cov_matrix = df_returns.cov()
    # Scale the covariance matrix by the number of trading days
    df_cov_matrix_scaled = df_cov_matrix * len(df_returns)
    return df_cov_matrix_scaled


# Function to calculate the optimal portfolio given the price history, number of iterations, and weight limit
def calculate_optimal_portfolio(df_price_history, df_holdings, dict_snp500_sector_exposures, dict_portfolio_thresholds):
    
    logging.info("")
    logging.info("--------------Begin: calculate_optimal_portfolio--------------")
    logging.info("")
    logging.info("Reading portfolio thresholds...")
    iterations, log_every_n, weight_limit, net_weight_range, short_weight_limit, long_weight_limit, sector_exposure_threshold, start_of_period_portfolio_value, max_no_of_days_to_cover = read_portfolio_thresholds(dict_portfolio_thresholds)    

    # Get the price of all holdings at the start of the period
    portfolio_start_holdings_value = df_price_history.iloc[0]
    df_portfolio_start_holdings_value = portfolio_start_holdings_value.rename('Price').to_frame()

    # Drop columns with all NaN values (where market data was not found for the ticker)
    df_price_history.dropna(axis=1, how='all', inplace=True)
    list_holdings = df_price_history.columns.tolist()

    # Merge holdings with price history to get the direction of each holding
    merged_df = pd.merge(df_holdings, pd.DataFrame(list_holdings, columns=['Tickers']), on='Tickers', how='inner')
    list_direction = np.array(merged_df['Direction'].tolist())

    # Calculate the total returns for each asset
    list_total_returns, df_returns = calculate_total_returns(df_price_history)

    num_assets, num_trading_days = len(list_holdings), len(df_price_history)
    logging.info(f"Number of assets: {num_assets}")
    logging.info(f"Number of trading days: {num_trading_days}")

    # Calculate the covariance matrix of the returns
    df_cov_matrix_scaled = covariance_matrix(df_returns)

    # Initialize arrays to store weights, returns, volatilities, and Sharpe ratios
    list_all_weights, list_return_array, list_volatility_array, list_sharpe_array = initialize_arrays(iterations, num_assets)

    # Create a list to store the sector exposures of the portfolio
    list_sector_exposures = list(range(iterations))

    # Create a list to store the days to cover/sell of the portfolio
    list_days_to_cover = list(range(iterations))

    portfolio_count = 0
    weights_iteration = 0
    breached_sector_threshold = 0
    breached_days_to_cover_threshold = 0

    # Generate random portfolios until the desired number of iterations is reached
    logging.info("")
    logging.info("-------Start: Portfolio Generation Simulations-------")
    logging.info("")

    while portfolio_count < iterations:
        # Increment weights_iteration counter
        weights_iteration += 1
        
        # Generate random weights for the assets
        list_weights = np.array(np.random.random(num_assets))
        
        # Apply negative sign to elements of list_weights where list_direction is 'S'
        list_weights[list_direction == 'S'] *= -1
        
        # Normalize the weights to ensure they sum up to 1
        normalized_weights = list_weights / np.sum(list_weights)
        
        # Generate a random number between the net weight range
        random_range = np.random.uniform(*net_weight_range)
        
        # Scale the weights to the random range
        list_weights = normalized_weights * random_range
        
        # Skip the current iteration if any weight exceeds the different weight limits
        if (np.any(list_weights > weight_limit)) or (np.any(list_weights < -weight_limit)) or \
        (np.sum(list_weights[list_weights < 0]) < short_weight_limit) or \
        (np.sum(list_weights[list_weights > 0]) < long_weight_limit):
            continue
        
        # Skip the current iteration if holding sector exposure exceeds the threshold with respect to the S&P500 sector exposure
        df_holdings_weights = pd.DataFrame(list(zip(list_holdings, list_weights)), columns=['Holdings', 'Weights'])
        df_holdings_weights = pd.merge(df_holdings_weights, df_holdings, left_on='Holdings', right_on='Tickers', how='inner')
        df_sector_weights = df_holdings_weights.groupby(['Sector']).sum()
        dict_holdings_sector_weights = df_sector_weights['Weights'].to_dict()
        
        try:
            for sector in dict_holdings_sector_weights:
                if sector in dict_snp500_sector_exposures:
                    if (dict_holdings_sector_weights[sector] > dict_snp500_sector_exposures[sector] + sector_exposure_threshold) or \
                    (dict_holdings_sector_weights[sector] < dict_snp500_sector_exposures[sector] - sector_exposure_threshold):
                        breached_sector_threshold += 1
                        raise Exception(f'Sector exposure threshold breached for sector: {sector}')
        except Exception:
            continue
        
        # Skip the current iteration if days to cover/sell exceeds the maximum number of days to cover/sell a position
        df_holdings_weights['Position Value'] = df_holdings_weights['Weights'] * start_of_period_portfolio_value
        df_holdings_weights = df_holdings_weights.merge(df_portfolio_start_holdings_value, left_on='Holdings', right_index=True, how='inner')
        df_holdings_weights['Shares'] = df_holdings_weights['Position Value'] / df_holdings_weights['Price']
        df_holdings_weights['Days to Cover'] = abs(df_holdings_weights['Shares']) / df_holdings_weights['Volume']

        if np.any(df_holdings_weights['Days to Cover'] > max_no_of_days_to_cover):
            breached_days_to_cover_threshold += 1
            continue
        
        # Store the weights, returns, volatilities, sector exposures, and days to cover/sell in the arrays
        list_all_weights[portfolio_count, :] = list_weights
        list_return_array[portfolio_count] = np.sum(list_weights * list_total_returns)
        list_volatility_array[portfolio_count] = calculate_portfolio_volatility(list_weights, df_cov_matrix_scaled)
        list_sector_exposures[portfolio_count] = dict_holdings_sector_weights
        list_days_to_cover[portfolio_count] = df_holdings_weights['Days to Cover'].to_list()
        list_sharpe_array[portfolio_count] = list_return_array[portfolio_count] / list_volatility_array[portfolio_count]
        
        portfolio_count += 1
        
        if portfolio_count % log_every_n == 0:
            logging.info(f"# Portfolio Iteration No: {portfolio_count}")
            logging.info(f"  Weights Iteration No: {weights_iteration}")
            logging.info(f"  Breached Sector Exposure Threshold No: {breached_sector_threshold}")
            logging.info(f"  Breached Days to Cover Threshold No: {breached_days_to_cover_threshold}")

    logging.info("")
    logging.info("-------End: Portfolio Generation Simulations-------")
    logging.info("")

    # Find optimal portfolio details
    max_sharpe_ratio = np.max(list_sharpe_array)
    max_sharpe_ind = np.argmax(list_sharpe_array)
    optimal_weights = list_all_weights[max_sharpe_ind, :]
    optimal_portfolio_return = list_return_array[max_sharpe_ind]
    optimal_portfolio_sector_exposures = list_sector_exposures[max_sharpe_ind]
    optimal_portfolio_dtc = list_days_to_cover[max_sharpe_ind]

    max_length = max(len(list_holdings), len(dict_snp500_sector_exposures))
    list_portfolio_sectors = list(dict_snp500_sector_exposures.keys())
    list_optimal_portfolio_sector_exposures = [optimal_portfolio_sector_exposures.get(sector, 0) for sector in list_portfolio_sectors]
    list_direction = merged_df['Direction'].tolist()

    # Store optimal portfolio details in a dataframe and save to an Excel file
    store_optimal_portfolio_details(list_holdings, list_direction, optimal_weights, list_total_returns, optimal_portfolio_dtc, max_no_of_days_to_cover, optimal_portfolio_return, list_portfolio_sectors, list_optimal_portfolio_sector_exposures, dict_snp500_sector_exposures, max_sharpe_ratio, max_length)

    # Call the plot_portfolio_scatter function to plot the efficient frontier and the optimal portfolio
    plot_portfolio_scatter(list_volatility_array, list_return_array, list_sharpe_array, max_sharpe_ind)

    logging.info("")
    logging.info("--------------End: calculate_optimal_portfolio--------------")


###################################### Market Data Functions ######################################

# Call get_closing_prices to get DataFrame with last one year data for entered stock list
def get_stocks_price_history(list_tickers, days):
    """
    Retrieves the historical closing prices for a list of tickers for the specified number of days.

    Args:
        list_tickers (list): List of stock tickers.
        days (int): Number of days to retrieve the historical data for.

    Returns:
        pandas.DataFrame: DataFrame containing the historical closing prices for the specified tickers.
    """
    logging.info("Getting price history for all portfolio holdings...")
    # Calculate the start and end dates for retrieving the historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Download the historical data using yfinance library
    data = yf.download(list_tickers, start=start_date, end=end_date)["Close"]

    return data


def get_sp500_tickers():
    """
    Retrieves the list of tickers for all S&P500 stocks.

    Returns:
        list: List of tickers for all S&P500 stocks.
    """
    logging.info("Getting S&P500 tickers...")
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return sp500_tickers['Symbol'].head(500).tolist()


def get_nasdaq100_tickers():
    """
    Retrieves the list of tickers for all NASDAQ100 stocks.

    Returns:
        list: List of tickers for all NASDAQ100 stocks.
    """
    nasdaq100_tickers = pd.read_html('https://en.wikipedia.org/wiki/NASDAQ-100')[4]
    return nasdaq100_tickers['Ticker'].head(100).tolist()
    

def get_5_pct_avg_daily_volume(tickers):
    """
    Retrieves 5% of the average daily volume for a list of tickers using the yfinance library.

    Args:
        tickers (list): List of stock tickers.

    Returns:
        list: List of 5% of the average daily volumes for the specified tickers.
    """
    logging.info("Getting 5% of the average daily volumes for all portfolio holdings...")
    avg_daily_volumes = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        avg_daily_volume = round(stock.info.get("averageDailyVolume10Day") * 0.05)
        avg_daily_volumes.append(avg_daily_volume)
    return avg_daily_volumes


def get_sectors(tickers):
    """
    Retrieves the sectors for a list of tickers using the yfinance library.

    Args:
        tickers (list): List of stock tickers.

    Returns:
        list: List of sectors for the specified tickers.
    """
    logging.info("Getting sectors for all portfolio holdings...")
    sectors = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        sector = stock.info["sector"]
        sectors.append(sector)
    return sectors


# get sector exposures by using market cap to calculate percentage for S&P500 index using yfinance library
def get_sector_exposures(tickers):
    """
    Retrieves the sector exposures for S&P500 tickers using the yfinance library.

    Returns:
        dict: Dictionary containing the sector exposures for the specified tickers.
    """
    logging.info("Getting sector exposures for all tickers...")
    sector_exposures = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            sector = stock.info["sector"]
            market_cap = stock.info["marketCap"]
            if sector in sector_exposures:
                sector_exposures[sector] += market_cap
            else:
                sector_exposures[sector] = market_cap
        except:
            continue
    total_market_cap = sum(sector_exposures.values())
    for sector in sector_exposures:
        sector_exposures[sector] = round(sector_exposures[sector] / total_market_cap, 2)
    return sector_exposures


if __name__ == "__main__":
    main()