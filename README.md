### LONG-SHORT PORTFOLIO OPTIMIZATION
-------------------------------------------------------------------------

### What is the Project all about?

This project is a Python application that optimizes a long-short portfolio. The portfolio details are read from an Excel file, `config/Portfolio_Details.xlsx`, which contains the holdings and their direction (Long/Short). This file also contains practical restrictions/thresholds used while constructing a portfolio.

The application calculates the portfolio volatility given the tickers, direction and number of trading days. The application also calculates the total returns for each asset and the covariance matrix of the returns. It then calculates the optimal portfolio given the portfolio restrictions/thresholds.

The optimal portfolio details are stored in a DataFrame and saved to an Excel file, `output/Optimal_Portfolio_Details.xlsx`. The details include the holdings, direction, optimal weights, total returns, days to cover, portfolio return, portfolio sectors, portfolio sector exposures, S&P500 sector exposures, max Sharpe ratio, and optimal portfolio return.

The application then plots the efficient frontier and the optimal portfolio and saves it at `output/scatter_plot.png`.

The application logs the progress of the portfolio generation simulations.

-------------------------------------------------------------------------
### Use Case

Portfolio Managers / Investors can use to assess their portfolio's performance against the best of the portfolios that could have existed given practical restrictions/thresholds that the users actually face while constructing their portfolio.

--------------------------------------------------------------------------

### How is this project different from the hundreds of other portfolio optimization project?

1. It uses actual historical returns and not mean daily returns
2. The optimization is done for Long/Short instead of the traditional Long only portfolio
3. Instead of an impractical portfolio suggestion we try to apply real world restrictions to the portfolio like
    - Days to Cover
    - Short, Long, Net Weight limits of the portfolio
    - Sector Exposures within a range of the the benchmark sector exposures
4. Uses real market data from Yahoo Finance using a python library instead of dummy prices

--------------------------------------------------------------------------

### How to Use?

1. Input the portfolio details like holdings and the different restrictions/thresholds in 
'config/Portfolio_Details.xlsx'
2. To run the application, execute the `optimize_long_short_portfolio.py` script.
3. Check the efficient frontier - `output/scatter_plot.png`
4. Check the optimal portfolio details - `output/Optimal_Portfolio_Details.xlsx`

--------------------------------------------------------------------------

### Future Scope 

1. Add more restrictions/thresholds like cap on weight of top n%, etc.
2. Make use of an actual portfolio and compare it against the optimal portfolio or top n optimal portfolios.
3. Plot different graphs for the comparison of the actual vs optimal portfolios.
4. Make the main loop that generates different weights run asyncronously to improve speed.

--------------------------------------------------------------------------
