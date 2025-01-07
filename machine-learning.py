import numpy as np
import pandas as pd
from datetime import datetime
from riskoptima import RiskOptima

import warnings
warnings.simplefilter("always", FutureWarning)

"""
 Author: Jordi Corbilla

 Date: 05/01/2025

 Project Description:
   This project demonstrates how to optimize a portfolio through machine learning and mean-variance optimization.
   It integrates predictive modeling to enhance traditional portfolio optimization techniques, combining modern
   statistical methods with robust investment strategies. The goal is to achieve improved risk-adjusted returns
   through dynamic optimization processes.

 Important Note:
   Past performance does not guarantee future results. This project is intended for educational purposes only
   and should not be considered as financial advice. Always consult with a financial advisor before making
   investment decisions.
"""

def fetch_portfolio_data(model_type):
    # Define portfolio assets and their corresponding weights (multiplied by 100k for dollar amounts)
    assets = ['MO', 'NWN', 'BKH', 'ED', 'PEP', 'NFG', 'KO', 'FRT', 'GPC', 'MSEX']
    my_current_weights = np.array([
        0.04, 
        0.14, 
        0.01, 
        0.01, 
        0.09, 
        0.16, 
        0.06, 
        0.28, 
        0.16, 
        0.05
    ])
    portfolio = {asset: weight * 100000 for asset, weight in zip(assets, my_current_weights)}
    
    # Define hypothetical market caps for the assets
    market_caps = {
        'MO': 110.0e9,   # Altria Group Inc.
        'NWN': 1.8e9,    # Northwest Natural Gas
        'BKH': 4.5e9,    # Black Hills Corp.
        'ED': 30.0e9,    # Con Edison
        'PEP': 255.0e9,  # PepsiCo Inc.
        'NFG': 5.6e9,    # National Fuel Gas
        'KO': 275.0e9,   # Coca-Cola Company
        'FRT': 9.8e9,    # Federal Realty Inv. Trust
        'GPC': 25.3e9,   # Genuine Parts Co.
        'MSEX': 2.4e9    # Middlesex Water Co.
    }
    
    # Define labels for each asset
    my_current_labels = np.array([
        'Altria Group Inc.', 
        'Northwest Natural Gas', 
        'Black Hills Corp.', 
        'Con Edison', 
        'PepsiCo Inc.', 
        'National Fuel Gas', 
        'Coca-Cola Company', 
        'Federal Realty Inv. Trust', 
        'Genuine Parts Co.', 
        'Middlesex Water Co.'
    ])
    
    # Column descriptions for the table
    column_descriptions = [
        "Security", 
        "Original\nPortfolio Weights", 
        "Mean-Variance\nOptimization", 
        f"{model_type} & Mean-Variance\nOptimization"
    ]
    
    # Define column colors for the chart table
    column_colors = [
        "white",          # For the first column (index column)
        "#e69a9a",        # Red for the unoptimized portfolio
        "#e6b89a",        # Orange for Mean-Variance Optimization        
        "#9ac7e6",        # Blue for Linear Regression & Mean-Variance Optimization
        "#9ae69b",        # Green for the benchmark (S&P 500)
    ]
    
    return portfolio, market_caps, my_current_labels, column_descriptions, column_colors

def main():
    model_type = 'Linear Regression'
    portfolio, market_caps, my_current_labels, column_descriptions, column_colors = fetch_portfolio_data(model_type)
    
    market_benchmark = ['SPY']
    
    training_start_date = '2022-01-01'
    training_end_date = '2023-11-27'
    backtesting_start_date = training_end_date
    backtesting_end_date = RiskOptima.get_previous_working_day()
    risk_free_rate = 0.05
    
    # conservative parameters, adjust as needed 
    max_volatility = 0.15
    min_weight, max_weight = 0.03, 0.2
    
    
    # Perform Mean-Variance Optimization
    tickers, weights = RiskOptima.calculate_portfolio_allocation(portfolio)
    optimized_weights_mv = RiskOptima.perform_mean_variance_optimization(tickers, training_start_date, training_end_date, max_volatility, min_allocation=min_weight, max_allocation=max_weight)
    
    # Begin ML Training on stock ticker data for Black Litterman Model
    investor_views, view_confidences = RiskOptima.generate_predictions_tickers(tickers, training_start_date, training_end_date, model_type)

    index_data = RiskOptima.fetch_historical_stock_prices(market_benchmark, training_start_date, training_end_date)
    index_return = (index_data['Close'].iloc[-1] / index_data['Close'].iloc[0]) - 1
    
    # Calculate market returns for each asset
    market_returns = RiskOptima.compute_market_returns(market_caps, index_return)
    
    historical_data = RiskOptima.fetch_historical_stock_prices(tickers, training_start_date, training_end_date)
    predicted_returns = RiskOptima.black_litterman_adjust_returns(market_returns, investor_views, view_confidences, historical_data)
    
    # Map adjusted returns to tickers
    predicted_returns = dict(zip(tickers, predicted_returns))
    
    # Convert adjusted returns to the format expected by the optimization function
    adjusted_returns_vector = np.array([predicted_returns[ticker] for ticker in tickers])
    
    # Perform mean-variance optimization with generated predicted returns
    optimized_weights_ml_mv = RiskOptima.perform_mean_variance_optimization(tickers, training_start_date, training_end_date, max_volatility, adjusted_returns_vector, min_weight, max_weight)
    
    # Download market data for backtesting and calculate performance of each asset
    historical_data_backtest = RiskOptima.fetch_historical_stock_prices(tickers, backtesting_start_date, backtesting_end_date)
    
    # Forward-fill missing values
    historical_data_filled = historical_data_backtest['Close'].ffill()
    
    # Calculate daily returns
    daily_returns_backtest = historical_data_filled.pct_change()
    
    # Calculate the cumulative performance of the machine learning mean variance optimized portfolio
    portfolio_returns_ml_mv = daily_returns_backtest.dot(optimized_weights_ml_mv)
    cumulative_returns_ml_mv = (1 + portfolio_returns_ml_mv).cumprod()
    
    # Calculate cumulative returns for the first mean variance optimized portfolio
    portfolio_returns_mv = daily_returns_backtest.dot(optimized_weights_mv)
    cumulative_returns_mv = (1 + portfolio_returns_mv).cumprod()
    
    # Download and calculate market index cumulative returns
    market_data = RiskOptima.fetch_historical_stock_prices(market_benchmark, backtesting_start_date, backtesting_end_date)['Close']
    
    market_data_filled = market_data.ffill()
    
    # Calculate daily returns
    market_returns = market_data_filled.pct_change()
    
    cumulative_market_returns = (1 + market_returns).cumprod()
    
    # Calculate cumulative returns for the unoptimized original portfolio
    portfolio_returns_unoptimized = daily_returns_backtest.dot(weights)
    cumulative_returns_unoptimized = (1 + portfolio_returns_unoptimized).cumprod()
    
    # Convert weights to percentages with 2 decimal places for formatting
    weights_pct = [f'{weight * 100:.2f}%' for weight in weights]
    optimized_weights_pct = [f'{weight * 100:.2f}%' for weight in optimized_weights_mv]
    optimized_weights_with_adjusted_returns_pct = [f'{weight * 100:.2f}%' for weight in optimized_weights_ml_mv]
    
    # Create a DataFrame and output it to show comparison between portfolio weights
    portfolio_comparison = pd.DataFrame(
        {'Original': weights_pct,
         'MV Optimization': optimized_weights_pct, 
         'ML & MV Optimization': optimized_weights_with_adjusted_returns_pct
         }, index=tickers)
    
    ax, plt, colors = RiskOptima.setup_chart_aesthetics(backtesting_start_date, backtesting_end_date)
    
    portfolio_comparison.index = my_current_labels
    RiskOptima.add_table_to_plot(ax, portfolio_comparison, column_descriptions, x=1.02, y=0.52)
    
    # Convert cumulative returns to percentage gain
    cumulative_returns_ml_mv_percent = (cumulative_returns_ml_mv - 1) * 100
    cumulative_returns_mv_percent = (cumulative_returns_mv - 1) * 100
    cumulative_returns_unoptimized_percent = (cumulative_returns_unoptimized - 1) * 100
    cumulative_market_returns_percent = (cumulative_market_returns - 1) * 100
    
    final_returns_ml_mv = cumulative_returns_ml_mv_percent.iloc[-1]
    final_returns_mv = cumulative_returns_mv_percent.iloc[-1]
    final_returns_unoptimized = cumulative_returns_unoptimized_percent.iloc[-1]
    final_returns_market = cumulative_market_returns_percent.iloc[-1]
    
    if isinstance(final_returns_market, pd.Series):
        final_returns_market = final_returns_market.iloc[0]
    
    ax.plot(cumulative_returns_unoptimized_percent, label='Original Unoptimized Portfolio', color=column_colors[1])
    ax.plot(cumulative_returns_mv_percent, label='Portfolio Optimized with Mean-Variance', color=column_colors[2])
    ax.plot(cumulative_market_returns_percent, label='Market Index Benchmark (S&P 500)', color=column_colors[3])
    ax.plot(cumulative_returns_ml_mv_percent, label=f'Portfolio Optimized with {model_type} and Mean-Variance', color=column_colors[4])
    
    RiskOptima.plot_performance_metrics(model_type, portfolio_returns_unoptimized, portfolio_returns_mv, portfolio_returns_ml_mv, 
                                 market_returns, risk_free_rate, final_returns_unoptimized, final_returns_mv, final_returns_ml_mv, 
                                 final_returns_market, ax, column_colors)
    
    RiskOptima.add_ratio_explanation(ax, x=1.02, y=0.01, fontsize=9)

    plot_title = "Portfolio Optimization and Benchmarking: Comparing Machine Learning and Statistical Models for Risk-Adjusted Performance"

    plt.title(plot_title, fontsize=14, pad=20)
    plt.xlabel('Date')
    plt.ylabel('Percentage Gain (%)')
    plt.legend(loc='lower center')
    plt.grid(True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"machine_learning_optimization_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    main()