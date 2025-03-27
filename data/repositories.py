"""
Data repository classes for Stock Prediction App
"""
import logging
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.fetchers import fetch_historical_data, get_stock_list, fetch_stock_news

class StockDataRepository:
    """Repository for accessing stock data with abstracted data sources"""
    
    def __init__(self):
        """Initialize the repository"""
        logging.info("Initializing StockDataRepository")
        self.cache = {}
    
    def get_stocks(self) -> List[Dict[str, str]]:
        """Get list of available stocks"""
        return get_stock_list()
    
    def get_historical_data(self, ticker: str, start_date: Union[datetime, str], end_date: Union[datetime, str]) -> pd.DataFrame:
        """Get historical data for a stock"""
        return fetch_historical_data(ticker, start_date, end_date)
    
    def get_news(self, ticker: str) -> List[Dict[str, str]]:
        """Get news for a stock"""
        return fetch_stock_news(ticker)
    
    def get_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a dataframe
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators added
        """
        df = df.copy()
        
        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        std_dev = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (std_dev * 2)
        df['BB_Lower'] = df['BB_Middle'] - (std_dev * 2)
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume Moving Average
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        return df
    
    def get_correlation_matrix(self, tickers: List[str], days: int = 90) -> Optional[pd.DataFrame]:
        """
        Calculate correlation matrix for a list of stocks
        
        Args:
            tickers (List[str]): List of ticker symbols
            days (int, optional): Number of days for historical data. Defaults to 90.
            
        Returns:
            Optional[pd.DataFrame]: Correlation matrix or None if error
        """
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            # Get historical data for each ticker
            close_prices = pd.DataFrame()
            
            for ticker in tickers:
                try:
                    df = self.get_historical_data(ticker, start_date, end_date)
                    if df is not None and not df.empty and 'Close' in df.columns:
                        close_prices[ticker] = df['Close']
                except Exception as e:
                    logging.error(f"Error getting data for {ticker}: {e}")
            
            # Calculate correlation matrix
            if not close_prices.empty:
                corr_matrix = close_prices.corr()
                return corr_matrix
            
            return None
        except Exception as e:
            logging.error(f"Error calculating correlation matrix: {e}")
            return None
    
    def detect_anomalies(self, df: pd.DataFrame, window: int = 20, threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect anomalies in stock price data
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            window (int, optional): Rolling window size. Defaults to 20.
            threshold (float, optional): Z-score threshold. Defaults to 2.0.
            
        Returns:
            pd.DataFrame: DataFrame with anomaly flags
        """
        df = df.copy()
        
        # Calculate rolling mean and standard deviation
        rolling_mean = df['Close'].rolling(window=window).mean()
        rolling_std = df['Close'].rolling(window=window).std()
        
        # Calculate Z-score
        df['Z_Score'] = (df['Close'] - rolling_mean) / rolling_std
        
        # Flag anomalies
        df['Is_Anomaly'] = df['Z_Score'].abs() > threshold
        
        return df

class PortfolioRepository:
    """Repository for portfolio management"""
    
    def __init__(self, stock_repo: StockDataRepository):
        """Initialize the repository"""
        logging.info("Initializing PortfolioRepository")
        self.stock_repo = stock_repo
    
    def get_portfolio_value(self, portfolio: Dict[str, Dict]) -> Tuple[float, pd.DataFrame]:
        """
        Calculate current value of portfolio
        
        Args:
            portfolio (Dict[str, Dict]): Portfolio data
            
        Returns:
            Tuple[float, pd.DataFrame]: Total value and portfolio dataframe
        """
        if not portfolio:
            return 0, pd.DataFrame()
        
        portfolio_df = pd.DataFrame()
        
        for ticker, details in portfolio.items():
            try:
                # Get the latest price
                df = self.stock_repo.get_historical_data(ticker, 
                                                        datetime.now().date() - timedelta(days=5), 
                                                        datetime.now().date())
                
                if df is not None and not df.empty:
                    latest_price = df['Close'].iloc[-1]
                    quantity = details['quantity']
                    position_value = latest_price * quantity
                    
                    # Add to dataframe
                    portfolio_df = pd.concat([portfolio_df, pd.DataFrame({
                        'Ticker': [ticker],
                        'Name': [details['name']],
                        'Quantity': [quantity],
                        'Current Price': [latest_price],
                        'Position Value': [position_value],
                        'Add Date': [details['add_date']]
                    })])
            except Exception as e:
                logging.error(f"Error getting current price for {ticker}: {e}")
                # Add with placeholder data
                portfolio_df = pd.concat([portfolio_df, pd.DataFrame({
                    'Ticker': [ticker],
                    'Name': [details['name']],
                    'Quantity': [details['quantity']],
                    'Current Price': [0],
                    'Position Value': [0],
                    'Add Date': [details['add_date']]
                })])
        
        # Calculate the total value
        if not portfolio_df.empty:
            total_value = portfolio_df['Position Value'].sum()
            return total_value, portfolio_df
        
        return 0, pd.DataFrame()
    
    def get_portfolio_performance(self, portfolio: Dict[str, Dict], days: int = 30) -> pd.DataFrame:
        """
        Get historical performance of portfolio
        
        Args:
            portfolio (Dict[str, Dict]): Portfolio data
            days (int, optional): Number of days for historical data. Defaults to 30.
            
        Returns:
            pd.DataFrame: DataFrame with portfolio performance
        """
        if not portfolio:
            return pd.DataFrame()
        
        # Get the historical data for each stock in the portfolio
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Create a dataframe with dates
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        performance_df = pd.DataFrame(index=date_range)
        
        # Add historical data for each stock
        for ticker, details in portfolio.items():
            try:
                df = self.stock_repo.get_historical_data(ticker, start_date, end_date)
                
                if df is not None and not df.empty:
                    # Get the close prices and multiply by quantity
                    df.set_index('Date', inplace=True)
                    close_prices = df['Close'] * details['quantity']
                    performance_df[ticker] = close_prices
            except Exception as e:
                logging.error(f"Error getting historical data for {ticker}: {e}")
        
        # Sum the values for each date to get the total portfolio value
        performance_df['Total'] = performance_df.sum(axis=1)
        
        return performance_df 