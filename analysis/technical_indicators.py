"""
Technical indicator calculation utilities for Stock Prediction App
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a dataframe
    
    Args:
        df (pd.DataFrame): Dataframe with stock data
        
    Returns:
        pd.DataFrame: Dataframe with added technical indicators
    """
    df = df.copy()
    
    try:
        # Moving averages
        df = add_moving_averages(df, windows=[5, 10, 20, 50, 200])
        
        # Bollinger Bands
        df = add_bollinger_bands(df)
        
        # Momentum indicators
        df = add_rsi(df)
        df = add_macd(df)
        
        # Volatility indicators
        df = add_atr(df)
        
        # Trend indicators
        df = add_adx(df)
        
        # Volume indicators
        if 'Volume' in df.columns:
            df = add_obv(df)
            df = add_volume_sma(df)
        
        return df
    
    except Exception as e:
        logging.error(f"Error adding technical indicators: {e}")
        return df

def add_moving_averages(df: pd.DataFrame, windows: List[int] = [5, 20, 50]) -> pd.DataFrame:
    """
    Add Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)
    
    Args:
        df (pd.DataFrame): Dataframe with 'Close' prices
        windows (List[int], optional): List of window sizes. Defaults to [5, 20, 50].
        
    Returns:
        pd.DataFrame: Dataframe with added moving averages
    """
    df = df.copy()
    
    try:
        for window in windows:
            # Simple Moving Average
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            
            # Exponential Moving Average
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        return df
    
    except Exception as e:
        logging.error(f"Error calculating moving averages: {e}")
        return df

def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Add Bollinger Bands
    
    Args:
        df (pd.DataFrame): Dataframe with 'Close' prices
        window (int, optional): Window size for moving average. Defaults to 20.
        num_std (float, optional): Number of standard deviations. Defaults to 2.0.
        
    Returns:
        pd.DataFrame: Dataframe with Bollinger Bands added
    """
    df = df.copy()
    
    try:
        # Calculate middle band (SMA)
        if f'SMA_{window}' not in df.columns:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Calculate standard deviation
        rolling_std = df['Close'].rolling(window=window).std()
        
        # Calculate upper and lower bands
        df['BB_Middle'] = df[f'SMA_{window}']
        df['BB_Upper'] = df['BB_Middle'] + (rolling_std * num_std)
        df['BB_Lower'] = df['BB_Middle'] - (rolling_std * num_std)
        
        # Calculate bandwidth and percent b
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_PercentB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        return df
    
    except Exception as e:
        logging.error(f"Error calculating Bollinger Bands: {e}")
        return df

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI)
    
    Args:
        df (pd.DataFrame): Dataframe with 'Close' prices
        window (int, optional): Window size for RSI calculation. Defaults to 14.
        
    Returns:
        pd.DataFrame: Dataframe with RSI added
    """
    df = df.copy()
    
    try:
        # Calculate price changes
        delta = df['Close'].diff()
        
        # Create gain and loss series
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Add RSI crossover signals
        df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
        df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
        
        return df
    
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}")
        return df

def add_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """
    Add Moving Average Convergence Divergence (MACD)
    
    Args:
        df (pd.DataFrame): Dataframe with 'Close' prices
        fast_period (int, optional): Fast EMA period. Defaults to 12.
        slow_period (int, optional): Slow EMA period. Defaults to 26.
        signal_period (int, optional): Signal line period. Defaults to 9.
        
    Returns:
        pd.DataFrame: Dataframe with MACD added
    """
    df = df.copy()
    
    try:
        # Calculate fast and slow EMAs
        fast_ema = df['Close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['Close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        df['MACD'] = fast_ema - slow_ema
        df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Add crossover signals
        df['MACD_Crossover'] = ((df['MACD'] > df['MACD_Signal']) & 
                              (df['MACD'].shift() <= df['MACD_Signal'].shift())).astype(int)
        df['MACD_Crossunder'] = ((df['MACD'] < df['MACD_Signal']) & 
                               (df['MACD'].shift() >= df['MACD_Signal'].shift())).astype(int)
        
        return df
    
    except Exception as e:
        logging.error(f"Error calculating MACD: {e}")
        return df

def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add Average True Range (ATR)
    
    Args:
        df (pd.DataFrame): Dataframe with 'High', 'Low' and 'Close' prices
        window (int, optional): Window size for ATR calculation. Defaults to 14.
        
    Returns:
        pd.DataFrame: Dataframe with ATR added
    """
    df = df.copy()
    
    try:
        # Calculate true range
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR
        df['ATR'] = true_range.rolling(window=window).mean()
        
        # Calculate Percent ATR (ATR as percentage of current price)
        df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100
        
        return df
    
    except Exception as e:
        logging.error(f"Error calculating ATR: {e}")
        return df

def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add Average Directional Index (ADX)
    
    Args:
        df (pd.DataFrame): Dataframe with 'High', 'Low' and 'Close' prices
        window (int, optional): Window size for ADX calculation. Defaults to 14.
        
    Returns:
        pd.DataFrame: Dataframe with ADX added
    """
    df = df.copy()
    
    try:
        # Calculate True Range
        if 'ATR' not in df.columns:
            df = add_atr(df, window=window)
        
        # Calculate directional movement
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff(-1).abs()
        
        # Positive DM
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        
        # Negative DM
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Calculate smoothed DM and TR
        tr = df['ATR'] * window
        plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False).mean() / tr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/window, adjust=False).mean() / tr)
        
        # Calculate directional movement index
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di
        
        # Calculate directional index
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).fillna(0)
        
        # Calculate ADX
        df['ADX'] = dx.ewm(alpha=1/window, adjust=False).mean()
        
        return df
    
    except Exception as e:
        logging.error(f"Error calculating ADX: {e}")
        return df

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add On-Balance Volume (OBV)
    
    Args:
        df (pd.DataFrame): Dataframe with 'Close' and 'Volume' data
        
    Returns:
        pd.DataFrame: Dataframe with OBV added
    """
    df = df.copy()
    
    try:
        # Check if Volume data is available
        if 'Volume' not in df.columns:
            logging.warning("Volume data not available, cannot calculate OBV")
            return df
        
        # Calculate OBV
        df['OBV'] = np.where(df['Close'] > df['Close'].shift(1),
                          df['Volume'],
                          np.where(df['Close'] < df['Close'].shift(1),
                                  -df['Volume'], 0)).cumsum()
        
        # Calculate OBV EMA
        df['OBV_EMA'] = df['OBV'].ewm(span=20, adjust=False).mean()
        
        return df
    
    except Exception as e:
        logging.error(f"Error calculating OBV: {e}")
        return df

def add_volume_sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add Volume Simple Moving Average
    
    Args:
        df (pd.DataFrame): Dataframe with 'Volume' data
        window (int, optional): Window size for Volume SMA. Defaults to 20.
        
    Returns:
        pd.DataFrame: Dataframe with Volume SMA added
    """
    df = df.copy()
    
    try:
        # Check if Volume data is available
        if 'Volume' not in df.columns:
            logging.warning("Volume data not available, cannot calculate Volume SMA")
            return df
        
        # Calculate Volume SMA
        df[f'Volume_SMA_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Calculate Volume Ratio (current volume / SMA)
        df['Volume_Ratio'] = df['Volume'] / df[f'Volume_SMA_{window}']
        
        # Add volume surge flag
        df['Volume_Surge'] = (df['Volume_Ratio'] > 2.0).astype(int)
        
        return df
    
    except Exception as e:
        logging.error(f"Error calculating Volume SMA: {e}")
        return df

def add_stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Add Stochastic Oscillator
    
    Args:
        df (pd.DataFrame): Dataframe with 'High', 'Low' and 'Close' prices
        k_period (int, optional): K period. Defaults to 14.
        d_period (int, optional): D period. Defaults to 3.
        
    Returns:
        pd.DataFrame: Dataframe with Stochastic Oscillator added
    """
    df = df.copy()
    
    try:
        # Calculate %K
        lowest_low = df['Low'].rolling(window=k_period).min()
        highest_high = df['High'].rolling(window=k_period).max()
        df['Stoch_K'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
        
        # Add crossover signals
        df['Stoch_Crossover'] = ((df['Stoch_K'] > df['Stoch_D']) & 
                               (df['Stoch_K'].shift() <= df['Stoch_D'].shift())).astype(int)
        df['Stoch_Crossunder'] = ((df['Stoch_K'] < df['Stoch_D']) & 
                                (df['Stoch_K'].shift() >= df['Stoch_D'].shift())).astype(int)
        
        return df
    
    except Exception as e:
        logging.error(f"Error calculating Stochastic Oscillator: {e}")
        return df

def add_ichimoku_cloud(df: pd.DataFrame, 
                      tenkan_period: int = 9, 
                      kijun_period: int = 26, 
                      senkou_span_b_period: int = 52, 
                      chikou_period: int = 26) -> pd.DataFrame:
    """
    Add Ichimoku Cloud
    
    Args:
        df (pd.DataFrame): Dataframe with 'High', 'Low' and 'Close' prices
        tenkan_period (int, optional): Tenkan-sen period. Defaults to 9.
        kijun_period (int, optional): Kijun-sen period. Defaults to 26.
        senkou_span_b_period (int, optional): Senkou Span B period. Defaults to 52.
        chikou_period (int, optional): Chikou Span period. Defaults to 26.
        
    Returns:
        pd.DataFrame: Dataframe with Ichimoku Cloud added
    """
    df = df.copy()
    
    try:
        # Calculate Tenkan-sen (Conversion Line)
        tenkan_high = df['High'].rolling(window=tenkan_period).max()
        tenkan_low = df['Low'].rolling(window=tenkan_period).min()
        df['Ichimoku_Tenkan'] = (tenkan_high + tenkan_low) / 2
        
        # Calculate Kijun-sen (Base Line)
        kijun_high = df['High'].rolling(window=kijun_period).max()
        kijun_low = df['Low'].rolling(window=kijun_period).min()
        df['Ichimoku_Kijun'] = (kijun_high + kijun_low) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        df['Ichimoku_SpanA'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(kijun_period)
        
        # Calculate Senkou Span B (Leading Span B)
        senkou_high = df['High'].rolling(window=senkou_span_b_period).max()
        senkou_low = df['Low'].rolling(window=senkou_span_b_period).min()
        df['Ichimoku_SpanB'] = ((senkou_high + senkou_low) / 2).shift(kijun_period)
        
        # Calculate Chikou Span (Lagging Span)
        df['Ichimoku_Chikou'] = df['Close'].shift(-chikou_period)
        
        # Add signal
        df['Ichimoku_Signal'] = np.where(df['Ichimoku_SpanA'] > df['Ichimoku_SpanB'], 1, 
                                      np.where(df['Ichimoku_SpanA'] < df['Ichimoku_SpanB'], -1, 0))
        
        return df
    
    except Exception as e:
        logging.error(f"Error calculating Ichimoku Cloud: {e}")
        return df

def add_fibonacci_levels(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Add Fibonacci Retracement Levels based on recent high and low
    
    Args:
        df (pd.DataFrame): Dataframe with 'High' and 'Low' prices
        window (int, optional): Window to find high and low points. Defaults to 50.
        
    Returns:
        pd.DataFrame: Dataframe with Fibonacci levels added
    """
    df = df.copy()
    
    try:
        # Find highest high and lowest low within the window
        high = df['High'].rolling(window=window).max()
        low = df['Low'].rolling(window=window).min()
        
        # Calculate Fibonacci levels
        diff = high - low
        df['Fib_0'] = low
        df['Fib_0.236'] = low + 0.236 * diff
        df['Fib_0.382'] = low + 0.382 * diff
        df['Fib_0.5'] = low + 0.5 * diff
        df['Fib_0.618'] = low + 0.618 * diff
        df['Fib_0.786'] = low + 0.786 * diff
        df['Fib_1'] = high
        
        return df
    
    except Exception as e:
        logging.error(f"Error calculating Fibonacci levels: {e}")
        return df

def calculate_support_resistance(df: pd.DataFrame, window: int = 20, tolerance: float = 0.02) -> Tuple[List[float], List[float]]:
    """
    Calculate support and resistance levels using pivot points
    
    Args:
        df (pd.DataFrame): Dataframe with 'High' and 'Low' prices
        window (int, optional): Window size to identify pivot points. Defaults to 20.
        tolerance (float, optional): Tolerance for grouping levels. Defaults to 0.02 (2%).
        
    Returns:
        Tuple[List[float], List[float]]: Support and resistance levels
    """
    try:
        highs = df['High'].values
        lows = df['Low'].values
        
        # Find pivot highs
        pivot_highs = []
        for i in range(window, len(highs) - window):
            if highs[i] > max(highs[i-window:i]) and highs[i] > max(highs[i+1:i+window+1]):
                pivot_highs.append(highs[i])
        
        # Find pivot lows
        pivot_lows = []
        for i in range(window, len(lows) - window):
            if lows[i] < min(lows[i-window:i]) and lows[i] < min(lows[i+1:i+window+1]):
                pivot_lows.append(lows[i])
        
        # Group similar levels
        support_levels = group_price_levels(pivot_lows, tolerance)
        resistance_levels = group_price_levels(pivot_highs, tolerance)
        
        return support_levels, resistance_levels
    
    except Exception as e:
        logging.error(f"Error calculating support/resistance levels: {e}")
        return [], []

def group_price_levels(price_levels: List[float], tolerance: float) -> List[float]:
    """
    Group similar price levels together
    
    Args:
        price_levels (List[float]): List of price levels
        tolerance (float): Tolerance for grouping as a percentage
        
    Returns:
        List[float]: Grouped price levels
    """
    if not price_levels:
        return []
    
    # Sort price levels
    sorted_levels = sorted(price_levels)
    
    # Group similar levels
    grouped_levels = []
    current_group = [sorted_levels[0]]
    
    for i in range(1, len(sorted_levels)):
        # If current level is within tolerance of the average of current group
        group_avg = sum(current_group) / len(current_group)
        if abs(sorted_levels[i] - group_avg) / group_avg <= tolerance:
            current_group.append(sorted_levels[i])
        else:
            # Add average of current group to results
            grouped_levels.append(sum(current_group) / len(current_group))
            # Start new group
            current_group = [sorted_levels[i]]
    
    # Add the last group
    grouped_levels.append(sum(current_group) / len(current_group))
    
    return grouped_levels 