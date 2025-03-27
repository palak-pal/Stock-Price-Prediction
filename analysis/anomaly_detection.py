"""
Anomaly detection utilities for Stock Prediction App
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from statsmodels.tsa.seasonal import seasonal_decompose

def detect_anomalies(df: pd.DataFrame, method: str = 'zscore', params: Dict = None) -> pd.DataFrame:
    """
    Detect anomalies in stock price data using various methods
    
    Args:
        df (pd.DataFrame): Dataframe with stock price data
        method (str, optional): Anomaly detection method. 
            Options: 'zscore', 'isolation_forest', 'dbscan', 'volatility'. 
            Defaults to 'zscore'.
        params (Dict, optional): Parameters for the selected method. Defaults to None.
        
    Returns:
        pd.DataFrame: DataFrame with added anomaly flags and scores
    """
    df = df.copy()
    
    # Default parameters for each method
    default_params = {
        'zscore': {'window': 20, 'threshold': 2.5},
        'isolation_forest': {'contamination': 0.05, 'random_state': 42},
        'dbscan': {'eps': 0.5, 'min_samples': 5},
        'volatility': {'window': 20, 'multiplier': 2.5}
    }
    
    # Use default parameters if none provided
    if params is None:
        params = default_params.get(method, {})
    else:
        # Merge provided params with defaults
        for k, v in default_params.get(method, {}).items():
            if k not in params:
                params[k] = v
    
    try:
        if method == 'zscore':
            return detect_anomalies_zscore(df, **params)
        elif method == 'isolation_forest':
            return detect_anomalies_isolation_forest(df, **params)
        elif method == 'dbscan':
            return detect_anomalies_dbscan(df, **params)
        elif method == 'volatility':
            return detect_anomalies_volatility(df, **params)
        else:
            logging.warning(f"Unknown anomaly detection method: {method}, using Z-score instead")
            return detect_anomalies_zscore(df, **default_params['zscore'])
    
    except Exception as e:
        logging.error(f"Error in anomaly detection: {e}")
        df['Is_Anomaly'] = False
        df['Anomaly_Score'] = 0.0
        return df

def detect_anomalies_zscore(df: pd.DataFrame, window: int = 20, threshold: float = 2.5) -> pd.DataFrame:
    """
    Detect anomalies using the Z-score method
    
    Args:
        df (pd.DataFrame): DataFrame with stock price data
        window (int, optional): Rolling window size. Defaults to 20.
        threshold (float, optional): Z-score threshold for anomalies. Defaults to 2.5.
        
    Returns:
        pd.DataFrame: DataFrame with added anomaly flags and Z-scores
    """
    df = df.copy()
    
    try:
        # Calculate rolling mean and standard deviation
        rolling_mean = df['Close'].rolling(window=window).mean()
        rolling_std = df['Close'].rolling(window=window).std()
        
        # Calculate Z-score
        df['Z_Score'] = (df['Close'] - rolling_mean) / rolling_std
        
        # Flag anomalies
        df['Is_Anomaly'] = (df['Z_Score'].abs() > threshold)
        
        # Set anomaly score equal to Z-score
        df['Anomaly_Score'] = df['Z_Score'].abs()
        
        return df
    
    except Exception as e:
        logging.error(f"Error in Z-score anomaly detection: {e}")
        df['Z_Score'] = 0.0
        df['Is_Anomaly'] = False
        df['Anomaly_Score'] = 0.0
        return df

def detect_anomalies_isolation_forest(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest algorithm
    
    Args:
        df (pd.DataFrame): DataFrame with stock price data
        contamination (float, optional): Expected proportion of anomalies. Defaults to 0.05.
        random_state (int, optional): Random seed. Defaults to 42.
        
    Returns:
        pd.DataFrame: DataFrame with added anomaly flags and scores
    """
    df = df.copy()
    
    try:
        # Prepare features for Isolation Forest
        # Use price, daily return, and volume if available
        features = ['Close']
        df['Daily_Return'] = df['Close'].pct_change() * 100
        features.append('Daily_Return')
        
        if 'Volume' in df.columns:
            df['Volume_Change'] = df['Volume'].pct_change() * 100
            features.append('Volume_Change')
            
        # Fill NaN values
        df_features = df[features].fillna(0)
        
        # Create and fit Isolation Forest model
        model = IsolationForest(contamination=contamination, random_state=random_state)
        df['Anomaly_Score'] = model.fit_predict(df_features)
        
        # Convert predictions to anomaly flag
        # Isolation Forest returns -1 for anomalies, 1 for normal data
        df['Is_Anomaly'] = (df['Anomaly_Score'] == -1)
        
        # Convert scores to positive anomaly scores (higher is more anomalous)
        df['Anomaly_Score'] = model.decision_function(df_features) * -1
        
        return df
    
    except Exception as e:
        logging.error(f"Error in Isolation Forest anomaly detection: {e}")
        df['Is_Anomaly'] = False
        df['Anomaly_Score'] = 0.0
        return df

def detect_anomalies_dbscan(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> pd.DataFrame:
    """
    Detect anomalies using DBSCAN clustering
    
    Args:
        df (pd.DataFrame): DataFrame with stock price data
        eps (float, optional): Maximum distance between samples for clustering. Defaults to 0.5.
        min_samples (int, optional): Minimum samples in a cluster. Defaults to 5.
        
    Returns:
        pd.DataFrame: DataFrame with added anomaly flags and scores
    """
    df = df.copy()
    
    try:
        # Prepare features for DBSCAN
        # Use price, daily return, and volume if available
        features = ['Close']
        df['Daily_Return'] = df['Close'].pct_change() * 100
        features.append('Daily_Return')
        
        if 'Volume' in df.columns:
            df['Volume_Change'] = df['Volume'].pct_change() * 100
            features.append('Volume_Change')
            
        # Fill NaN values
        df_features = df[features].fillna(0)
        
        # Normalize features
        df_norm = (df_features - df_features.mean()) / df_features.std()
        
        # Create and fit DBSCAN model
        model = DBSCAN(eps=eps, min_samples=min_samples)
        df['Cluster'] = model.fit_predict(df_norm)
        
        # Points with cluster label -1 are anomalies
        df['Is_Anomaly'] = (df['Cluster'] == -1)
        
        # Calculate anomaly score based on distance to nearest cluster
        # For simplicity, using absolute daily return as a proxy for anomaly score
        df['Anomaly_Score'] = df['Daily_Return'].abs()
        
        # Drop temporary cluster column
        df.drop('Cluster', axis=1, inplace=True)
        
        return df
    
    except Exception as e:
        logging.error(f"Error in DBSCAN anomaly detection: {e}")
        df['Is_Anomaly'] = False
        df['Anomaly_Score'] = 0.0
        return df

def detect_anomalies_volatility(df: pd.DataFrame, window: int = 20, multiplier: float = 2.5) -> pd.DataFrame:
    """
    Detect anomalies based on volatility
    
    Args:
        df (pd.DataFrame): DataFrame with stock price data
        window (int, optional): Window for volatility calculation. Defaults to 20.
        multiplier (float, optional): Multiplier for volatility threshold. Defaults to 2.5.
        
    Returns:
        pd.DataFrame: DataFrame with added anomaly flags and scores
    """
    df = df.copy()
    
    try:
        # Calculate daily returns
        df['Daily_Return'] = df['Close'].pct_change() * 100
        
        # Calculate rolling volatility (standard deviation of returns)
        df['Volatility'] = df['Daily_Return'].rolling(window=window).std()
        
        # Calculate volatility of volatility
        df['Vol_of_Vol'] = df['Volatility'].rolling(window=window).std()
        
        # Calculate Z-score of volatility
        df['Volatility_Z'] = ((df['Volatility'] - df['Volatility'].rolling(window=window).mean()) / 
                           df['Vol_of_Vol'])
        
        # Flag volatility anomalies
        df['Is_Anomaly'] = (df['Volatility_Z'].abs() > multiplier)
        
        # Set anomaly score
        df['Anomaly_Score'] = df['Volatility_Z'].abs()
        
        return df
    
    except Exception as e:
        logging.error(f"Error in volatility anomaly detection: {e}")
        df['Is_Anomaly'] = False
        df['Anomaly_Score'] = 0.0
        return df

def detect_seasonality_and_anomalies(df: pd.DataFrame, period: int = 252, model: str = 'additive') -> pd.DataFrame:
    """
    Detect seasonality and anomalies from trend/seasonal patterns
    
    Args:
        df (pd.DataFrame): DataFrame with stock price data
        period (int, optional): Period for seasonality. Defaults to 252 (trading days in year).
        model (str, optional): Decomposition model type ('additive' or 'multiplicative'). 
            Defaults to 'additive'.
        
    Returns:
        pd.DataFrame: DataFrame with added seasonality components and anomalies
    """
    df = df.copy()
    
    try:
        # Need enough data for at least 2 periods
        if len(df) < period * 2:
            logging.warning(f"Not enough data for seasonality detection with period={period}")
            period = min(len(df) // 2, 20)  # Use smaller period or default to 20
            logging.info(f"Using smaller period = {period} for seasonality detection")
        
        # Set index to datetime for seasonal_decompose
        df_decompose = df.copy()
        if not isinstance(df_decompose.index, pd.DatetimeIndex):
            if 'Date' in df_decompose.columns:
                df_decompose.set_index('Date', inplace=True)
            else:
                # Create artificial DatetimeIndex
                df_decompose.index = pd.date_range(start='2000-01-01', periods=len(df_decompose))
        
        # Decompose time series
        result = seasonal_decompose(df_decompose['Close'], model=model, period=period)
        
        # Add components back to original dataframe
        if 'Date' in df.columns:
            df['Trend'] = result.trend.values
            df['Seasonal'] = result.seasonal.values
            df['Residual'] = result.resid.values
        else:
            df['Trend'] = result.trend.values
            df['Seasonal'] = result.seasonal.values
            df['Residual'] = result.resid.values
        
        # Detect anomalies in residual component
        residual_mean = df['Residual'].mean()
        residual_std = df['Residual'].std()
        
        df['Residual_Z'] = (df['Residual'] - residual_mean) / residual_std
        df['Is_Seasonal_Anomaly'] = (df['Residual_Z'].abs() > 2.5)
        
        return df
    
    except Exception as e:
        logging.error(f"Error in seasonality detection: {e}")
        return df

def detect_change_points(df: pd.DataFrame, window: int = 20, threshold: float = 2.0) -> pd.DataFrame:
    """
    Detect change points in time series data
    
    Args:
        df (pd.DataFrame): DataFrame with stock price data
        window (int, optional): Window size for rolling statistics. Defaults to 20.
        threshold (float, optional): Threshold for change point detection. Defaults to 2.0.
        
    Returns:
        pd.DataFrame: DataFrame with added change point flags
    """
    df = df.copy()
    
    try:
        # Calculate rolling mean and standard deviation
        rolling_mean = df['Close'].rolling(window=window).mean()
        rolling_std = df['Close'].rolling(window=window).std()
        
        # Calculate mean shift and volatility ratio
        df['Mean_Shift'] = ((rolling_mean - rolling_mean.shift(window)) / 
                        rolling_mean.shift(window).abs()) * 100
        df['Vol_Ratio'] = (rolling_std / rolling_std.shift(window))
        
        # Detect trend changes
        df['Trend_Change'] = ((df['Mean_Shift'].abs() > threshold) | 
                           (df['Vol_Ratio'] > (1 + threshold/100)) | 
                           (df['Vol_Ratio'] < (1 - threshold/100)))
        
        # Detect significant volume changes if volume data is available
        if 'Volume' in df.columns:
            rolling_vol_mean = df['Volume'].rolling(window=window).mean()
            df['Volume_Change'] = ((rolling_vol_mean - rolling_vol_mean.shift(window)) / 
                                rolling_vol_mean.shift(window).abs()) * 100
            df['Volume_Surge'] = (df['Volume_Change'] > threshold * 2)
            
            # Combine with trend changes
            df['Is_Change_Point'] = (df['Trend_Change'] | df['Volume_Surge'])
        else:
            df['Is_Change_Point'] = df['Trend_Change']
        
        return df
    
    except Exception as e:
        logging.error(f"Error in change point detection: {e}")
        df['Is_Change_Point'] = False
        return df

def analyze_anomalies(df: pd.DataFrame) -> Dict:
    """
    Analyze detected anomalies and generate a summary
    
    Args:
        df (pd.DataFrame): DataFrame with anomaly flags
        
    Returns:
        Dict: Dictionary with anomaly statistics and insights
    """
    result = {}
    
    try:
        # Check if anomaly detection has been run
        if 'Is_Anomaly' not in df.columns:
            return {"error": "Anomaly detection has not been run on this data"}
        
        # Count anomalies
        anomaly_count = df['Is_Anomaly'].sum()
        total_points = len(df)
        anomaly_percentage = (anomaly_count / total_points) * 100
        
        result['anomaly_count'] = int(anomaly_count)
        result['total_points'] = total_points
        result['anomaly_percentage'] = round(anomaly_percentage, 2)
        
        # Get anomaly dates
        anomaly_df = df[df['Is_Anomaly']]
        if 'Date' in anomaly_df.columns:
            anomaly_dates = anomaly_df['Date'].tolist()
        else:
            anomaly_dates = anomaly_df.index.tolist()
        
        result['anomaly_dates'] = anomaly_dates
        
        # Get most significant anomalies (highest scores)
        if 'Anomaly_Score' in df.columns:
            top_anomalies = anomaly_df.sort_values('Anomaly_Score', ascending=False).head(5)
            result['top_anomalies'] = []
            
            for _, row in top_anomalies.iterrows():
                anomaly_info = {
                    'date': row.get('Date', str(row.name)),
                    'close': row['Close'],
                    'score': row['Anomaly_Score']
                }
                
                if 'Daily_Return' in row:
                    anomaly_info['return'] = row['Daily_Return']
                    
                if 'Volume' in row:
                    anomaly_info['volume'] = row['Volume']
                    
                result['top_anomalies'].append(anomaly_info)
        
        # Check for clusters of anomalies
        result['has_anomaly_clusters'] = False
        
        if anomaly_count > 1:
            # Check if anomalies are consecutive
            consecutive_count = 0
            for i in range(1, len(df)):
                if df['Is_Anomaly'].iloc[i-1] and df['Is_Anomaly'].iloc[i]:
                    consecutive_count += 1
            
            result['consecutive_anomalies'] = consecutive_count
            result['has_anomaly_clusters'] = (consecutive_count > 0)
        
        # Check for correlation with market events if available
        if 'Is_Change_Point' in df.columns:
            # Count anomalies that also are change points
            anomaly_change_points = ((df['Is_Anomaly']) & (df['Is_Change_Point'])).sum()
            result['anomaly_change_points'] = int(anomaly_change_points)
            result['anomaly_change_point_percentage'] = round((anomaly_change_points / anomaly_count * 100), 2) if anomaly_count > 0 else 0
        
        return result
    
    except Exception as e:
        logging.error(f"Error in anomaly analysis: {e}")
        return {"error": str(e)}

def adjust_anomaly_threshold(df: pd.DataFrame, target_percentage: float = 0.05, 
                          max_iterations: int = 10, method: str = 'zscore') -> Tuple[pd.DataFrame, float]:
    """
    Adjust anomaly detection threshold to achieve target percentage of anomalies
    
    Args:
        df (pd.DataFrame): DataFrame with stock price data
        target_percentage (float, optional): Target percentage of anomalies. Defaults to 0.05.
        max_iterations (int, optional): Maximum iterations for adjustment. Defaults to 10.
        method (str, optional): Anomaly detection method. Defaults to 'zscore'.
        
    Returns:
        Tuple[pd.DataFrame, float]: DataFrame with anomalies and final threshold
    """
    # Initial parameters
    if method == 'zscore':
        threshold = 2.5
        params = {'window': 20, 'threshold': threshold}
    elif method == 'isolation_forest':
        threshold = target_percentage  # Isolation Forest uses contamination
        params = {'contamination': threshold, 'random_state': 42}
    elif method == 'volatility':
        threshold = 2.5
        params = {'window': 20, 'multiplier': threshold}
    else:
        # For DBSCAN and other methods, not easily adjustable
        logging.warning(f"Threshold adjustment not implemented for method: {method}")
        df_result = detect_anomalies(df, method=method)
        return df_result, 0.0
    
    # Iteratively adjust threshold
    for i in range(max_iterations):
        df_result = detect_anomalies(df, method=method, params=params)
        anomaly_percentage = df_result['Is_Anomaly'].mean()
        
        # Check if we're close enough to target
        if abs(anomaly_percentage - target_percentage) < 0.005:
            break
            
        # Adjust threshold
        if anomaly_percentage > target_percentage:
            # Too many anomalies, increase threshold
            if method == 'isolation_forest':
                threshold *= 0.8  # Reduce contamination
            else:
                threshold *= 1.1  # Increase Z-score threshold
        else:
            # Too few anomalies, decrease threshold
            if method == 'isolation_forest':
                threshold *= 1.2  # Increase contamination
            else:
                threshold *= 0.9  # Decrease Z-score threshold
                
        # Update parameters
        if method == 'zscore':
            params['threshold'] = threshold
        elif method == 'isolation_forest':
            params['contamination'] = min(threshold, 0.5)  # Cap at 0.5
        elif method == 'volatility':
            params['multiplier'] = threshold
    
    return df_result, threshold 