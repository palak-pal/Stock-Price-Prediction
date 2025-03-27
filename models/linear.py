"""
Linear Regression model for stock prediction
"""
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from typing import Tuple, List, Dict

class LinearRegressionModel:
    """Linear Regression model for stock prediction"""
    
    def __init__(self):
        """Initialize the model"""
        self.model = None
        self.is_fitted = False
        self.metrics = {}
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training the model
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: X and y for model training
        """
        # Create feature for training - convert date to ordinal for simple model
        df['DateOrdinal'] = pd.to_datetime(df['Date']).map(datetime.toordinal)
        
        # Simple model just uses the date as a feature
        X = df[['DateOrdinal']]
        y = df['Close']
        
        return X, y
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit the model to the data
        
        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): Target
        """
        try:
            logging.info("Fitting Linear Regression model")
            self.model = LinearRegression()
            self.model.fit(X, y)
            self.is_fitted = True
            
            # Calculate training metrics
            y_pred = self.model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y - y_pred))
            r2 = self.model.score(X, y)
            
            self.metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            logging.info(f"Model fitted with metrics: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")
        except Exception as e:
            logging.error(f"Error fitting Linear Regression model: {e}")
            self.is_fitted = False
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            logging.error("Model is not fitted")
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            logging.info("Making predictions with Linear Regression model")
            return self.model.predict(X)
        except Exception as e:
            logging.error(f"Error making predictions with Linear Regression model: {e}")
            raise
    
    def predict_future(self, X: pd.DataFrame, y: pd.DataFrame, days_to_predict: int) -> Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions for future dates
        
        Args:
            X (pd.DataFrame): Features
            y (pd.DataFrame): Target
            days_to_predict (int): Number of days to predict
            
        Returns:
            Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray]: Future dates, predictions, upper bounds, lower bounds
        """
        # Fit the model if not already fitted
        if not self.is_fitted:
            self.fit(X, y)
        
        # Generate future dates
        last_date_ordinal = X['DateOrdinal'].iloc[-1]
        future_date_ordinals = np.array([last_date_ordinal + i for i in range(1, days_to_predict + 1)])
        
        # Predict future prices
        future_dates = [datetime.fromordinal(int(date)) for date in future_date_ordinals]
        future_X = pd.DataFrame({'DateOrdinal': future_date_ordinals})
        predictions = self.predict(future_X)
        
        # Calculate confidence intervals
        y_pred = self.predict(X)
        residuals = y - y_pred
        std_residuals = np.std(residuals)
        
        upper_bounds = predictions + 1.96 * std_residuals
        lower_bounds = predictions - 1.96 * std_residuals
        
        return future_dates, predictions, upper_bounds, lower_bounds
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get model metrics
        
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        return self.metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance
        
        Returns:
            Dict[str, float]: Dictionary of feature importances
        """
        if not self.is_fitted:
            logging.error("Model is not fitted")
            raise ValueError("Model must be fitted before getting feature importance")
        
        try:
            return {'DateOrdinal': self.model.coef_[0]}
        except Exception as e:
            logging.error(f"Error getting feature importance: {e}")
            return {'DateOrdinal': 0.0} 