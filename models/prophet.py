"""
Prophet model for stock prediction
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional, Union

class ProphetModel:
    """Prophet model for stock prediction"""
    
    def __init__(self):
        """Initialize the model"""
        self.model = None
        self.is_fitted = False
        self.metrics = {}
        self.has_prophet = self._check_prophet()
    
    def _check_prophet(self) -> bool:
        """
        Check if Prophet is available
        
        Returns:
            bool: True if Prophet is available, False otherwise
        """
        try:
            from prophet import Prophet
            
            # Store the Prophet class for later use
            self.Prophet = Prophet
            
            logging.info("Prophet is available")
            return True
        except ImportError:
            logging.warning("Prophet is not available, Prophet model will not be used")
            return False
    
    def prepare_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare data for training the model
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            
        Returns:
            Optional[pd.DataFrame]: DataFrame for Prophet or None if Prophet not available
        """
        if not self.has_prophet:
            return None
        
        try:
            # Prepare data for Prophet - ensure timezone-naive datetimes
            prophet_df = df[['Date', 'Close']].copy()
            # Convert to timezone-naive datetime if needed
            if hasattr(prophet_df['Date'].dtype, 'tz') and prophet_df['Date'].dtype.tz is not None:
                prophet_df['Date'] = prophet_df['Date'].dt.tz_localize(None)
            prophet_df = prophet_df.rename(columns={'Date': 'ds', 'Close': 'y'})
            
            return prophet_df
        except Exception as e:
            logging.error(f"Error preparing data for Prophet model: {e}")
            return None
    
    def fit(self, prophet_df: pd.DataFrame) -> None:
        """
        Fit the model to the data
        
        Args:
            prophet_df (pd.DataFrame): DataFrame with ds and y columns
        """
        if not self.has_prophet or prophet_df is None:
            logging.error("Cannot fit Prophet model: Prophet not available or invalid data")
            return
        
        try:
            logging.info("Fitting Prophet model")
            
            # Create and fit model
            self.model = self.Prophet(daily_seasonality=True)
            self.model.fit(prophet_df)
            
            self.is_fitted = True
            
            # Calculate cross-validation metrics if we have enough data
            if len(prophet_df) >= 90:  # At least 90 days of data
                try:
                    from prophet.diagnostics import cross_validation, performance_metrics
                    
                    # Perform cross-validation
                    initial = int(len(prophet_df) * 0.5)  # Initial training period is 50% of data
                    period = int(len(prophet_df) * 0.1)   # Each cutoff is 10% of data
                    horizon = int(len(prophet_df) * 0.2)  # Forecast horizon is 20% of data
                    
                    df_cv = cross_validation(
                        self.model, 
                        initial=f"{initial} days",
                        period=f"{period} days", 
                        horizon=f"{horizon} days"
                    )
                    
                    # Calculate performance metrics
                    df_p = performance_metrics(df_cv)
                    
                    # Store metrics
                    self.metrics = {
                        'mae': df_p['mae'].mean(),
                        'rmse': df_p['rmse'].mean(),
                        'mape': df_p['mape'].mean()
                    }
                    
                    logging.info(f"Model fitted with metrics: MAE={self.metrics['mae']:.2f}, RMSE={self.metrics['rmse']:.2f}, MAPE={self.metrics['mape']:.2f}")
                except Exception as e:
                    logging.warning(f"Error calculating cross-validation metrics: {e}")
                    self.metrics = {}
            else:
                logging.warning("Not enough data for cross-validation")
                self.metrics = {}
        except Exception as e:
            logging.error(f"Error fitting Prophet model: {e}")
            self.is_fitted = False
    
    def predict_future(self, df: pd.DataFrame, days_to_predict: int) -> Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions for future dates
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            days_to_predict (int): Number of days to predict
            
        Returns:
            Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray]: Future dates, predictions, upper bounds, lower bounds
        """
        if not self.has_prophet:
            logging.error("Cannot predict with Prophet model: Prophet not available")
            return [], np.array([]), np.array([]), np.array([])
        
        try:
            # Prepare data
            prophet_df = self.prepare_data(df)
            
            # Fit the model if not already fitted
            if not self.is_fitted:
                self.fit(prophet_df)
            
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=days_to_predict)
            forecast = self.model.predict(future)
            
            # Extract predictions and confidence intervals
            future_forecast = forecast.iloc[-days_to_predict:]
            
            future_dates = future_forecast['ds'].dt.to_pydatetime().tolist()
            predictions = future_forecast['yhat'].values
            upper_bounds = future_forecast['yhat_upper'].values
            lower_bounds = future_forecast['yhat_lower'].values
            
            return future_dates, predictions, upper_bounds, lower_bounds
        
        except Exception as e:
            logging.error(f"Error predicting future with Prophet model: {e}")
            return [], np.array([]), np.array([]), np.array([])
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get model metrics
        
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        return self.metrics
    
    def get_components(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Get model components (trend, seasonality, etc.)
        
        Returns:
            Optional[Dict[str, pd.DataFrame]]: Dictionary of components or None if not available
        """
        if not self.has_prophet or not self.is_fitted:
            logging.error("Cannot get components: Prophet not available or model not fitted")
            return None
        
        try:
            # Get forecast for the most recent period
            forecast = self.model.predict()
            
            # Extract components
            components = {
                'trend': forecast[['ds', 'trend']],
                'yearly': forecast[['ds', 'yearly']] if 'yearly' in forecast.columns else None,
                'weekly': forecast[['ds', 'weekly']] if 'weekly' in forecast.columns else None,
                'daily': forecast[['ds', 'daily']] if 'daily' in forecast.columns else None
            }
            
            return components
        except Exception as e:
            logging.error(f"Error getting model components: {e}")
            return None 