"""
LSTM Neural Network model for stock prediction
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional, Union

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class LSTMModel:
    """LSTM Neural Network model for stock prediction"""
    
    def __init__(self, sequence_length: int = config.LSTM_SEQUENCE_LENGTH):
        """
        Initialize the model
        
        Args:
            sequence_length (int, optional): Length of input sequences. Defaults to config value.
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.metrics = {}
        self.has_tensorflow = self._check_tensorflow()
    
    def _check_tensorflow(self) -> bool:
        """
        Check if TensorFlow is available
        
        Returns:
            bool: True if TensorFlow is available, False otherwise
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from sklearn.preprocessing import MinMaxScaler
            
            # Store these imported modules for later use
            self.tf = tf
            self.Sequential = Sequential
            self.LSTM = LSTM
            self.Dense = Dense
            self.Dropout = Dropout
            self.MinMaxScaler = MinMaxScaler
            
            logging.info("TensorFlow is available")
            return True
        except ImportError:
            logging.warning("TensorFlow is not available, LSTM model will not be used")
            return False
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare data for training the model
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: X and y for model training, or None if TensorFlow not available
        """
        if not self.has_tensorflow:
            return None, None
        
        try:
            # Extract close prices
            data = df['Close'].values.reshape(-1, 1)
            
            # Scale the data
            self.scaler = self.MinMaxScaler(feature_range=(0, 1))
            scaled_data = self.scaler.fit_transform(data)
            
            # Create sequences
            if len(scaled_data) < self.sequence_length:
                raise ValueError(f"Not enough data points. Need at least {self.sequence_length}.")
                
            X = []
            y = []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i, 0])
                y.append(scaled_data[i, 0])
                
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            return X, y
        except Exception as e:
            logging.error(f"Error preparing data for LSTM model: {e}")
            return None, None
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture
        
        Args:
            input_shape (Tuple[int, int]): Shape of input data (sequence_length, 1)
        """
        if not self.has_tensorflow:
            return
        
        try:
            # Build LSTM model
            model = self.Sequential()
            model.add(self.LSTM(units=50, return_sequences=True, input_shape=input_shape))
            model.add(self.Dropout(0.2))
            model.add(self.LSTM(units=50, return_sequences=False))
            model.add(self.Dropout(0.2))
            model.add(self.Dense(units=1))
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            self.model = model
        except Exception as e:
            logging.error(f"Error building LSTM model: {e}")
            self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = config.LSTM_EPOCHS, batch_size: int = config.LSTM_BATCH_SIZE) -> None:
        """
        Fit the model to the data
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target
            epochs (int, optional): Number of epochs. Defaults to config value.
            batch_size (int, optional): Batch size. Defaults to config value.
        """
        if not self.has_tensorflow or X is None or y is None:
            logging.error("Cannot fit LSTM model: TensorFlow not available or invalid data")
            return
        
        try:
            logging.info(f"Fitting LSTM model with {epochs} epochs and batch size {batch_size}")
            
            # Build model if it doesn't exist
            if self.model is None:
                self.build_model((X.shape[1], 1))
            
            # Train the model
            history = self.model.fit(
                X, y, 
                epochs=epochs, 
                batch_size=batch_size, 
                verbose=0
            )
            
            self.is_fitted = True
            
            # Calculate training metrics
            self.metrics = {
                'loss': history.history['loss'][-1],
                'val_loss': history.history.get('val_loss', [0])[-1]
            }
            
            logging.info(f"Model fitted with loss: {self.metrics['loss']:.4f}")
        except Exception as e:
            logging.error(f"Error fitting LSTM model: {e}")
            self.is_fitted = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model
        
        Args:
            X (np.ndarray): Features
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.has_tensorflow or not self.is_fitted:
            logging.error("Cannot predict with LSTM model: TensorFlow not available or model not fitted")
            return np.array([])
        
        try:
            logging.info("Making predictions with LSTM model")
            predictions = self.model.predict(X, verbose=0)
            return predictions
        except Exception as e:
            logging.error(f"Error making predictions with LSTM model: {e}")
            return np.array([])
    
    def predict_future(self, df: pd.DataFrame, days_to_predict: int) -> Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions for future dates
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            days_to_predict (int): Number of days to predict
            
        Returns:
            Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray]: Future dates, predictions, upper bounds, lower bounds
        """
        if not self.has_tensorflow:
            logging.error("Cannot predict with LSTM model: TensorFlow not available")
            return [], np.array([]), np.array([]), np.array([])
        
        try:
            # Prepare data
            X, y = self.prepare_data(df)
            
            # Fit the model if not already fitted
            if not self.is_fitted:
                self.fit(X, y)
            
            # Get the last sequence from data
            data = df['Close'].values.reshape(-1, 1)
            scaled_data = self.scaler.transform(data)
            
            last_sequence = scaled_data[-self.sequence_length:]
            future_predictions = []
            
            # Generate predictions iteratively
            for _ in range(days_to_predict):
                # Reshape sequence for prediction
                next_seq = last_sequence.reshape(1, self.sequence_length, 1)
                
                # Predict next value
                next_pred = self.model.predict(next_seq, verbose=0)
                future_predictions.append(next_pred[0, 0])
                
                # Update sequence with new prediction
                last_sequence = np.append(last_sequence[1:], next_pred)
                last_sequence = last_sequence.reshape(-1, 1)
            
            # Convert predictions back to original scale
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = self.scaler.inverse_transform(future_predictions)
            
            # Generate future dates
            last_date = pd.to_datetime(df['Date'].iloc[-1])
            future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
            
            # Calculate error bounds
            predictions = future_predictions.flatten()
            
            # Calculate average volatility from historical data for error bounds
            daily_returns = df['Close'].pct_change().dropna()
            avg_volatility = daily_returns.std()
            volatility_factor = 1.96  # 95% confidence interval
            
            upper_bounds = predictions * (1 + avg_volatility * volatility_factor)
            lower_bounds = predictions * (1 - avg_volatility * volatility_factor)
            
            return future_dates, predictions, upper_bounds, lower_bounds
        
        except Exception as e:
            logging.error(f"Error predicting future with LSTM model: {e}")
            return [], np.array([]), np.array([]), np.array([])
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get model metrics
        
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        return self.metrics 