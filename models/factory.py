"""
Model factory for stock prediction models
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Import model classes
from models.linear import LinearRegressionModel
from models.lstm import LSTMModel
from models.prophet import ProphetModel

class ModelFactory:
    """Factory class for creating and managing prediction models"""
    
    def __init__(self):
        """Initialize the factory"""
        self.models = {}
        self.model_classes = {
            'linear': LinearRegressionModel,
            'lstm': LSTMModel,
            'prophet': ProphetModel
        }
        self.model_performance = {}
    
    def get_model(self, model_type: str) -> Any:
        """
        Get a model instance of the specified type
        
        Args:
            model_type (str): Type of model to get
            
        Returns:
            Any: Model instance
        """
        model_type = model_type.lower()
        
        # Check if we already have an instance of this model
        if model_type in self.models:
            logging.info(f"Returning existing {model_type} model")
            return self.models[model_type]
        
        # Check if we have the model class
        if model_type not in self.model_classes:
            logging.error(f"Unknown model type: {model_type}")
            # Fall back to linear regression
            model_type = 'linear'
            logging.info(f"Falling back to {model_type} model")
        
        # Create a new model instance
        try:
            logging.info(f"Creating new {model_type} model")
            model = self.model_classes[model_type]()
            self.models[model_type] = model
            return model
        except Exception as e:
            logging.error(f"Error creating {model_type} model: {e}")
            # Fall back to linear regression
            if model_type != 'linear':
                logging.info("Falling back to linear regression model")
                return self.get_model('linear')
            raise
    
    def predict(self, model_type: str, df: pd.DataFrame, days_to_predict: int) -> Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with the specified model
        
        Args:
            model_type (str): Type of model to use
            df (pd.DataFrame): DataFrame with stock data
            days_to_predict (int): Number of days to predict
            
        Returns:
            Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray]: Future dates, predictions, upper bounds, lower bounds
        """
        model_type = model_type.lower()
        
        # Map Streamlit-friendly names to internal names
        model_map = {
            'linear regression': 'linear',
            'lstm neural network': 'lstm',
            'prophet': 'prophet'
        }
        
        if model_type in model_map:
            model_type = model_map[model_type]
        
        try:
            model = self.get_model(model_type)
            
            if model_type == 'linear':
                # Linear Regression model expects X and y
                X, y = model.prepare_data(df)
                return model.predict_future(X, y, days_to_predict)
            elif model_type == 'lstm' or model_type == 'prophet':
                # LSTM and Prophet models expect the entire DataFrame
                return model.predict_future(df, days_to_predict)
            else:
                logging.error(f"Unknown model type for prediction: {model_type}")
                # Fall back to linear regression
                model = self.get_model('linear')
                X, y = model.prepare_data(df)
                return model.predict_future(X, y, days_to_predict)
        except Exception as e:
            logging.error(f"Error predicting with {model_type} model: {e}")
            # Fall back to linear regression
            try:
                logging.info("Falling back to linear regression for prediction")
                model = self.get_model('linear')
                X, y = model.prepare_data(df)
                return model.predict_future(X, y, days_to_predict)
            except Exception as e2:
                logging.error(f"Error with fallback prediction: {e2}")
                raise
    
    def compare_models(self, df: pd.DataFrame, days_to_predict: int) -> Dict[str, Dict[str, Any]]:
        """
        Compare performance of all available models
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            days_to_predict (int): Number of days to predict
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of model comparisons
        """
        results = {}
        
        for model_type in self.model_classes.keys():
            try:
                model = self.get_model(model_type)
                
                # Track execution time
                start_time = datetime.now()
                
                if model_type == 'linear':
                    X, y = model.prepare_data(df)
                    future_dates, predictions, upper_bounds, lower_bounds = model.predict_future(X, y, days_to_predict)
                elif model_type == 'lstm' or model_type == 'prophet':
                    if (model_type == 'lstm' and not model.has_tensorflow) or (model_type == 'prophet' and not model.has_prophet):
                        continue  # Skip unavailable models
                    future_dates, predictions, upper_bounds, lower_bounds = model.predict_future(df, days_to_predict)
                
                # Calculate execution time
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Get model metrics
                metrics = model.get_metrics()
                
                # Calculate prediction range (difference between upper and lower bounds)
                prediction_range = np.mean(upper_bounds - lower_bounds)
                
                results[model_type] = {
                    'future_dates': future_dates,
                    'predictions': predictions,
                    'upper_bounds': upper_bounds,
                    'lower_bounds': lower_bounds,
                    'metrics': metrics,
                    'execution_time': execution_time,
                    'prediction_range': prediction_range
                }
                
                # Update model performance metrics
                self.model_performance[model_type] = {
                    'last_run': datetime.now(),
                    'execution_time': execution_time,
                    'metrics': metrics
                }
                
            except Exception as e:
                logging.error(f"Error comparing {model_type} model: {e}")
        
        return results
    
    def get_best_model(self, df: pd.DataFrame, metric: str = 'rmse') -> str:
        """
        Get the best model based on historical performance
        
        Args:
            df (pd.DataFrame): DataFrame with stock data to train models
            metric (str, optional): Metric to use for comparison. Defaults to 'rmse'.
            
        Returns:
            str: Name of the best model
        """
        if not self.model_performance:
            # No performance data available, perform comparison
            _ = self.compare_models(df, days_to_predict=30)
        
        # Find the best model based on the specified metric
        best_model = 'linear'  # Default to linear regression
        best_metric_value = float('inf')
        
        for model_type, performance in self.model_performance.items():
            if 'metrics' in performance and metric in performance['metrics']:
                # Lower metric value is better (rmse, mae, etc.)
                if performance['metrics'][metric] < best_metric_value:
                    best_metric_value = performance['metrics'][metric]
                    best_model = model_type
        
        return best_model
    
    def get_recommended_models(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Get recommended models for different scenarios
        
        Args:
            df (pd.DataFrame): DataFrame with stock data to train models
            
        Returns:
            Dict[str, str]: Dictionary of recommended models for different scenarios
        """
        # Perform comparison if not already done
        if not self.model_performance:
            _ = self.compare_models(df, days_to_predict=30)
        
        # Initialize recommendations
        recommendations = {
            'short_term': 'linear',
            'long_term': 'linear',
            'volatile_market': 'linear',
            'stable_market': 'linear',
            'overall': 'linear'
        }
        
        # Check for Prophet availability for long-term forecasts
        if 'prophet' in self.models and self.models['prophet'].has_prophet:
            recommendations['long_term'] = 'prophet'
        
        # Check for LSTM availability for volatile markets
        if 'lstm' in self.models and self.models['lstm'].has_tensorflow:
            recommendations['volatile_market'] = 'lstm'
        
        # Recommend the model with the best rmse as overall
        recommendations['overall'] = self.get_best_model(df, 'rmse')
        
        return recommendations 