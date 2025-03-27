"""
Configuration parameters for the Stock Prediction App
"""

# API Settings
NEWS_API_RETRIES = 2
NEWS_API_RETRY_DELAY = 1
DATA_API_RETRIES = 3
DATA_API_RETRY_DELAY = 2

# Data Cache Settings
STOCK_LIST_CACHE_TTL = 3600  # 1 hour
DATA_CACHE_TTL = 3600  # 1 hour
NEWS_CACHE_TTL = 3600  # 1 hour

# Model Settings
LSTM_SEQUENCE_LENGTH = 60
LSTM_EPOCHS = 25
LSTM_BATCH_SIZE = 32

# Prediction Settings
DEFAULT_FORECAST_DAYS = 30
MAX_FORECAST_DAYS = 90

# Visualization Settings
DEFAULT_CHART_TYPE = "line"
MOVING_AVERAGE_OPTIONS = ["SMA 5", "SMA 20", "SMA 50"]
CHART_TEMPLATE = "plotly_white"

# Portfolio Settings
DEFAULT_STOCK_QUANTITY = 10

# Error Handling
MAX_RETRIES = 3
EXPONENTIAL_BACKOFF_BASE = 2

# Logging
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO' 