# Real-Time Stock Price Prediction App

A powerful Streamlit web application for predicting stock prices using multiple machine learning models.

## Features

- **Stock Ticker Selection**: Choose from a dynamically loaded list of stocks (S&P 500)
- **Multiple Prediction Models**: 
  - Linear Regression
  - LSTM Neural Network
  - Facebook Prophet
- **Interactive Visualizations**: Beautiful Plotly charts with customizable options
- **Real-Time Data**: Fetches latest stock data from Yahoo Finance
- **News Integration**: Displays recent news about the selected stock
- **Export Functionality**: Download prediction results as CSV
- **Robust Error Handling**: Fallback mechanisms when API calls fail

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Stock-Prediction-App
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   
   Note: TensorFlow and Prophet are optional dependencies. If not installed, the app will fall back to Linear Regression.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Select your desired stock, date range, prediction model, and visualization options in the sidebar

4. Explore the predictions and download the results if needed

5. Review : https://stock-price-prediction-1-x6lw.onrender.com/



## Requirements

- Python 3.7+
- Libraries: streamlit, yfinance, pandas, numpy, plotly, scikit-learn, etc. (see requirements.txt)


# Stock-Price-Prediction



