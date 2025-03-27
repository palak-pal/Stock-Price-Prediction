"""
Sentiment analysis utilities for Stock Prediction App
"""
import logging
import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta

# Check if NLTK is available for advanced sentiment analysis
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available, using basic sentiment analysis")

# Try importing transformers for advanced sentiment analysis
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, using basic sentiment analysis")

class SentimentAnalyzer:
    """Class for analyzing sentiment of news and social media content related to stocks"""
    
    def __init__(self, method: str = 'auto'):
        """
        Initialize the SentimentAnalyzer
        
        Args:
            method (str, optional): Sentiment analysis method to use
                'vader': NLTK's VADER sentiment analyzer (requires NLTK)
                'transformers': Hugging Face Transformers (requires transformers)
                'basic': Simple lexicon-based analyzer
                'auto': Use best available method (default)
        """
        self.method = method
        self.analyzer = None
        self.initialized = False
        
        # Initialize the appropriate analyzer
        if method == 'auto':
            self._initialize_auto()
        elif method == 'vader':
            self._initialize_vader()
        elif method == 'transformers':
            self._initialize_transformers()
        elif method == 'basic':
            self._initialize_basic()
        else:
            logging.warning(f"Unknown method: {method}, using auto")
            self._initialize_auto()
    
    def _initialize_auto(self):
        """Initialize the best available sentiment analyzer"""
        if TRANSFORMERS_AVAILABLE:
            self._initialize_transformers()
        elif NLTK_AVAILABLE:
            self._initialize_vader()
        else:
            self._initialize_basic()
    
    def _initialize_vader(self):
        """Initialize NLTK's VADER sentiment analyzer"""
        if NLTK_AVAILABLE:
            try:
                self.analyzer = SentimentIntensityAnalyzer()
                self.method = 'vader'
                self.initialized = True
                logging.info("Using NLTK VADER for sentiment analysis")
            except Exception as e:
                logging.error(f"Error initializing VADER: {e}")
                self._initialize_basic()
        else:
            logging.warning("NLTK not available for VADER, falling back to basic")
            self._initialize_basic()
    
    def _initialize_transformers(self):
        """Initialize HuggingFace Transformers sentiment analyzer"""
        if TRANSFORMERS_AVAILABLE:
            try:
                self.analyzer = pipeline('sentiment-analysis')
                self.method = 'transformers'
                self.initialized = True
                logging.info("Using Transformers for sentiment analysis")
            except Exception as e:
                logging.error(f"Error initializing Transformers: {e}")
                # Fall back to VADER if available
                if NLTK_AVAILABLE:
                    self._initialize_vader()
                else:
                    self._initialize_basic()
        else:
            logging.warning("Transformers not available, falling back to VADER or basic")
            if NLTK_AVAILABLE:
                self._initialize_vader()
            else:
                self._initialize_basic()
    
    def _initialize_basic(self):
        """Initialize basic lexicon-based sentiment analyzer"""
        try:
            # Load positive and negative word lexicons
            self.positive_words = set([
                'up', 'upward', 'rise', 'rising', 'rose', 'high', 'higher', 'gain', 'gains', 'positive',
                'increase', 'increased', 'increasing', 'grow', 'growing', 'grew', 'growth',
                'strong', 'stronger', 'strength', 'strengthen', 'strengthened', 'strengthening',
                'opportunity', 'opportunities', 'promising', 'profit', 'profitable', 'profitability',
                'outperform', 'outperformed', 'outperforming', 'outperformance', 'beat', 'beats',
                'beating', 'exceed', 'exceeds', 'exceeded', 'exceeding', 'momentum', 'rally',
                'bullish', 'bull', 'buy', 'buying', 'bought', 'recommend', 'recommended', 'recommending',
                'upgrade', 'upgraded', 'upgrading', 'improve', 'improved', 'improving', 'improvement',
                'recovery', 'recovering', 'recovered', 'positive', 'advance', 'advancing', 'advanced',
                'support', 'supporting', 'supported', 'confidence', 'confident', 'optimistic', 'optimism'
            ])
            
            self.negative_words = set([
                'down', 'downward', 'fall', 'falling', 'fell', 'low', 'lower', 'loss', 'losses', 'negative',
                'decrease', 'decreased', 'decreasing', 'shrink', 'shrinking', 'shrank', 'shrinkage',
                'weak', 'weaker', 'weakness', 'weaken', 'weakened', 'weakening', 'threat', 'threats',
                'disappointing', 'loss', 'unprofitable', 'unprofitability', 'underperform', 'underperformed',
                'underperforming', 'underperformance', 'miss', 'missed', 'missing', 'fail', 'failed',
                'failing', 'failure', 'decline', 'declining', 'declined', 'bearish', 'bear', 'sell',
                'selling', 'sold', 'avoid', 'avoided', 'avoiding', 'downgrade', 'downgraded', 'downgrading',
                'worsen', 'worsened', 'worsening', 'deteriorate', 'deteriorated', 'deteriorating',
                'deterioration', 'struggle', 'struggling', 'struggled', 'negative', 'retreat', 'retreating',
                'retreated', 'resistance', 'resisting', 'resisted', 'concern', 'concerned', 'concerning',
                'pessimistic', 'pessimism', 'risk', 'risky', 'danger', 'dangerous', 'warning', 'warn', 'warned'
            ])
            
            self.method = 'basic'
            self.initialized = True
            logging.info("Using basic lexicon-based sentiment analysis")
        except Exception as e:
            logging.error(f"Error initializing basic sentiment analysis: {e}")
            self.initialized = False
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Dictionary with sentiment scores
        """
        if not self.initialized:
            logging.error("Sentiment analyzer not properly initialized")
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        try:
            if self.method == 'vader':
                return self._analyze_vader(text)
            elif self.method == 'transformers':
                return self._analyze_transformers(text)
            elif self.method == 'basic':
                return self._analyze_basic(text)
            else:
                logging.error(f"Unknown method: {self.method}")
                return {
                    'positive': 0.0,
                    'negative': 0.0,
                    'neutral': 1.0,
                    'compound': 0.0
                }
        except Exception as e:
            logging.error(f"Error analyzing sentiment: {e}")
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
    
    def _analyze_vader(self, text: str) -> Dict[str, float]:
        """Analyze using NLTK's VADER"""
        scores = self.analyzer.polarity_scores(text)
        return scores
    
    def _analyze_transformers(self, text: str) -> Dict[str, float]:
        """Analyze using HuggingFace Transformers"""
        # Limit text length to avoid OOM errors
        if len(text) > 512:
            text = text[:512]
            
        result = self.analyzer(text)[0]
        label = result['label'].lower()
        score = result['score']
        
        # Convert to VADER-like format for consistency
        if label == 'positive' or label == 'pos':
            return {
                'positive': score,
                'negative': 1.0 - score,
                'neutral': 0.0,
                'compound': score * 2 - 1  # Scale to [-1, 1]
            }
        elif label == 'negative' or label == 'neg':
            return {
                'positive': 1.0 - score,
                'negative': score,
                'neutral': 0.0,
                'compound': (1.0 - score) * 2 - 1  # Scale to [-1, 1]
            }
        else:  # neutral
            return {
                'positive': 0.5,
                'negative': 0.5,
                'neutral': score,
                'compound': 0.0
            }
    
    def _analyze_basic(self, text: str) -> Dict[str, float]:
        """Basic lexicon-based sentiment analysis"""
        # Tokenize and clean text
        text = text.lower()
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text)
        
        if not words:
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        total_words = len(words)
        
        # Calculate scores
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = 1.0 - (positive_score + negative_score)
        
        # Ensure scores are non-negative
        neutral_score = max(0.0, neutral_score)
        
        # Compound score is the balance of positive vs negative [-1, 1]
        compound = (positive_score - negative_score) / (positive_score + negative_score + 0.001)
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': neutral_score,
            'compound': compound
        }
    
    def analyze_news_items(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of a list of news items
        
        Args:
            news_items (List[Dict[str, Any]]): List of news items with 'title' and optionally 'description'
            
        Returns:
            List[Dict[str, Any]]: News items with added sentiment scores
        """
        result = []
        
        for item in news_items:
            if not isinstance(item, dict):
                continue
                
            # Create a copy of the item
            enriched_item = item.copy()
            
            # Analyze title sentiment
            title_text = item.get('title', '')
            
            # Optionally include description in analysis
            description = item.get('description', '')
            if description:
                analysis_text = f"{title_text}. {description}"
            else:
                analysis_text = title_text
            
            if analysis_text:
                sentiment = self.analyze_text(analysis_text)
                enriched_item['sentiment'] = sentiment
            
            result.append(enriched_item)
        
        return result
    
    def create_sentiment_dataframe(self, news_items: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame with sentiment scores from news items
        
        Args:
            news_items (List[Dict[str, Any]]): List of news items with sentiment scores
            
        Returns:
            pd.DataFrame: DataFrame with sentiment data by date
        """
        if not news_items:
            return pd.DataFrame()
        
        # Extract dates and sentiment scores
        dates = []
        compounds = []
        positives = []
        negatives = []
        neutrals = []
        
        for item in news_items:
            # Check if item has sentiment and published date
            if 'sentiment' in item and 'published' in item:
                try:
                    sentiment = item['sentiment']
                    
                    # Parse date
                    if isinstance(item['published'], str):
                        # Extract just the date part (YYYY-MM-DD)
                        date_str = item['published'].split('T')[0].split(' ')[0]
                        date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    else:
                        date = item['published']
                    
                    dates.append(date)
                    compounds.append(sentiment.get('compound', 0.0))
                    positives.append(sentiment.get('positive', 0.0))
                    negatives.append(sentiment.get('negative', 0.0))
                    neutrals.append(sentiment.get('neutral', 0.0))
                except Exception as e:
                    logging.error(f"Error processing news item for sentiment dataframe: {e}")
        
        if not dates:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Compound': compounds,
            'Positive': positives,
            'Negative': negatives,
            'Neutral': neutrals
        })
        
        # Group by date and calculate averages
        df_grouped = df.groupby('Date').agg({
            'Compound': 'mean',
            'Positive': 'mean',
            'Negative': 'mean',
            'Neutral': 'mean'
        }).reset_index()
        
        return df_grouped
    
    def create_rolling_sentiment(self, sentiment_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Create rolling average sentiment
        
        Args:
            sentiment_df (pd.DataFrame): DataFrame with daily sentiment scores
            window (int, optional): Rolling window size. Defaults to 5.
            
        Returns:
            pd.DataFrame: DataFrame with rolling sentiment
        """
        if sentiment_df.empty:
            return pd.DataFrame()
        
        # Ensure DataFrame is sorted by date
        sentiment_df = sentiment_df.sort_values('Date')
        
        # Calculate rolling averages
        df_rolling = sentiment_df.copy()
        for col in ['Compound', 'Positive', 'Negative', 'Neutral']:
            if col in df_rolling.columns:
                df_rolling[f'{col}_Rolling'] = df_rolling[col].rolling(window=window).mean()
        
        return df_rolling
    
    def calculate_sentiment_signals(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals based on sentiment
        
        Args:
            sentiment_df (pd.DataFrame): DataFrame with sentiment scores
            
        Returns:
            pd.DataFrame: DataFrame with added sentiment signals
        """
        if sentiment_df.empty:
            return pd.DataFrame()
        
        df = sentiment_df.copy()
        
        # Calculate sentiment-based signals
        # Strong positive: compound > 0.5
        df['Signal_Strong_Positive'] = (df['Compound'] > 0.5).astype(int)
        
        # Strong negative: compound < -0.5
        df['Signal_Strong_Negative'] = (df['Compound'] < -0.5).astype(int)
        
        # Sentiment change: significant change in compound score
        df['Compound_Change'] = df['Compound'].diff()
        df['Signal_Sentiment_Change'] = (df['Compound_Change'].abs() > 0.3).astype(int)
        
        # Sentiment trend: consistent positive/negative sentiment over multiple days
        if 'Compound_Rolling' in df.columns:
            df['Signal_Positive_Trend'] = (df['Compound_Rolling'] > 0.3).astype(int)
            df['Signal_Negative_Trend'] = (df['Compound_Rolling'] < -0.3).astype(int)
        
        # Overall sentiment signal: -1 (bearish), 0 (neutral), 1 (bullish)
        df['Sentiment_Signal'] = 0
        
        # Bullish conditions
        bullish_mask = (
            (df['Compound'] > 0.3) | 
            (df['Signal_Strong_Positive'] == 1) | 
            (df.get('Signal_Positive_Trend', 0) == 1)
        )
        df.loc[bullish_mask, 'Sentiment_Signal'] = 1
        
        # Bearish conditions
        bearish_mask = (
            (df['Compound'] < -0.3) | 
            (df['Signal_Strong_Negative'] == 1) | 
            (df.get('Signal_Negative_Trend', 0) == 1)
        )
        df.loc[bearish_mask, 'Sentiment_Signal'] = -1
        
        return df

    def merge_sentiment_with_price(self, 
                                  sentiment_df: pd.DataFrame, 
                                  price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge sentiment data with price data
        
        Args:
            sentiment_df (pd.DataFrame): DataFrame with sentiment data
            price_df (pd.DataFrame): DataFrame with price data
            
        Returns:
            pd.DataFrame: Merged DataFrame
        """
        if sentiment_df.empty or price_df.empty:
            return price_df.copy()
        
        # Ensure date columns are properly formatted
        sentiment_copy = sentiment_df.copy()
        price_copy = price_df.copy()
        
        # Convert Date to datetime if it's not already
        if 'Date' in sentiment_copy.columns and not pd.api.types.is_datetime64_dtype(sentiment_copy['Date']):
            sentiment_copy['Date'] = pd.to_datetime(sentiment_copy['Date'])
            
        if 'Date' in price_copy.columns and not pd.api.types.is_datetime64_dtype(price_copy['Date']):
            price_copy['Date'] = pd.to_datetime(price_copy['Date'])
        
        # Merge DataFrames on Date
        try:
            merged_df = pd.merge(price_copy, sentiment_copy, on='Date', how='left')
        except Exception as e:
            logging.error(f"Error merging sentiment with price data: {e}")
            return price_copy
        
        # Forward fill sentiment data for trading days without news
        for col in ['Compound', 'Positive', 'Negative', 'Neutral', 'Sentiment_Signal']:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(method='ffill')
        
        return merged_df
    
    def analyze_sentiment_impact(self, 
                               merged_df: pd.DataFrame, 
                               lag_days: int = 1,
                               correlation_window: int = 30) -> Dict[str, Any]:
        """
        Analyze the impact of sentiment on price movements
        
        Args:
            merged_df (pd.DataFrame): Merged DataFrame with price and sentiment data
            lag_days (int, optional): Lag days for analysis. Defaults to 1.
            correlation_window (int, optional): Window for rolling correlation. Defaults to 30.
            
        Returns:
            Dict[str, Any]: Dictionary with analysis results
        """
        result = {}
        
        if merged_df.empty or 'Compound' not in merged_df.columns or 'Close' not in merged_df.columns:
            return {"error": "Insufficient data for sentiment impact analysis"}
        
        df = merged_df.copy()
        
        try:
            # Calculate price returns
            df['Return'] = df['Close'].pct_change() * 100
            
            # Create lagged sentiment
            df[f'Compound_Lag_{lag_days}'] = df['Compound'].shift(lag_days)
            
            # Calculate correlation between sentiment and returns
            overall_corr = df['Return'].corr(df[f'Compound_Lag_{lag_days}'])
            result['overall_correlation'] = overall_corr
            
            # Calculate rolling correlation
            df['Rolling_Correlation'] = df['Return'].rolling(correlation_window).corr(df[f'Compound_Lag_{lag_days}'])
            
            # Calculate accuracy of sentiment as a trading signal
            df['Sentiment_Direction'] = np.sign(df[f'Compound_Lag_{lag_days}'])
            df['Return_Direction'] = np.sign(df['Return'])
            
            # Calculate accuracy (only for non-zero sentiment)
            accuracy_df = df[(df['Sentiment_Direction'] != 0) & df['Return_Direction'].notna()]
            correct_predictions = (accuracy_df['Sentiment_Direction'] == accuracy_df['Return_Direction']).sum()
            total_predictions = len(accuracy_df)
            
            result['prediction_accuracy'] = round(correct_predictions / total_predictions * 100, 2) if total_predictions > 0 else 0
            result['total_predictions'] = total_predictions
            
            # Calculate impact by sentiment strength
            strong_positive = df[df[f'Compound_Lag_{lag_days}'] > 0.5]['Return'].mean()
            moderate_positive = df[(df[f'Compound_Lag_{lag_days}'] > 0.2) & (df[f'Compound_Lag_{lag_days}'] <= 0.5)]['Return'].mean()
            neutral = df[(df[f'Compound_Lag_{lag_days}'] >= -0.2) & (df[f'Compound_Lag_{lag_days}'] <= 0.2)]['Return'].mean()
            moderate_negative = df[(df[f'Compound_Lag_{lag_days}'] < -0.2) & (df[f'Compound_Lag_{lag_days}'] >= -0.5)]['Return'].mean()
            strong_negative = df[df[f'Compound_Lag_{lag_days}'] < -0.5]['Return'].mean()
            
            result['return_by_sentiment'] = {
                'strong_positive': round(strong_positive, 2) if not pd.isna(strong_positive) else 0,
                'moderate_positive': round(moderate_positive, 2) if not pd.isna(moderate_positive) else 0,
                'neutral': round(neutral, 2) if not pd.isna(neutral) else 0,
                'moderate_negative': round(moderate_negative, 2) if not pd.isna(moderate_negative) else 0,
                'strong_negative': round(strong_negative, 2) if not pd.isna(strong_negative) else 0
            }
            
            # Recent sentiment trend
            recent_sentiment = df['Compound'].tail(5).mean()
            result['recent_sentiment'] = round(recent_sentiment, 2)
            
            # Sentiment volatility
            sentiment_volatility = df['Compound'].std()
            result['sentiment_volatility'] = round(sentiment_volatility, 2)
            
            return result
        
        except Exception as e:
            logging.error(f"Error analyzing sentiment impact: {e}")
            return {"error": str(e)}

def preprocess_news_data(news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preprocess news data for sentiment analysis
    
    Args:
        news_data (List[Dict[str, Any]]): Raw news data
        
    Returns:
        List[Dict[str, Any]]: Preprocessed news data
    """
    if not news_data:
        return []
    
    processed_data = []
    
    for item in news_data:
        if not isinstance(item, dict):
            continue
        
        processed_item = {}
        
        # Extract essential fields
        for field in ['title', 'description', 'content', 'published', 'url', 'source']:
            if field in item:
                processed_item[field] = item[field]
        
        # Ensure 'published' is properly formatted
        if 'published' in processed_item and isinstance(processed_item['published'], str):
            try:
                # Try to parse datetime
                date_obj = datetime.strptime(processed_item['published'].split('T')[0].split(' ')[0], '%Y-%m-%d')
                processed_item['published'] = date_obj.strftime('%Y-%m-%d')
            except Exception:
                # Keep original if parsing fails
                pass
        
        # Add to processed data
        if processed_item and 'title' in processed_item:
            processed_data.append(processed_item)
    
    return processed_data

def filter_news_by_relevance(news_data: List[Dict[str, Any]], 
                          ticker: str, 
                          keywords: List[str] = None,
                          min_date: Optional[str] = None,
                          max_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Filter news data by relevance to a specific stock ticker
    
    Args:
        news_data (List[Dict[str, Any]]): News data
        ticker (str): Stock ticker symbol
        keywords (List[str], optional): Additional keywords for filtering. Defaults to None.
        min_date (Optional[str], optional): Minimum date for filtering (YYYY-MM-DD). Defaults to None.
        max_date (Optional[str], optional): Maximum date for filtering (YYYY-MM-DD). Defaults to None.
        
    Returns:
        List[Dict[str, Any]]: Filtered news data
    """
    if not news_data:
        return []
    
    # Prepare ticker and keywords for matching
    ticker = ticker.upper()
    ticker_variations = [ticker, f"${ticker}"]
    
    # Add common variations like "Apple Inc" for "AAPL"
    company_names = {
        'AAPL': ['Apple', 'Apple Inc'],
        'MSFT': ['Microsoft', 'Microsoft Corp'],
        'AMZN': ['Amazon', 'Amazon.com'],
        'GOOGL': ['Google', 'Alphabet'],
        'FB': ['Facebook', 'Meta'],
        'TSLA': ['Tesla', 'Tesla Inc'],
        'NVDA': ['Nvidia', 'Nvidia Corp'],
        'JPM': ['JPMorgan', 'JP Morgan', 'J.P. Morgan'],
        'V': ['Visa', 'Visa Inc'],
        'JNJ': ['Johnson & Johnson', 'Johnson and Johnson'],
        'WMT': ['Walmart', 'Wal-Mart'],
        'BAC': ['Bank of America', 'BofA'],
        'PG': ['Procter & Gamble', 'P&G'],
        'MA': ['Mastercard', 'MasterCard Inc'],
        'DIS': ['Disney', 'Walt Disney'],
        'NFLX': ['Netflix', 'Netflix Inc'],
        'XOM': ['Exxon', 'Exxon Mobil', 'ExxonMobil']
    }
    
    if ticker in company_names:
        ticker_variations.extend(company_names[ticker])
    
    # Add all keywords to check
    all_keywords = ticker_variations.copy()
    if keywords:
        all_keywords.extend(keywords)
    
    # Prepare date filters
    min_date_obj = None
    max_date_obj = None
    
    if min_date:
        try:
            min_date_obj = datetime.strptime(min_date, '%Y-%m-%d').date()
        except Exception as e:
            logging.error(f"Error parsing min_date: {e}")
    
    if max_date:
        try:
            max_date_obj = datetime.strptime(max_date, '%Y-%m-%d').date()
        except Exception as e:
            logging.error(f"Error parsing max_date: {e}")
    
    # Filter news
    filtered_news = []
    
    for item in news_data:
        if not isinstance(item, dict):
            continue
        
        # Check date filters
        if 'published' in item:
            try:
                # Parse date from published field
                if isinstance(item['published'], str):
                    date_str = item['published'].split('T')[0].split(' ')[0]
                    item_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                else:
                    item_date = item['published'].date() if hasattr(item['published'], 'date') else None
                
                # Skip if outside date range
                if min_date_obj and item_date < min_date_obj:
                    continue
                if max_date_obj and item_date > max_date_obj:
                    continue
            except Exception as e:
                logging.debug(f"Error parsing date: {e}")
        
        # Check if any keywords are in title or description
        is_relevant = False
        title = item.get('title', '').lower()
        description = item.get('description', '').lower()
        content = item.get('content', '').lower()
        
        for keyword in all_keywords:
            if keyword.lower() in title or keyword.lower() in description or keyword.lower() in content:
                is_relevant = True
                break
        
        if is_relevant:
            filtered_news.append(item)
    
    return filtered_news

def check_sentiment_forecast_signals(sentiment_df: pd.DataFrame, threshold: float = 0.3) -> Dict[str, Any]:
    """
    Check for trading signals based on sentiment forecast
    
    Args:
        sentiment_df (pd.DataFrame): DataFrame with sentiment data
        threshold (float, optional): Sentiment threshold for signals. Defaults to 0.3.
        
    Returns:
        Dict[str, Any]: Dictionary with signal information
    """
    if sentiment_df.empty or 'Compound' not in sentiment_df.columns:
        return {'signal': 'neutral', 'strength': 0, 'explanation': 'Insufficient sentiment data'}
    
    # Get the most recent sentiment
    recent_rows = sentiment_df.tail(5)
    latest_sentiment = recent_rows['Compound'].iloc[-1] if not recent_rows.empty else 0
    
    # Calculate sentiment trend
    sentiment_trend = 0
    if len(recent_rows) >= 3:
        first_half = recent_rows['Compound'].iloc[:-2].mean()
        second_half = recent_rows['Compound'].iloc[-2:].mean()
        sentiment_trend = second_half - first_half
    
    # Determine signal strength (0-100)
    signal_strength = min(100, abs(latest_sentiment * 100))
    
    # Generate signal
    if latest_sentiment > threshold:
        signal = 'bullish'
        if sentiment_trend > 0:
            signal_type = 'strengthening'
            explanation = f"Bullish sentiment ({latest_sentiment:.2f}) with positive trend"
        else:
            signal_type = 'weakening'
            explanation = f"Bullish sentiment ({latest_sentiment:.2f}) but potentially weakening"
    elif latest_sentiment < -threshold:
        signal = 'bearish'
        if sentiment_trend < 0:
            signal_type = 'strengthening'
            explanation = f"Bearish sentiment ({latest_sentiment:.2f}) with negative trend"
        else:
            signal_type = 'weakening'
            explanation = f"Bearish sentiment ({latest_sentiment:.2f}) but potentially improving"
    else:
        signal = 'neutral'
        signal_type = 'stable'
        explanation = f"Neutral sentiment ({latest_sentiment:.2f}) without strong direction"
    
    return {
        'signal': signal,
        'signal_type': signal_type,
        'strength': int(signal_strength),
        'latest_sentiment': round(latest_sentiment, 2),
        'sentiment_trend': round(sentiment_trend, 2),
        'explanation': explanation
    } 