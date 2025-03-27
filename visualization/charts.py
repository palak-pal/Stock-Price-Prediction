"""
Chart creation utilities for Stock Prediction App
"""
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Union, Any, Tuple
from datetime import datetime

def create_prediction_chart(df: pd.DataFrame, 
                           future_dates: List[datetime], 
                           predictions: np.ndarray, 
                           upper_bounds: np.ndarray, 
                           lower_bounds: np.ndarray, 
                           ticker: str, 
                           chart_type: str = 'line',
                           show_volume: bool = False) -> go.Figure:
    """
    Create an interactive Plotly chart with historical data and predictions
    
    Args:
        df (pd.DataFrame): DataFrame with historical data
        future_dates (List[datetime]): List of future dates
        predictions (np.ndarray): Array of predicted values
        upper_bounds (np.ndarray): Array of upper bound values
        lower_bounds (np.ndarray): Array of lower bound values
        ticker (str): Stock ticker symbol
        chart_type (str, optional): Type of chart ('line' or 'candlestick'). Defaults to 'line'.
        show_volume (bool, optional): Whether to show volume data. Defaults to False.
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Convert dates to strings for plotting
    future_dates_str = [date.strftime('%Y-%m-%d') for date in future_dates]
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    if chart_type.lower() == 'candlestick':
        fig.add_trace(
            go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Historical',
                increasing_line_color='green',
                decreasing_line_color='red'
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            )
        )
    
    # Add predictions
    fig.add_trace(
        go.Scatter(
            x=future_dates_str,
            y=predictions,
            mode='lines',
            name='Prediction',
            line=dict(color='red', dash='dash', width=2)
        )
    )
    
    # Add confidence intervals
    fig.add_trace(
        go.Scatter(
            x=future_dates_str,
            y=upper_bounds,
            mode='lines',
            name='Upper Bound',
            line=dict(color='rgba(0,255,0,0.3)', dash='dot'),
            hoverinfo='skip'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates_str,
            y=lower_bounds,
            mode='lines',
            name='Lower Bound',
            line=dict(color='rgba(255,0,0,0.3)', dash='dot'),
            fill='tonexty',  # Fill area between upper and lower bounds
            fillcolor='rgba(68, 68, 68, 0.2)',
            hoverinfo='skip'
        )
    )
    
    # Add volume data if requested
    if show_volume and 'Volume' in df.columns:
        # Create secondary y-axis for volume
        fig.add_trace(
            go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker=dict(color='rgba(100, 100, 250, 0.3)'),
                opacity=0.4,
                yaxis='y2'
            )
        )
        
        # Update layout to include secondary y-axis
        fig.update_layout(
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                showgrid=False
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template='plotly_white',
        xaxis=dict(
            rangeslider=dict(visible=False),
            type='date'
        )
    )
    
    # Add range buttons for time selection
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {
                    'args': [{'xaxis.range': [df['Date'].iloc[-30], future_dates_str[-1]]}],
                    'label': '1M',
                    'method': 'relayout'
                },
                {
                    'args': [{'xaxis.range': [df['Date'].iloc[-90], future_dates_str[-1]]}],
                    'label': '3M',
                    'method': 'relayout'
                },
                {
                    'args': [{'xaxis.range': [df['Date'].iloc[-180], future_dates_str[-1]]}],
                    'label': '6M',
                    'method': 'relayout'
                },
                {
                    'args': [{'xaxis.range': [df['Date'].iloc[0], future_dates_str[-1]]}],
                    'label': 'All',
                    'method': 'relayout'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': True,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.1,
            'yanchor': 'top'
        }]
    )
    
    return fig

def add_technical_indicators(fig: go.Figure, df: pd.DataFrame, selected_indicators: List[str]) -> go.Figure:
    """
    Add technical indicators to the chart
    
    Args:
        fig (go.Figure): Plotly figure object
        df (pd.DataFrame): DataFrame with technical indicators
        selected_indicators (List[str]): List of indicators to show
        
    Returns:
        go.Figure: Updated Plotly figure
    """
    # Define indicator colors
    colors = {
        'SMA_5': 'purple',
        'SMA_20': 'orange',
        'SMA_50': 'green',
        'EMA_5': 'cyan',
        'EMA_20': 'magenta',
        'BB_Upper': 'rgba(0,128,0,0.3)',
        'BB_Middle': 'rgba(0,128,0,0.6)',
        'BB_Lower': 'rgba(0,128,0,0.3)',
        'RSI': 'brown',
        'MACD': 'blue',
        'MACD_Signal': 'red',
        'ATR': 'gray'
    }
    
    # Add selected indicators
    for indicator in selected_indicators:
        if indicator in df.columns:
            if indicator.startswith('SMA') or indicator.startswith('EMA'):
                # Add moving averages to main chart
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df[indicator],
                        mode='lines',
                        name=indicator.replace('_', ' '),
                        line=dict(color=colors.get(indicator, 'gray'), width=1.5)
                    )
                )
            elif indicator.startswith('BB_'):
                # Add Bollinger Bands to main chart
                if indicator == 'BB_Upper':
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df[indicator],
                            mode='lines',
                            name='Bollinger Upper',
                            line=dict(color=colors.get(indicator, 'gray'), width=1, dash='dot')
                        )
                    )
                elif indicator == 'BB_Middle':
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df[indicator],
                            mode='lines',
                            name='Bollinger Middle',
                            line=dict(color=colors.get(indicator, 'gray'), width=1)
                        )
                    )
                elif indicator == 'BB_Lower':
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df[indicator],
                            mode='lines',
                            name='Bollinger Lower',
                            line=dict(color=colors.get(indicator, 'gray'), width=1, dash='dot'),
                            fill='tonexty',
                            fillcolor='rgba(0,128,0,0.1)'
                        )
                    )
            elif indicator == 'RSI':
                # Add RSI subplot
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df[indicator],
                        mode='lines',
                        name='RSI',
                        line=dict(color=colors.get(indicator, 'brown')),
                        yaxis='y3'
                    )
                )
                
                # Add horizontal lines at 70 and 30
                fig.add_trace(
                    go.Scatter(
                        x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                        y=[70, 70],
                        mode='lines',
                        name='RSI Overbought',
                        line=dict(color='red', width=1, dash='dash'),
                        yaxis='y3'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                        y=[30, 30],
                        mode='lines',
                        name='RSI Oversold',
                        line=dict(color='green', width=1, dash='dash'),
                        yaxis='y3'
                    )
                )
                
                # Update layout to include RSI subplot
                fig.update_layout(
                    yaxis3=dict(
                        title='RSI',
                        overlaying='y',
                        side='right',
                        showgrid=False,
                        range=[0, 100]
                    )
                )
            elif indicator.startswith('MACD'):
                # Add MACD subplot
                if indicator == 'MACD':
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df[indicator],
                            mode='lines',
                            name='MACD',
                            line=dict(color=colors.get(indicator, 'blue')),
                            yaxis='y4'
                        )
                    )
                elif indicator == 'MACD_Signal':
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df[indicator],
                            mode='lines',
                            name='MACD Signal',
                            line=dict(color=colors.get(indicator, 'red')),
                            yaxis='y4'
                        )
                    )
                elif indicator == 'MACD_Hist':
                    # Add MACD histogram
                    fig.add_trace(
                        go.Bar(
                            x=df['Date'],
                            y=df[indicator],
                            name='MACD Histogram',
                            marker=dict(
                                color=np.where(df[indicator] >= 0, 'green', 'red')
                            ),
                            yaxis='y4'
                        )
                    )
                
                # Update layout to include MACD subplot
                fig.update_layout(
                    yaxis4=dict(
                        title='MACD',
                        anchor='free',
                        overlaying='y',
                        side='right',
                        position=0.95,
                        showgrid=False
                    )
                )
    
    return fig

def create_correlation_matrix_chart(correlation_matrix: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap visualization of a correlation matrix
    
    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Create heatmap
    fig = px.imshow(
        correlation_matrix,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        text_auto='.2f'
    )
    
    fig.update_layout(
        title='Stock Correlation Matrix',
        xaxis_title='',
        yaxis_title='',
        coloraxis_colorbar=dict(
            title='Correlation',
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1: Perfect negative', '-0.5', '0: No correlation', '0.5', '1: Perfect positive']
        )
    )
    
    return fig

def create_sentiment_chart(news_items: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a chart visualizing sentiment from news data
    
    Args:
        news_items (List[Dict[str, Any]]): List of news items with sentiment scores
        
    Returns:
        go.Figure: Plotly figure object
    """
    if not news_items:
        return None
    
    # Extract dates and sentiment scores
    dates = []
    positive_scores = []
    negative_scores = []
    neutral_scores = []
    compound_scores = []
    titles = []
    
    for item in news_items:
        if 'published' in item and 'sentiment' in item:
            try:
                # Convert string date to datetime if needed
                if isinstance(item['published'], str):
                    date = item['published'].split(' ')[0]  # Get just the date part
                else:
                    date = item['published']
                
                dates.append(date)
                sentiment = item['sentiment']
                positive_scores.append(sentiment.get('positive', 0))
                negative_scores.append(sentiment.get('negative', 0))
                neutral_scores.append(sentiment.get('neutral', 0))
                compound_scores.append(sentiment.get('compound', 0))
                titles.append(item.get('title', 'No title'))
            except Exception as e:
                logging.error(f"Error processing news item for sentiment chart: {e}")
    
    # Create dataframe
    df = pd.DataFrame({
        'Date': dates,
        'Positive': positive_scores,
        'Negative': negative_scores,
        'Neutral': neutral_scores,
        'Compound': compound_scores,
        'Title': titles
    })
    
    # Create figure
    fig = go.Figure()
    
    # Add sentiment scores
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Positive'],
            name='Positive',
            marker_color='green',
            hovertext=df['Title']
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Negative'],
            name='Negative',
            marker_color='red',
            hovertext=df['Title']
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Neutral'],
            name='Neutral',
            marker_color='gray',
            hovertext=df['Title']
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Compound'],
            mode='lines+markers',
            name='Compound Score',
            line=dict(color='blue', width=2),
            hovertext=df['Title']
        )
    )
    
    # Update layout
    fig.update_layout(
        title='News Sentiment Analysis',
        barmode='stack',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig

def create_anomaly_detection_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a chart highlighting anomalies in stock price data
    
    Args:
        df (pd.DataFrame): DataFrame with anomaly flags
        
    Returns:
        go.Figure: Plotly figure object
    """
    if 'Is_Anomaly' not in df.columns or 'Z_Score' not in df.columns:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue')
        )
    )
    
    # Add anomaly points
    anomalies = df[df['Is_Anomaly']]
    
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies['Date'],
                y=anomalies['Close'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='circle',
                    line=dict(width=2, color='black')
                ),
                hoverinfo='text',
                hovertext=[f"Date: {date}<br>Close: ${close:.2f}<br>Z-Score: {z:.2f}" 
                           for date, close, z in zip(anomalies['Date'], anomalies['Close'], anomalies['Z_Score'])]
            )
        )
    
    # Add Z-score subplot
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Z_Score'],
            mode='lines',
            name='Z-Score',
            line=dict(color='purple'),
            yaxis='y2'
        )
    )
    
    # Add threshold lines
    threshold = 2.0  # Default threshold
    fig.add_trace(
        go.Scatter(
            x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
            y=[threshold, threshold],
            mode='lines',
            name=f'Threshold (+{threshold})',
            line=dict(color='red', width=1, dash='dash'),
            yaxis='y2'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
            y=[-threshold, -threshold],
            mode='lines',
            name=f'Threshold (-{threshold})',
            line=dict(color='red', width=1, dash='dash'),
            yaxis='y2'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Stock Price Anomaly Detection',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        yaxis2=dict(
            title='Z-Score',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_model_comparison_chart(model_results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Create a chart comparing predictions from different models
    
    Args:
        model_results (Dict[str, Dict[str, Any]]): Dictionary of model results
        
    Returns:
        go.Figure: Plotly figure object
    """
    if not model_results:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add predictions from each model
    for model_name, result in model_results.items():
        # Get prediction data
        future_dates = result.get('future_dates', [])
        predictions = result.get('predictions', [])
        
        if not future_dates or not predictions.any():
            continue
        
        # Convert dates to strings for plotting
        future_dates_str = [date.strftime('%Y-%m-%d') for date in future_dates]
        
        # Add predictions
        fig.add_trace(
            go.Scatter(
                x=future_dates_str,
                y=predictions,
                mode='lines',
                name=f'{model_name.capitalize()} Prediction',
                line=dict(width=2)
            )
        )
        
        # Add execution time annotation
        execution_time = result.get('execution_time', 0)
        fig.add_annotation(
            x=future_dates_str[-1],
            y=predictions[-1],
            text=f"{execution_time:.2f}s",
            showarrow=True,
            arrowhead=7,
            ax=50,
            ay=0
        )
    
    # Update layout
    fig.update_layout(
        title='Model Comparison',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_what_if_scenario_chart(df: pd.DataFrame, 
                                 base_prediction: np.ndarray,
                                 scenario_predictions: Dict[str, np.ndarray],
                                 future_dates: List[datetime]) -> go.Figure:
    """
    Create a chart comparing different what-if scenarios
    
    Args:
        df (pd.DataFrame): DataFrame with historical data
        base_prediction (np.ndarray): Base case prediction
        scenario_predictions (Dict[str, np.ndarray]): Dictionary of scenario predictions
        future_dates (List[datetime]): List of future dates
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        )
    )
    
    # Convert dates to strings for plotting
    future_dates_str = [date.strftime('%Y-%m-%d') for date in future_dates]
    
    # Add base prediction
    fig.add_trace(
        go.Scatter(
            x=future_dates_str,
            y=base_prediction,
            mode='lines',
            name='Base Prediction',
            line=dict(color='black', dash='dash', width=2)
        )
    )
    
    # Add scenario predictions
    colors = px.colors.qualitative.Plotly
    for i, (scenario_name, predictions) in enumerate(scenario_predictions.items()):
        fig.add_trace(
            go.Scatter(
                x=future_dates_str,
                y=predictions,
                mode='lines',
                name=scenario_name,
                line=dict(color=colors[i % len(colors)], width=2)
            )
        )
    
    # Update layout
    fig.update_layout(
        title='What-If Scenario Analysis',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig 