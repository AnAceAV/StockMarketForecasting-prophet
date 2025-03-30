import streamlit as st
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error

# Function to fetch stock data
def get_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError(f"No data found for {ticker} in the given date range.")
        stock_data.reset_index(inplace=True)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Function to forecast stock data with hyperparameter tuning
# Function to forecast stock data with hyperparameter tuning
def forecast_stock_data(stock_data, periods):
    try:
        if 'Date' not in stock_data.columns or 'Close' not in stock_data.columns:
            raise ValueError("Input DataFrame must contain 'Date' and 'Close' columns.")
        
        df = stock_data[['Date', 'Close']]
        df.columns = ['ds', 'y']
        
        # Validate and preprocess the data
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df['ds'] = df['ds'].dt.tz_localize(None)  # Remove timezone information
        if df['ds'].isnull().any():
            raise ValueError("Invalid dates found in the 'Date' column.")
        if df['y'].isnull().any():
            raise ValueError("Missing values found in the 'Close' column.")
        
        # Apply log transformation
        df['y'] = df['y'].apply(lambda x: np.log(x) if x > 0 else np.nan)
        if df['y'].isnull().any():
            raise ValueError("Log transformation resulted in NaN values. Ensure all 'Close' values are positive.")
        
        # Create and configure the Prophet model
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True,
                        changepoint_prior_scale=0.05, seasonality_mode='multiplicative')
        model.add_country_holidays(country_name='US')
        model.fit(df)
        
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Reverse log transformation
        forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].apply(np.exp)
        
        return forecast, model
    except Exception as e:
        st.error(f"Error during forecasting: {e}")
        return pd.DataFrame(), None

# Function to evaluate model
def evaluate_model(model, df):
    try:
        df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
        df_p = performance_metrics(df_cv)
        return df_p
    except Exception as e:
        st.error(f"Error during model evaluation: {e}")
        return pd.DataFrame()

# Function to calculate RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

# Main application content
def main_app():
    st.title('Stock Market Forecasting')

    # User inputs
    ticker = st.text_input('Enter stock ticker', 'AAPL')
    start_date = st.date_input('Start date', pd.to_datetime('2014-01-01'))
    end_date = st.date_input('End date', pd.to_datetime('2023-01-01'))
    forecast_days = st.slider('Days to forecast', 1, 365, 30)

    # Fetch and display stock data
    if ticker:
        stock_data = get_stock_data(ticker, start_date, end_date)
        if not stock_data.empty:
            st.write(f"Showing data for {ticker} from {start_date} to {end_date}")
            st.dataframe(stock_data)

            # Plot stock data
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Close'))
            fig.update_layout(title='Stock Price Over Time', xaxis_title='Date', yaxis_title='Close Price')
            st.plotly_chart(fig)

            # Forecast and display forecast data
            if st.button('Forecast'):
                forecast_data, model = forecast_stock_data(stock_data, forecast_days)
                if not forecast_data.empty:
                    st.write(f"Forecasting {forecast_days} days ahead")
                    st.dataframe(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

                    # Evaluate model
                    evaluation = evaluate_model(model, stock_data)
                    if not evaluation.empty:
                        st.write('Model Evaluation Metrics:')
                        st.dataframe(evaluation)
                    
                    # Calculate and display RMSE
                    actual = stock_data['Close']
                    predicted = forecast_data['yhat'][:len(actual)]
                    rmse = calculate_rmse(actual, predicted)
                    st.markdown(f'**Root Mean Squared Error (RMSE):** {rmse:.2f}', unsafe_allow_html=True)

                    # Plot forecast data
                    forecast_fig = go.Figure()
                    forecast_fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat'], mode='lines', name='Forecast'))
                    forecast_fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat_lower'], mode='lines', name='Lower Confidence Interval', line=dict(dash='dash')))
                    forecast_fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat_upper'], mode='lines', name='Upper Confidence Interval', line=dict(dash='dash')))
                    forecast_fig.update_layout(title=f'{ticker} Stock Price Forecast', xaxis_title='Date', yaxis_title='Forecasted Price')
                    st.plotly_chart(forecast_fig)
                    
                    # Evaluate accuracy by plotting actual vs forecasted
                    eval_fig = go.Figure()
                    eval_fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Actual'))
                    eval_fig.add_trace(go.Scatter(x=forecast_data['ds'][:len(actual)], y=predicted, mode='lines', name='Forecast'))
                    eval_fig.update_layout(title=f'{ticker} Actual vs Forecasted', xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(eval_fig)

if __name__ == "__main__":
    main_app()
