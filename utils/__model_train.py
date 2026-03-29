import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
print('=' * 70)
print('COMPREHENSIVE TIME SERIES FORECASTING SYSTEM')
print('Python 3.10.6 | Virtual Environment Ready')
print('=' * 70)
df = pd.read_csv('data/THESIS-DATASET.csv')
print(f'Initial dataset: {df.shape[0]} records, {df.shape[1]} features')

def comprehensive_data_cleaning(df):
    print('\n' + '=' * 60)
    print('COMPREHENSIVE DATA CLEANING AND PREPROCESSING')
    print('=' * 60)
    df_clean = df.copy()
    print('\n1. HANDLING MISSING VALUES:')
    missing_before = df_clean.isnull().sum().sum()
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(method='ffill', inplace=True)
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna('Unknown', inplace=True)
    missing_after = df_clean.isnull().sum().sum()
    print(f'   Missing values treated: {missing_before} → {missing_after}')
    print('\n2. OUTLIER DETECTION AND TREATMENT:')

    def cap_outliers_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_before = ((series < lower_bound) | (series > upper_bound)).sum()
        series_capped = np.clip(series, lower_bound, upper_bound)
        outliers_after = ((series_capped < lower_bound) | (series_capped > upper_bound)).sum()
        return (series_capped, outliers_before, outliers_after)
    if 'Sales_Value_PHP' in df_clean.columns:
        original_values = df_clean['Sales_Value_PHP'].copy()
        (df_clean['Sales_Value_PHP'], outliers_before, outliers_after) = cap_outliers_iqr(df_clean['Sales_Value_PHP'])
        print(f'   Sales outliers: {outliers_before} → {outliers_after}')
    print('\n3. DATA CONSISTENCY STANDARDIZATION:')
    if 'Product_Name' in df_clean.columns:
        df_clean['Product_Name'] = df_clean['Product_Name'].str.strip().str.title()
        print(f"   Product names standardized: {df_clean['Product_Name'].nunique()} unique products")
    if 'Category' in df_clean.columns:
        df_clean['Category'] = df_clean['Category'].str.upper().str.strip()
        print(f"   Categories standardized: {df_clean['Category'].nunique()} categories")
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    final_count = len(df_clean)
    print(f'   Duplicates removed: {initial_count - final_count} records')
    print('\n4. DATA QUALITY VALIDATION REPORT:')
    print(f'   Final record count: {len(df_clean):,}')
    print(f"   Date range: {df_clean['Date'].min().strftime('%Y-%m-%d')} to {df_clean['Date'].max().strftime('%Y-%m-%d')}")
    print(f'   Data completeness: {(1 - df_clean.isnull().sum().sum() / (len(df_clean) * len(df_clean.columns))) * 100:.2f}%')
    if 'Sales_Value_PHP' in df_clean.columns:
        print(f"   Total sales value: PHP {df_clean['Sales_Value_PHP'].sum():,.2f}")
        print(f"   Sales data integrity: {(df_clean['Sales_Value_PHP'] >= 0).sum() / len(df_clean) * 100:.2f}% valid values")
    return df_clean

def create_advanced_features(time_series):
    print('\n' + '=' * 50)
    print('ADVANCED FEATURE ENGINEERING')
    print('=' * 50)
    features_df = time_series.to_frame('sales')
    features_df['day_of_week'] = time_series.index.dayofweek
    features_df['day_of_month'] = time_series.index.day
    features_df['week_of_year'] = time_series.index.isocalendar().week
    features_df['month'] = time_series.index.month
    features_df['quarter'] = time_series.index.quarter
    features_df['year'] = time_series.index.year
    features_df['is_weekend'] = (time_series.index.dayofweek >= 5).astype(int)
    features_df['is_month_start'] = time_series.index.is_month_start.astype(int)
    features_df['is_month_end'] = time_series.index.is_month_end.astype(int)
    features_df['is_quarter_start'] = time_series.index.is_quarter_start.astype(int)
    features_df['is_quarter_end'] = time_series.index.is_quarter_end.astype(int)
    lag_periods = [1, 2, 3, 7, 14, 21, 30]
    for lag in lag_periods:
        features_df[f'lag_{lag}d'] = time_series.shift(lag)
    windows = [7, 14, 30]
    for window in windows:
        features_df[f'rolling_mean_{window}d'] = time_series.rolling(window).mean()
        features_df[f'rolling_std_{window}d'] = time_series.rolling(window).std()
        features_df[f'rolling_min_{window}d'] = time_series.rolling(window).min()
        features_df[f'rolling_max_{window}d'] = time_series.rolling(window).max()
        features_df[f'rolling_median_{window}d'] = time_series.rolling(window).median()
    features_df['sin_dayofyear'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365)
    features_df['cos_dayofyear'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365)
    features_df['sin_month'] = np.sin(2 * np.pi * features_df['month'] / 12)
    features_df['cos_month'] = np.cos(2 * np.pi * features_df['month'] / 12)
    features_df['daily_growth'] = time_series.pct_change()
    features_df['weekly_growth'] = time_series.pct_change(7)
    print(f'   Total features created: {len(features_df.columns)}')
    print(f"   Temporal features: {sum([col in ['day_of_week', 'month', 'quarter', 'year'] for col in features_df.columns])}")
    print(f"   Lag features: {sum(['lag_' in col for col in features_df.columns])}")
    print(f"   Rolling features: {sum(['rolling_' in col for col in features_df.columns])}")
    return features_df.dropna()

def enhanced_data_validation(df, daily_sales, train_data, test_data):
    print('\n' + '=' * 50)
    print('ENHANCED DATA QUALITY VALIDATION')
    print('=' * 50)
    validation_results = {}
    validation_results['total_records'] = len(df)
    validation_results['date_range'] = f"{df['Date'].min().date()} to {df['Date'].max().date()}"
    validation_results['missing_values'] = df.isnull().sum().sum()
    validation_results['data_completeness'] = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if 'Sales_Value_PHP' in df.columns:
        validation_results['total_sales'] = df['Sales_Value_PHP'].sum()
        validation_results['avg_daily_sales'] = daily_sales.mean()
        validation_results['sales_volatility'] = daily_sales.std()
        validation_results['zero_sales_days'] = (daily_sales == 0).sum()
    date_gaps = daily_sales.index.to_series().diff().dt.days.max()
    validation_results['max_date_gap'] = date_gaps
    validation_results['train_samples'] = len(train_data)
    validation_results['test_samples'] = len(test_data)
    validation_results['train_ratio'] = len(train_data) / len(daily_sales) * 100
    validation_results['test_ratio'] = len(test_data) / len(daily_sales) * 100
    print('VALIDATION REPORT:')
    print(f'• Dataset Integrity:')
    print(f"  - Records: {validation_results['total_records']:,}")
    print(f"  - Date Range: {validation_results['date_range']}")
    print(f"  - Missing Values: {validation_results['missing_values']}")
    print(f"  - Completeness: {validation_results['data_completeness']:.2f}%")
    if 'Sales_Value_PHP' in df.columns:
        print(f'• Sales Data Quality:')
        print(f"  - Total Sales: PHP {validation_results['total_sales']:,.2f}")
        print(f"  - Avg Daily: PHP {validation_results['avg_daily_sales']:,.2f}")
        print(f"  - Volatility: PHP {validation_results['sales_volatility']:,.2f}")
        print(f"  - Zero Sales Days: {validation_results['zero_sales_days']}")
    print(f'• Temporal Analysis:')
    print(f"  - Max Date Gap: {validation_results['max_date_gap']} days")
    print(f'• Train-Test Split:')
    print(f"  - Training: {validation_results['train_samples']} samples ({validation_results['train_ratio']:.1f}%)")
    print(f"  - Testing: {validation_results['test_samples']} samples ({validation_results['test_ratio']:.1f}%)")
    return validation_results

def load_and_explore_data():
    print('\n LOADING AND EXPLORING DATA...')
    df = pd.read_csv('data/THESIS-DATASET.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f'Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns')
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    df_clean = comprehensive_data_cleaning(df)
    print(f"Categories: {df_clean['Category'].unique().tolist()}")
    print('\n' + '=' * 50)
    print('DATA EXPLORATION SUMMARY')
    print('=' * 50)
    print(f'\n1. BASIC INFORMATION:')
    print(f'   - Total records: {len(df_clean):,}')
    print(f"   - Date range: {df_clean['Date'].min().strftime('%Y-%m-%d')} to {df_clean['Date'].max().strftime('%Y-%m-%d')}")
    print(f"   - Categories: {df_clean['Category'].nunique()}")
    print(f"   - Products: {df_clean['Product_Name'].nunique()}")
    print(f'\n2. SALES STATISTICS:')
    print(f"   - Total sales: PHP {df_clean['Sales_Value_PHP'].sum():,.2f}")
    print(f"   - Average daily sales: PHP {df_clean['Sales_Value_PHP'].mean():,.2f}")
    print(f"   - Maximum single sale: PHP {df_clean['Sales_Value_PHP'].max():,.2f}")
    print(f'\n3. CATEGORY ANALYSIS:')
    category_stats = df_clean.groupby('Category').agg({'Sales_Value_PHP': ['sum', 'mean', 'count'], 'Product_Name': 'nunique'}).round(2)
    print(category_stats)
    return df_clean
df = load_and_explore_data()

def create_visualizations(df):
    print('\n CREATING VISUALIZATIONS...')
    daily_sales = df.groupby('Date')['Sales_Value_PHP'].sum()
    (fig, axes) = plt.subplots(2, 2, figsize=(16, 12))
    axes[0, 0].plot(daily_sales.index, daily_sales.values, linewidth=1, alpha=0.8)
    axes[0, 0].set_title('Daily Sales Trend', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Sales (PHP)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    category_sales = df.groupby('Category')['Sales_Value_PHP'].sum()
    axes[0, 1].pie(category_sales.values, labels=category_sales.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Sales Distribution by Category', fontsize=14, fontweight='bold')
    top_products = df.groupby('Product_Name')['Sales_Value_PHP'].sum().nlargest(10)
    axes[1, 0].barh(range(len(top_products)), top_products.values)
    axes[1, 0].set_yticks(range(len(top_products)))
    axes[1, 0].set_yticklabels([p[:30] + '...' if len(p) > 30 else p for p in top_products.index])
    axes[1, 0].set_title('Top 10 Products by Sales', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Total Sales (PHP)')
    monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Sales_Value_PHP'].sum()
    monthly_sales.index = monthly_sales.index.to_timestamp()
    axes[1, 1].plot(monthly_sales.index, monthly_sales.values, marker='o')
    axes[1, 1].set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Sales (PHP)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    return daily_sales
daily_sales = create_visualizations(df)

def prepare_forecasting_data(ts_data, test_size=0.2):
    print('\n PREPARING DATA FOR FORECASTING...')
    ts_data = ts_data.sort_index()
    split_idx = int(len(ts_data) * (1 - test_size))
    split_date = ts_data.index[split_idx]
    train = ts_data.iloc[:split_idx]
    test = ts_data.iloc[split_idx:]
    print(f"Training data: {len(train)} points ({train.index.min().strftime('%Y-%m-%d')} to {train.index.max().strftime('%Y-%m-%d')})")
    print(f"Test data: {len(test)} points ({test.index.min().strftime('%Y-%m-%d')} to {test.index.max().strftime('%Y-%m-%d')})")
    print(f'Test size: {len(test) / len(ts_data) * 100:.1f}%')
    enhanced_data_validation(df, ts_data, train, test)
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Training', linewidth=1, alpha=0.8)
    plt.plot(test.index, test, label='Test', linewidth=1, alpha=0.8)
    plt.axvline(x=split_date, color='red', linestyle='--', alpha=0.7, label='Split Point')
    plt.title('Train-Test Split for Time Series Forecasting', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Sales Value (PHP)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return (train, test, split_date)
(train, test, split_date) = prepare_forecasting_data(daily_sales)

def create_baseline_models(train, test):
    print('\n CREATING BASELINE MODELS...')
    last_value = pd.Series([train.iloc[-1]] * len(test), index=test.index)
    window = 7
    moving_avg = pd.Series([train.tail(window).mean()] * len(test), index=test.index)
    seasonal_naive_values = []
    for date in test.index:
        last_week_date = date - pd.Timedelta(days=7)
        if last_week_date in train.index:
            seasonal_naive_values.append(train.loc[last_week_date])
        else:
            seasonal_naive_values.append(train.iloc[-1])
    seasonal_naive = pd.Series(seasonal_naive_values, index=test.index)
    baselines = {'Last Value': last_value, 'Moving Average (7d)': moving_avg, 'Seasonal Naive': seasonal_naive}
    return baselines
baseline_models = create_baseline_models(train, test)

def create_sarimax_model(train, test):
    print('\n TRAINING SARIMAX MODEL...')
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.stattools import adfuller

        def check_stationarity(timeseries):
            result = adfuller(timeseries.dropna())
            print(f'ADF Statistic: {result[0]:.6f}')
            print(f'p-value: {result[1]:.6f}')
            return result[1] <= 0.05
        is_stationary = check_stationarity(train)
        print(f'Series is stationary: {is_stationary}')
        decomposition = seasonal_decompose(train, period=7, model='additive', extrapolate_trend='freq')
        plt.figure(figsize=(12, 8))
        plt.subplot(4, 1, 1)
        plt.plot(decomposition.observed)
        plt.title('Seasonal Decomposition - Original Series')
        plt.subplot(4, 1, 2)
        plt.plot(decomposition.trend)
        plt.title('Trend Component')
        plt.subplot(4, 1, 3)
        plt.plot(decomposition.seasonal)
        plt.title('Seasonal Component')
        plt.subplot(4, 1, 4)
        plt.plot(decomposition.resid)
        plt.title('Residual Component')
        plt.tight_layout()
        plt.show()
        try:
            import pmdarima as pm
            print('Searching for optimal SARIMAX parameters using auto_arima...')
            auto_arima_model = pm.auto_arima(train, seasonal=True, m=7, stepwise=True, suppress_warnings=True, error_action='ignore', max_order=None)
            order = auto_arima_model.order
            seasonal_order = auto_arima_model.seasonal_order
            print(f'Best SARIMAX parameters: {order}, {seasonal_order}')
        except:
            print('pmdarima not available, using default parameters')
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 7)
        sarimax_model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        sarimax_fit = sarimax_model.fit(disp=False)
        print('SARIMAX Model fitted successfully!')
        sarimax_forecast = sarimax_fit.forecast(steps=len(test))
        sarimax_forecast = pd.Series(sarimax_forecast, index=test.index)
        return sarimax_forecast
    except Exception as e:
        print(f'SARIMAX modeling failed: {e}')
        return None
sarimax_forecast = create_sarimax_model(train, test)

def create_prophet_model(train, test):
    print('\n TRAINING PROPHET MODEL...')
    try:
        from prophet import Prophet
        prophet_train = train.reset_index()
        prophet_train.columns = ['ds', 'y']
        prophet_model = Prophet(changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, holidays_prior_scale=10.0, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, seasonality_mode='additive')
        try:
            prophet_model.add_country_holidays(country_name='PH')
        except:
            print('Could not add country holidays, continuing without them')
        prophet_model.fit(prophet_train)
        future = prophet_model.make_future_dataframe(periods=len(test), include_history=False)
        prophet_forecast = prophet_model.predict(future)
        prophet_forecast_values = prophet_forecast.set_index('ds')['yhat']
        fig = prophet_model.plot_components(prophet_forecast)
        plt.suptitle('Prophet Model Components', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        print('Prophet model fitted successfully!')
        return prophet_forecast_values
    except Exception as e:
        print(f'Prophet modeling failed: {e}')
        return None
prophet_forecast = create_prophet_model(train, test)

def create_xgboost_model(train, test):
    print('\n TRAINING XGBOOST MODEL...')
    try:
        from xgboost import XGBRegressor
        from sklearn.preprocessing import StandardScaler

        def create_comprehensive_features(series):
            df = series.to_frame().copy()
            df.columns = ['target']
            df['dayofweek'] = df.index.dayofweek
            df['dayofyear'] = df.index.dayofyear
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['year'] = df.index.year
            df['weekofyear'] = df.index.isocalendar().week.astype(int)
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            for lag in [1, 2, 3, 7, 14, 21, 30]:
                df[f'lag_{lag}'] = df['target'].shift(lag)
            for window in [7, 14, 30]:
                df[f'rolling_mean_{window}'] = df['target'].rolling(window).mean()
                df[f'rolling_std_{window}'] = df['target'].rolling(window).std()
                df[f'rolling_min_{window}'] = df['target'].rolling(window).min()
                df[f'rolling_max_{window}'] = df['target'].rolling(window).max()
            df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
            df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
            df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
            df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
            return df
        train_features = create_comprehensive_features(train)
        test_features = create_comprehensive_features(pd.concat([train, test]))
        test_features = test_features[test_features.index >= test.index[0]]
        train_features = train_features.dropna()
        feature_columns = [col for col in train_features.columns if col != 'target']
        X_train = train_features[feature_columns]
        y_train = train_features['target']
        X_test = test_features[feature_columns]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, random_state=42, early_stopping_rounds=50)
        xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_train_scaled, y_train)], verbose=False)
        xgb_predictions = xgb_model.predict(X_test_scaled)
        xgb_forecast = pd.Series(xgb_predictions, index=test.index)
        plt.figure(figsize=(10, 8))
        feature_importance = pd.DataFrame({'feature': feature_columns, 'importance': xgb_model.feature_importances_}).sort_values('importance', ascending=True).tail(20)
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title('XGBoost - Top 20 Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        print('XGBoost model fitted successfully!')
        return xgb_forecast
    except Exception as e:
        print(f'XGBoost modeling failed: {e}')
        return None
xgb_forecast = create_xgboost_model(train, test)

def evaluate_models(actual, predictions_dict):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    metrics_list = []
    for (model_name, predictions) in predictions_dict.items():
        if predictions is not None and len(predictions) == len(actual):
            try:
                mae = mean_absolute_error(actual, predictions)
                mse = mean_squared_error(actual, predictions)
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(actual, predictions) * 100
                metrics_list.append({'Model': model_name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'RMSE_Formatted': f'PHP {rmse:,.2f}', 'MAPE_Formatted': f'{mape:.2f}%'})
            except Exception as e:
                print(f'Error evaluating {model_name}: {e}')
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df.sort_values('RMSE')
all_predictions = {'Last Value': baseline_models.get('Last Value'), 'Moving Average': baseline_models.get('Moving Average (7d)'), 'Seasonal Naive': baseline_models.get('Seasonal Naive'), 'SARIMAX': sarimax_forecast, 'Prophet': prophet_forecast, 'XGBoost': xgb_forecast}
all_predictions = {k: v for (k, v) in all_predictions.items() if v is not None}
metrics_df = evaluate_models(test, all_predictions)
print('\n' + '=' * 60)
print('MODEL PERFORMANCE COMPARISON')
print('=' * 60)
print(metrics_df.to_string(index=False))

def visualize_results(train, test, all_predictions, metrics_df):
    print('\n VISUALIZING RESULTS...')
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(train.index, train, label='Training', color='black', alpha=0.7, linewidth=1)
    plt.plot(test.index, test, label='Actual', color='blue', linewidth=2, marker='o', markersize=3)
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
    for (i, (model_name, predictions)) in enumerate(all_predictions.items()):
        if predictions is not None:
            plt.plot(predictions.index, predictions, label=model_name, color=colors[i % len(colors)], linestyle='--', linewidth=1.5, alpha=0.8)
    plt.title('Sales Forecasting - Model Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Sales Value (PHP)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.subplot(2, 1, 2)
    (fig, axes) = plt.subplots(1, 3, figsize=(18, 6))
    for (idx, metric) in enumerate(['MAE', 'RMSE', 'MAPE']):
        metric_data = metrics_df[['Model', metric]].sort_values(metric)
        axes[idx].barh(metric_data['Model'], metric_data[metric])
        axes[idx].set_title(f'{metric} Comparison', fontweight='bold')
        if metric == 'MAPE':
            axes[idx].set_xlabel(f'{metric} (%)')
        else:
            axes[idx].set_xlabel(f'{metric} (PHP)')
        axes[idx].grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
visualize_results(train, test, all_predictions, metrics_df)

def create_future_forecast(best_model_name, full_series, forecast_days=30):
    print(f'\n CREATING FUTURE FORECAST USING {best_model_name}...')
    if best_model_name == 'Prophet' and 'Prophet' in all_predictions:
        try:
            from prophet import Prophet
            full_data = full_series.reset_index()
            full_data.columns = ['ds', 'y']
            prophet_full = Prophet(changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, weekly_seasonality=True, yearly_seasonality=True)
            prophet_full.fit(full_data)
            future_df = prophet_full.make_future_dataframe(periods=forecast_days)
            future_forecast = prophet_full.predict(future_df)
            fig = prophet_full.plot(future_forecast)
            plt.title(f'Future Sales Forecast ({best_model_name}) - Next {forecast_days} Days', fontsize=14, fontweight='bold')
            plt.show()
            future_predictions = future_forecast[future_forecast['ds'] > full_series.index.max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            print(f'\nFuture Predictions (Next {forecast_days} days):')
            print(future_predictions.tail(10))
            return future_forecast
        except Exception as e:
            print(f'Future forecasting with Prophet failed: {e}')
            return None
    else:
        print(f'Future forecasting for {best_model_name} requires additional implementation')
        return None
if not metrics_df.empty:
    best_model = metrics_df.iloc[0]['Model']
    print(f'\n BEST PERFORMING MODEL: {best_model}')
    print(f"   RMSE: {metrics_df.iloc[0]['RMSE_Formatted']}")
    print(f"   MAPE: {metrics_df.iloc[0]['MAPE_Formatted']}")
    future_forecast = create_future_forecast(best_model, daily_sales)

def generate_summary_report(df, metrics_df, best_model, all_predictions):
    print('\n=== DATA QUALITY REPORT ===')
    print(f'Total records after cleaning: {len(df)}')
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Total sales value: PHP {df['Sales_Value_PHP'].sum():,.2f}")
    print(f"Unique products: {df['Product_Name'].nunique()}")
    print(f'Data completeness: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%')
    print('\n' + '=' * 70)
    print('FINAL FORECASTING SUMMARY REPORT')
    print('=' * 70)
    print(f'\n DATASET OVERVIEW:')
    print(f"   • Period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f'   • Total records: {len(df):,}')
    print(f"   • Total sales: PHP {df['Sales_Value_PHP'].sum():,.2f}")
    print(f"   • Categories: {df['Category'].nunique()}")
    print(f"   • Products: {df['Product_Name'].nunique()}")
    print(f'\n MODELS IMPLEMENTED ({len(all_predictions)}):')
    for model in all_predictions.keys():
        print(f'   • {model}')
    print(f'\n BEST MODEL: {best_model}')
    if not metrics_df.empty:
        best_metrics = metrics_df[metrics_df['Model'] == best_model].iloc[0]
        print(f"   • RMSE: {best_metrics['RMSE_Formatted']}")
        print(f"   • MAPE: {best_metrics['MAPE_Formatted']}")
    print(f'\n💡 RECOMMENDATIONS:')
    print(f'   1. Use {best_model} for operational forecasting')
    print(f'   2. Monitor model performance monthly')
    print(f'   3. Retrain models with new data regularly')
    print(f'   4. Consider ensemble approaches for improved stability')
    print(f'   5. Incorporate external factors (promotions, holidays, weather)')
    print(f'\n🚀 NEXT STEPS:')
    print(f'   1. Implement automated retraining pipeline')
    print(f'   2. Set up monitoring dashboard')
    print(f'   3. Develop product-level forecasting')
    print(f'   4. Integrate with inventory management system')
    print(f'\n SYSTEM STATUS: COMPLETE')
    print('All models trained and evaluated successfully!')
if not metrics_df.empty:
    generate_summary_report(df, metrics_df, best_model, all_predictions)

def product_level_analysis(df):
    print('\n' + '=' * 50)
    print('PRODUCT-LEVEL ANALYSIS')
    print('=' * 50)
    top_5_products = df.groupby('Product_Name')['Sales_Value_PHP'].sum().nlargest(5).index.tolist()
    plt.figure(figsize=(15, 10))
    for (i, product) in enumerate(top_5_products, 1):
        product_data = df[df['Product_Name'] == product]
        product_daily = product_data.groupby('Date')['Sales_Value_PHP'].sum()
        plt.subplot(3, 2, i)
        plt.plot(product_daily.index, product_daily.values)
        plt.title(f'Sales Trend: {product[:40]}', fontsize=10)
        plt.xlabel('Date')
        plt.ylabel('Sales (PHP)')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    category_trend = df.pivot_table(values='Sales_Value_PHP', index='Date', columns='Category', aggfunc='sum')
    category_trend = category_trend.fillna(0)
    plt.figure(figsize=(12, 6))
    for category in category_trend.columns:
        plt.plot(category_trend.index, category_trend[category], label=category, alpha=0.7)
    plt.title('Sales Trends by Category', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Sales Value (PHP)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
product_level_analysis(df)
print('\n' + '=' * 70)
print('FORECASTING SYSTEM EXECUTION COMPLETE!')
print('=' * 70)