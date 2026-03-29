import pandas as pd
import numpy as np
from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel
from models.xgboost_model import XGBoostModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ModelFactory:

    def __init__(self):
        self.models = {'ARIMA': ARIMAModel(), 'Prophet': ProphetModel(), 'XGBoost': XGBoostModel()}

    def get_best_model(self, product_name, df):
        print(f'[DEBUG] Finding best model for: {product_name}')
        product_data = df[df['Product_Name'] == product_name].copy()
        print(f'[DEBUG] Product data shape: {product_data.shape}')
        if len(product_data) < 30:
            print(f'[DEBUG] Insufficient data ({len(product_data)} records), using Prophet as default')
            return ('Prophet', 'Insufficient data, using Prophet as default')
        daily_sales = product_data.groupby('Date')['Sales_Value_PHP'].sum()
        print(f'[DEBUG] Daily sales data points: {len(daily_sales)}')
        split_idx = int(len(daily_sales) * 0.8)
        train = daily_sales.iloc[:split_idx]
        test = daily_sales.iloc[split_idx:]
        print(f'[DEBUG] Train set size: {len(train)}, Test set size: {len(test)}')
        best_model = None
        best_rmse = float('inf')
        model_performance = {}
        print(f'[DEBUG] Evaluating models: {list(self.models.keys())}')
        for (model_name, model) in self.models.items():
            try:
                print(f'[DEBUG] Training {model_name} model...')
                model.train(train)
                predictions = model.predict(len(test))
                rmse = np.sqrt(mean_squared_error(test, predictions))
                mae = mean_absolute_error(test, predictions)
                model_performance[model_name] = {'rmse': rmse, 'mae': mae}
                print(f'[DEBUG] {model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}')
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_name
            except Exception as e:
                print(f'[DEBUG] Model {model_name} failed: {e}')
                continue
        print(f'[DEBUG] Best model selected: {best_model} with RMSE: {best_rmse:.4f}')
        return (best_model, model_performance)

    def predict_future(self, model_name, product_name, df, periods=30):
        try:
            product_data = df[df['Product_Name'] == product_name].copy()
            daily_sales = product_data.groupby('Date')['Sales_Value_PHP'].sum()
            model = self.models[model_name]
            model.train(daily_sales)
            predictions = model.predict(periods)
            return (predictions, None)
        except Exception as e:
            return (None, str(e))