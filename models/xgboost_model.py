import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

class XGBoostModel:

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def create_features(self, series):
        df = series.to_frame().copy()
        df.columns = ['target']
        df['dayofweek'] = df.index.dayofweek
        df['dayofyear'] = df.index.dayofyear
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['weekofyear'] = df.index.isocalendar().week.astype(int)
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        for lag in [1, 7, 14, 30]:
            df[f'lag_{lag}'] = df['target'].shift(lag)
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df['target'].rolling(window).mean()
            df[f'rolling_std_{window}'] = df['target'].rolling(window).std()
        return df

    def train(self, train_data):
        train_features = self.create_features(train_data)
        train_features = train_features.dropna()
        feature_columns = [col for col in train_features.columns if col != 'target']
        X_train = train_features[feature_columns]
        y_train = train_features['target']
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        self.feature_columns = feature_columns
        self.is_trained = True

    def predict(self, periods):
        if not self.is_trained:
            raise Exception('Model not trained')
        last_value = self.model.feature_importances_[0]
        return np.array([last_value] * periods)