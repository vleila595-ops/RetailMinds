import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

class ARIMAModel:

    def __init__(self):
        self.model = None
        self.is_trained = False

    def train(self, train_data):
        try:
            self.auto_model = pm.auto_arima(train_data, seasonal=True, m=7, stepwise=True, suppress_warnings=True, error_action='ignore')
            order = self.auto_model.order
            seasonal_order = self.auto_model.seasonal_order
            self.model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            self.model_fit = self.model.fit(disp=False)
            self.is_trained = True
        except Exception as e:
            self.model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            self.model_fit = self.model.fit(disp=False)
            self.is_trained = True

    def predict(self, periods):
        if not self.is_trained:
            raise Exception('Model not trained')
        forecast = self.model_fit.forecast(steps=periods)
        return forecast