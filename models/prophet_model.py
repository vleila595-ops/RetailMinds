import pandas as pd
from prophet import Prophet

class ProphetModel:

    def __init__(self):
        self.model = None
        self.is_trained = False

    def train(self, train_data):
        df = train_data.reset_index()
        df.columns = ['ds', 'y']
        self.model = Prophet(changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
        self.model.fit(df)
        self.is_trained = True

    def predict(self, periods):
        if not self.is_trained:
            raise Exception('Model not trained')
        future = self.model.make_future_dataframe(periods=periods, include_history=False)
        forecast = self.model.predict(future)
        return forecast['yhat'].values