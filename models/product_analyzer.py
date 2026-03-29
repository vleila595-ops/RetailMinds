import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ProductAnalyzer:

    def __init__(self):
        self.df = None
        self.product_metrics = None

    def load_data(self, df):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.calculate_metrics()

    def calculate_metrics(self):
        product_stats = []
        for product in self.df['Product_Name'].unique():
            product_data = self.df[self.df['Product_Name'] == product]
            total_sales = product_data['Sales_Value_PHP'].sum()
            total_quantity = product_data['Quantity_Sold'].sum()
            avg_price = product_data['Unit_Price_PHP'].mean()
            total_transactions = len(product_data)
            days_active = product_data['Date'].nunique()
            avg_daily_sales = total_sales / days_active if days_active > 0 else 0
            avg_daily_quantity = total_quantity / days_active if days_active > 0 else 0
            avg_stock = product_data['Remaining_Stocks'].mean()
            avg_restock = product_data['Restock_Quantity'].mean()
            stock_turnover = total_quantity / avg_stock if avg_stock > 0 else 0
            price_quartile = self.get_price_quartile(avg_price)
            product_stats.append({'Product_Name': product, 'Category': product_data['Category'].iloc[0], 'Total_Sales_PHP': total_sales, 'Total_Quantity_Sold': total_quantity, 'Average_Price_PHP': avg_price, 'Total_Transactions': total_transactions, 'Days_Active': days_active, 'Avg_Daily_Sales_PHP': avg_daily_sales, 'Avg_Daily_Quantity': avg_daily_quantity, 'Avg_Stock_Level': avg_stock, 'Avg_Restock_Quantity': avg_restock, 'Stock_Turnover_Rate': stock_turnover, 'Price_Quartile': price_quartile, 'Sales_Velocity': avg_daily_quantity * avg_price, 'Price_Position': self.get_price_position(avg_price, avg_daily_quantity)})
        self.product_metrics = pd.DataFrame(product_stats)
        self.calculate_segments()

    def get_price_quartile(self, price):
        all_prices = self.df['Unit_Price_PHP'].unique()
        if price <= np.percentile(all_prices, 25):
            return 'Low Price'
        elif price <= np.percentile(all_prices, 50):
            return 'Medium-Low Price'
        elif price <= np.percentile(all_prices, 75):
            return 'Medium-High Price'
        else:
            return 'High Price'

    def get_price_position(self, price, daily_quantity):
        avg_price = self.df['Unit_Price_PHP'].mean()
        avg_quantity = self.df.groupby('Product_Name')['Quantity_Sold'].sum().mean()
        if price > avg_price and daily_quantity > avg_quantity:
            return 'Premium Fast-Mover'
        elif price > avg_price and daily_quantity <= avg_quantity:
            return 'Premium Slow-Mover'
        elif price <= avg_price and daily_quantity > avg_quantity:
            return 'Value Fast-Mover'
        else:
            return 'Value Slow-Mover'

    def calculate_segments(self):
        features = self.product_metrics[['Average_Price_PHP', 'Avg_Daily_Quantity', 'Stock_Turnover_Rate']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=4, random_state=42)
        segments = kmeans.fit_predict(features_scaled)
        segment_names = {0: 'High-Volume Essentials', 1: 'Premium Specialty', 2: 'Slow-Moving Basics', 3: 'Opportunity Products'}
        self.product_metrics['Product_Segment'] = [segment_names.get(s, 'Other') for s in segments]

    def get_transaction_analysis(self):
        daily_transactions = self.df.groupby('Date').agg({'Sales_Value_PHP': 'sum', 'Quantity_Sold': 'sum', 'Product_Name': 'count'}).rename(columns={'Product_Name': 'Transaction_Count'})
        daily_transactions['Avg_Transaction_Value'] = daily_transactions['Sales_Value_PHP'] / daily_transactions['Transaction_Count']
        daily_transactions['Items_Per_Transaction'] = daily_transactions['Quantity_Sold'] / daily_transactions['Transaction_Count']
        return daily_transactions

    def get_price_analysis(self):
        price_stats = self.df.groupby('Product_Name').agg({'Unit_Price_PHP': ['min', 'max', 'mean', 'std'], 'Quantity_Sold': 'sum', 'Sales_Value_PHP': 'sum', 'Category': 'first'}).round(2)
        price_stats.columns = ['Min_Price', 'Max_Price', 'Avg_Price', 'Price_Std', 'Total_Quantity', 'Total_Sales', 'Category']
        price_stats['Price_Variability'] = price_stats['Price_Std'] / price_stats['Avg_Price']
        return price_stats.sort_values('Avg_Price', ascending=False)

    def get_fast_moving_analysis(self):
        fast_moving = self.product_metrics[(self.product_metrics['Avg_Daily_Quantity'] > self.product_metrics['Avg_Daily_Quantity'].median()) & (self.product_metrics['Stock_Turnover_Rate'] > self.product_metrics['Stock_Turnover_Rate'].median())].sort_values('Avg_Daily_Quantity', ascending=False)
        slow_moving = self.product_metrics[(self.product_metrics['Avg_Daily_Quantity'] < self.product_metrics['Avg_Daily_Quantity'].median()) & (self.product_metrics['Stock_Turnover_Rate'] < self.product_metrics['Stock_Turnover_Rate'].median())].sort_values('Avg_Daily_Quantity', ascending=True)
        return {'fast_moving': fast_moving, 'slow_moving': slow_moving}

    def get_profit_potential_analysis(self):
        analysis = self.product_metrics.copy()
        analysis['Sales_Velocity_Rank'] = analysis['Avg_Daily_Quantity'].rank(ascending=False)
        analysis['Price_Rank'] = analysis['Average_Price_PHP'].rank(ascending=False)
        analysis['Profit_Potential_Score'] = (analysis['Sales_Velocity_Rank'] + analysis['Price_Rank']) / 2
        high_potential = analysis[(analysis['Price_Position'] == 'Value Fast-Mover') | (analysis['Price_Position'] == 'Premium Fast-Mover')].sort_values('Profit_Potential_Score', ascending=False)
        return high_potential

    def get_category_analysis(self):
        category_stats = self.df.groupby('Category').agg({'Sales_Value_PHP': 'sum', 'Quantity_Sold': 'sum', 'Product_Name': 'nunique', 'Unit_Price_PHP': 'mean', 'Quantity_Sold': 'mean'}).round(2)
        category_stats.columns = ['Total_Sales', 'Total_Quantity', 'Unique_Products', 'Avg_Price', 'Avg_Quantity_Per_Transaction']
        category_stats['Sales_Per_Product'] = category_stats['Total_Sales'] / category_stats['Unique_Products']
        return category_stats.sort_values('Total_Sales', ascending=False)