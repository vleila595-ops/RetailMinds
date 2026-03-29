from flask import Blueprint, request, jsonify, render_template
from models.product_analyzer import ProductAnalyzer
from models.model_factory import ModelFactory
import pandas as pd
product_bp = Blueprint('product', __name__)
analyzer = ProductAnalyzer()
model_factory = ModelFactory()

@product_bp.route('/api/product/search')
def search_products():
    query = request.args.get('q', '').lower()
    try:
        if analyzer.df is None:
            df = pd.read_csv('data/THESIS-DATASET.csv')
            analyzer.load_data(df)
        products = analyzer.df[['Product_Name', 'Category']].drop_duplicates()
        products = products[products['Product_Name'].str.lower().str.contains(query)]
        products = products.head(10)
        return jsonify({'success': True, 'products': products.to_dict('records')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@product_bp.route('/api/product/<product_name>')
def get_product_details(product_name):
    try:
        if analyzer.df is None:
            df = pd.read_csv('data/THESIS-DATASET.csv')
            analyzer.load_data(df)
        product_metrics = analyzer.product_metrics[analyzer.product_metrics['Product_Name'] == product_name]
        if product_metrics.empty:
            return jsonify({'success': False, 'error': 'Product not found'})
        metrics = product_metrics.iloc[0]
        product_data = analyzer.df[analyzer.df['Product_Name'] == product_name]
        price_history_df = product_data.groupby('Date').agg({'Unit_Price_PHP': 'mean', 'Quantity_Sold': 'sum', 'Sales_Value_PHP': 'sum'}).reset_index().sort_values('Date')
        price_history = []
        for (_, row) in price_history_df.iterrows():
            price_history.append({'Date': str(row['Date']), 'Unit_Price_PHP': float(row['Unit_Price_PHP']), 'Quantity_Sold': int(row['Quantity_Sold']), 'Sales_Value_PHP': float(row['Sales_Value_PHP'])})
        try:
            recent_prices = product_data.tail(7)['Unit_Price_PHP']
            if len(recent_prices) >= 2:
                trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
                predicted_price = float(recent_prices.iloc[-1] + trend)
                confidence = min(0.95, max(0.6, 1 - abs(trend) / recent_prices.iloc[-1]))
            else:
                predicted_price = float(metrics['Average_Price_PHP'] * 1.01)
                confidence = 0.7
            prediction = {'next_day_price': predicted_price, 'confidence': confidence}
        except Exception as e:
            prediction = {'next_day_price': float(metrics['Average_Price_PHP'] * 1.005), 'confidence': 0.65}
        price_std = product_data['Unit_Price_PHP'].std()
        price_mean = product_data['Unit_Price_PHP'].mean()
        price_variability = price_std / price_mean if price_mean != 0 else 0
        insights = {'total_sales': float(metrics['Total_Sales_PHP']), 'total_quantity': int(metrics['Total_Quantity_Sold']), 'avg_price': float(metrics['Average_Price_PHP']), 'price_variability': float(price_variability), 'days_active': int(metrics['Days_Active']), 'avg_daily_sales': float(metrics['Avg_Daily_Sales_PHP']), 'avg_daily_quantity': float(metrics['Avg_Daily_Quantity']), 'stock_turnover': float(metrics['Stock_Turnover_Rate']), 'price_position': str(metrics['Price_Position']), 'product_segment': str(metrics['Product_Segment']), 'category': str(metrics['Category'])}
        return jsonify({'success': True, 'product_name': product_name, 'insights': insights, 'price_history': price_history, 'prediction': prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@product_bp.route('/product')
def product_search():
    return render_template('product.html')

@product_bp.route('/product/<product_name>')
def product_details(product_name):
    return render_template('product_details.html', product_name=product_name)