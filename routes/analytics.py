from flask import Blueprint, request, jsonify
import pandas as pd
from models.product_analyzer import ProductAnalyzer
analytics_bp = Blueprint('analytics', __name__)
analyzer = ProductAnalyzer()

@analytics_bp.route('/api/analytics/load')
def load_analytics():
    try:
        df = pd.read_csv('data/THESIS-DATASET.csv')
        analyzer.load_data(df)
        return jsonify({'success': True, 'message': 'Data analyzed successfully', 'total_products': len(analyzer.product_metrics), 'total_categories': df['Category'].nunique()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@analytics_bp.route('/api/analytics/transactions')
def get_transaction_analysis():
    try:
        if analyzer.df is None:
            df = pd.read_csv('data/THESIS-DATASET.csv')
            analyzer.load_data(df)
        transactions = analyzer.get_transaction_analysis()
        return jsonify({'success': True, 'data': transactions.reset_index().to_dict('records')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@analytics_bp.route('/api/analytics/price-analysis')
def get_price_analysis():
    try:
        if analyzer.df is None:
            df = pd.read_csv('data/THESIS-DATASET.csv')
            analyzer.load_data(df)
        price_analysis = analyzer.get_price_analysis()
        return jsonify({'success': True, 'data': price_analysis.reset_index().to_dict('records')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@analytics_bp.route('/api/analytics/movement-analysis')
def get_movement_analysis():
    try:
        movement = analyzer.get_fast_moving_analysis()
        return jsonify({'success': True, 'fast_moving': movement['fast_moving'].to_dict('records'), 'slow_moving': movement['slow_moving'].to_dict('records')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@analytics_bp.route('/api/analytics/profit-potential')
def get_profit_potential():
    try:
        profit_potential = analyzer.get_profit_potential_analysis()
        return jsonify({'success': True, 'data': profit_potential.to_dict('records')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@analytics_bp.route('/api/analytics/category-analysis')
def get_category_analysis():
    try:
        category_analysis = analyzer.get_category_analysis()
        return jsonify({'success': True, 'data': category_analysis.reset_index().to_dict('records')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@analytics_bp.route('/api/analytics/product-segments')
def get_product_segments():
    try:
        segments = analyzer.product_metrics[['Product_Name', 'Category', 'Product_Segment', 'Price_Position', 'Avg_Daily_Quantity', 'Average_Price_PHP']]
        return jsonify({'success': True, 'data': segments.to_dict('records')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@analytics_bp.route('/api/analytics/summary')
def get_analytics_summary():
    try:
        if analyzer.df is None:
            df = pd.read_csv('data/THESIS-DATASET.csv')
            analyzer.load_data(df)
        total_products = len(analyzer.product_metrics)
        total_categories = analyzer.df['Category'].nunique()
        total_transactions = len(analyzer.df)
        highest_price_product = analyzer.product_metrics.loc[analyzer.product_metrics['Average_Price_PHP'].idxmax()]
        lowest_price_product = analyzer.product_metrics.loc[analyzer.product_metrics['Average_Price_PHP'].idxmin()]
        fastest_moving = analyzer.product_metrics.loc[analyzer.product_metrics['Avg_Daily_Quantity'].idxmax()]
        slowest_moving = analyzer.product_metrics.loc[analyzer.product_metrics['Avg_Daily_Quantity'].idxmin()]
        segment_counts = analyzer.product_metrics['Product_Segment'].value_counts().to_dict()
        price_position_counts = analyzer.product_metrics['Price_Position'].value_counts().to_dict()
        return jsonify({'success': True, 'summary': {'total_products': total_products, 'total_categories': total_categories, 'total_transactions': total_transactions, 'date_range': {'start': analyzer.df['Date'].min().strftime('%Y-%m-%d'), 'end': analyzer.df['Date'].max().strftime('%Y-%m-%d')}, 'price_extremes': {'highest': {'product': highest_price_product['Product_Name'], 'price': highest_price_product['Average_Price_PHP'], 'daily_quantity': highest_price_product['Avg_Daily_Quantity']}, 'lowest': {'product': lowest_price_product['Product_Name'], 'price': lowest_price_product['Average_Price_PHP'], 'daily_quantity': lowest_price_product['Avg_Daily_Quantity']}}, 'movement_extremes': {'fastest': {'product': fastest_moving['Product_Name'], 'daily_quantity': fastest_moving['Avg_Daily_Quantity'], 'price': fastest_moving['Average_Price_PHP']}, 'slowest': {'product': slowest_moving['Product_Name'], 'daily_quantity': slowest_moving['Avg_Daily_Quantity'], 'price': slowest_moving['Average_Price_PHP']}}, 'segments': segment_counts, 'price_positions': price_position_counts}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})