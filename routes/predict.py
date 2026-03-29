from flask import Blueprint, request, jsonify, session
import pandas as pd
from models.model_factory import ModelFactory
predict_bp = Blueprint('predict', __name__)
model_factory = ModelFactory()

@predict_bp.route('/api/predict', methods=['POST'])
def predict_sales():
    try:
        data = request.get_json()
        product_name = data.get('product_name')
        forecast_days = int(data.get('forecast_days', 30))
        df = pd.read_csv('data/THESIS-DATASET.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        available_products = df['Product_Name'].unique().tolist()
        if product_name not in available_products:
            return (jsonify({'error': f'Product "{product_name}" not found in dataset. Available products: {available_products[:5]}...'}), 400)
        (best_model, performance) = model_factory.get_best_model(product_name, df)
        (predictions, error) = model_factory.predict_future(best_model, product_name, df, forecast_days)
        if error:
            return (jsonify({'error': error}), 500)
        response = {'product_name': product_name, 'best_model': best_model, 'model_performance': performance, 'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions), 'forecast_days': forecast_days, 'available_products_count': len(available_products)}
        return jsonify(response)
    except Exception as e:
        return (jsonify({'error': str(e)}), 500)

@predict_bp.route('/api/products')
def get_products():
    try:
        print("[DEBUG] Fetching available products...")
        df = pd.read_csv('data/THESIS-DATASET.csv')
        products = df['Product_Name'].unique().tolist()
        print(f"[DEBUG] Found {len(products)} unique products")
        return jsonify({'products': products})
    except Exception as e:
        print(f"[DEBUG] Error fetching products: {e}")
        return (jsonify({'error': str(e)}), 500)