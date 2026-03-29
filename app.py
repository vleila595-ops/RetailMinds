from flask import Flask, render_template, request, jsonify, session
import os
import pandas as pd
from models.model_factory import ModelFactory
from routes.predict import predict_bp
from routes.data import data_bp
from routes.analytics import analytics_bp
from routes.product import product_bp
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.register_blueprint(predict_bp)
app.register_blueprint(data_bp)
app.register_blueprint(analytics_bp)
app.register_blueprint(product_bp)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/analyze_data')
def analyze_data():
    try:
        print("[DEBUG] Analyzing data...")
        df = pd.read_csv('data/THESIS-DATASET.csv')
        print(f"[DEBUG] Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        df['Date'] = pd.to_datetime(df['Date'])
        stats = {'total_products': df['Product_Name'].nunique(), 'total_categories': df['Category'].nunique(), 'total_sales': f"PHP {df['Sales_Value_PHP'].sum():,.2f}", 'date_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}", 'top_products': df.groupby('Product_Name')['Sales_Value_PHP'].sum().nlargest(5).to_dict()}
        print(f"[DEBUG] Data analysis complete: {stats['total_products']} products, {stats['total_categories']} categories")
        return jsonify(stats)
    except Exception as e:
        print(f"[DEBUG] Error in data analysis: {e}")
        return jsonify({'error': str(e)})

@app.route('/data')
def data_management():
    return render_template('data.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')
if __name__ == '__main__':
    print("[DEBUG] Starting web server...")
    print("[DEBUG] Server configuration:")
    print(f"[DEBUG] - Host: 0.0.0.0")
    print(f"[DEBUG] - Port: 5500")
    print(f"[DEBUG] - Debug mode: True")
    print(f"[DEBUG] - Threaded: True")
    print("[DEBUG] Server starting up...")
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5500,
        threaded=True
        )