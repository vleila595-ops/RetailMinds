from flask import Blueprint, request, jsonify, render_template
import pandas as pd
import os
from datetime import datetime
data_bp = Blueprint('data', __name__)
CSV_FILE_PATH = 'data/THESIS-DATASET.csv'

@data_bp.route('/data')
def data_management():
    return render_template('data.html')

@data_bp.route('/api/data')
def get_data():
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date', ascending=False)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        data = df.to_dict('records')
        return jsonify({'success': True, 'data': data, 'total_records': len(data)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@data_bp.route('/api/data/add', methods=['POST'])
def add_data():
    try:
        new_record = request.get_json()
        required_fields = ['Date', 'Product_Name', 'Category', 'Unit_Size_kg', 'Unit_Price_PHP', 'Quantity_Sold', 'Sales_Value_PHP', 'Remaining_Stocks', 'Restock_Quantity']
        for field in required_fields:
            if field not in new_record:
                return jsonify({'success': False, 'error': f'Missing field: {field}'})
        df = pd.read_csv(CSV_FILE_PATH)
        new_df = pd.DataFrame([new_record])
        df = pd.concat([df, new_df], ignore_index=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date', ascending=False)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df.to_csv(CSV_FILE_PATH, index=False)
        return jsonify({'success': True, 'message': 'Record added successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@data_bp.route('/api/data/update/<int:index>', methods=['PUT'])
def update_data(index):
    try:
        updated_record = request.get_json()
        df = pd.read_csv(CSV_FILE_PATH)
        if index >= len(df):
            return jsonify({'success': False, 'error': 'Record not found'})
        for (key, value) in updated_record.items():
            if key in df.columns:
                df.at[index, key] = value
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date', ascending=False)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df.to_csv(CSV_FILE_PATH, index=False)
        return jsonify({'success': True, 'message': 'Record updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@data_bp.route('/api/data/delete/<int:index>', methods=['DELETE'])
def delete_data(index):
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        if index >= len(df):
            return jsonify({'success': False, 'error': 'Record not found'})
        df = df.drop(index)
        df.reset_index(drop=True, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date', ascending=False)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df.to_csv(CSV_FILE_PATH, index=False)
        return jsonify({'success': True, 'message': 'Record deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@data_bp.route('/api/data/bulk_upload', methods=['POST'])
def bulk_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        if file and file.filename.endswith('.csv'):
            new_df = pd.read_csv(file)
            existing_df = pd.read_csv(CSV_FILE_PATH)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df['Date'] = pd.to_datetime(combined_df['Date'])
            combined_df = combined_df.sort_values('Date', ascending=False)
            combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
            combined_df.to_csv(CSV_FILE_PATH, index=False)
            return jsonify({'success': True, 'message': f'Successfully added {len(new_df)} records'})
        else:
            return jsonify({'success': False, 'error': 'Invalid file format'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@data_bp.route('/api/data/sort', methods=['POST'])
def sort_data():
    try:
        sort_by = request.json.get('sort_by', 'Date')
        ascending = request.json.get('ascending', False)
        df = pd.read_csv(CSV_FILE_PATH)
        if sort_by == 'Date':
            df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(sort_by, ascending=ascending)
        if sort_by == 'Date':
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df.to_csv(CSV_FILE_PATH, index=False)
        return jsonify({'success': True, 'message': f'Data sorted by {sort_by}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})