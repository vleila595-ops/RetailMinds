import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime

class ModelEvaluator:

    def __init__(self, output_dir='model_evaluation'):
        self.output_dir = output_dir
        self.results = {}
        self.setup_plot_style()
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/charts', exist_ok=True)
        os.makedirs(f'{output_dir}/reports', exist_ok=True)

    def setup_plot_style(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette('husl')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def evaluate_models_comprehensive(self, df, test_size=0.2, min_data_points=30):
        print('🚀 Starting Comprehensive Model Evaluation Across ALL Products...')
        all_products = self._get_products_with_sufficient_data(df, min_data_points)
        print(f'📊 Evaluating {len(all_products)} products with sufficient data...')
        from models.arima_model import ARIMAModel
        from models.prophet_model import ProphetModel
        from models.xgboost_model import XGBoostModel
        models = {'ARIMA': ARIMAModel(), 'Prophet': ProphetModel(), 'XGBoost': XGBoostModel()}
        model_performance = {model_name: [] for model_name in models.keys()}
        all_predictions = {model_name: [] for model_name in models.keys()}
        all_actuals = {model_name: [] for model_name in models.keys()}
        product_results = {}
        for (i, product_name) in enumerate(all_products, 1):
            print(f'\n📦 Evaluating Product {i}/{len(all_products)}: {product_name}')
            try:
                product_data = df[df['Product_Name'] == product_name].copy()
                daily_sales = product_data.groupby('Date')['Sales_Value_PHP'].sum().sort_index()
                split_idx = int(len(daily_sales) * (1 - test_size))
                train = daily_sales.iloc[:split_idx]
                test = daily_sales.iloc[split_idx:]
                product_metrics = {'product_name': product_name, 'train_size': len(train), 'test_size': len(test), 'models': {}}
                for (model_name, model) in models.items():
                    try:
                        print(f'  🔧 Testing {model_name}...')
                        model_results = self._evaluate_single_model(model, model_name, train, test)
                        product_metrics['models'][model_name] = model_results
                        model_performance[model_name].append({'product': product_name, 'rmse': model_results['regression_metrics']['RMSE'], 'mae': model_results['regression_metrics']['MAE'], 'mape': model_results['regression_metrics']['MAPE'], 'r2': model_results['regression_metrics']['R2'], 'accuracy': model_results['classification_metrics']['accuracy']})
                        all_predictions[model_name].extend(model_results['predictions'])
                        all_actuals[model_name].extend(model_results['actual_values'])
                    except Exception as e:
                        print(f'  ❌ {model_name} failed for {product_name}: {e}')
                        continue
                product_results[product_name] = product_metrics
            except Exception as e:
                print(f'❌ Error evaluating {product_name}: {e}')
                continue
        overall_results = self._generate_overall_model_reports(model_performance, all_predictions, all_actuals)
        self._create_overall_visualizations(overall_results, product_results)
        self._save_comprehensive_reports(overall_results, product_results)
        print(f'\n✅ Comprehensive Evaluation Complete! Results saved to {self.output_dir}/')
        return (overall_results, product_results)

    def _get_products_with_sufficient_data(self, df, min_points=30):
        product_data_counts = df.groupby('Product_Name').size()
        sufficient_products = product_data_counts[product_data_counts >= min_points].index.tolist()
        print(f'📈 Found {len(sufficient_products)} products with ≥{min_points} data points')
        return sufficient_products

    def _evaluate_single_model(self, model, model_name, train, test):
        model.train(train)
        predictions = model.predict(len(test))
        if len(predictions) != len(test):
            predictions = predictions[:len(test)]
        mae = mean_absolute_error(test, predictions)
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(test, predictions) * 100
        r2 = self._calculate_r2(test, predictions)
        mase = self._calculate_mase(test, predictions, train)
        classification_metrics = self._calculate_classification_metrics(test, predictions)
        feature_importance = self._get_feature_importance(model, model_name)
        residuals = test.values - predictions
        residual_stats = {'mean': np.mean(residuals), 'std': np.std(residuals), 'skewness': float(pd.Series(residuals).skew()), 'kurtosis': float(pd.Series(residuals).kurtosis())}
        return {'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions), 'regression_metrics': {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'MASE': mase}, 'classification_metrics': classification_metrics, 'residual_analysis': residual_stats, 'feature_importance': feature_importance, 'actual_values': test.values.tolist()}

    def _generate_overall_model_reports(self, model_performance, all_predictions, all_actuals):
        overall_results = {}
        for (model_name, performances) in model_performance.items():
            if not performances:
                continue
            perf_df = pd.DataFrame(performances)
            overall_metrics = {'products_evaluated': len(performances), 'mean_rmse': perf_df['rmse'].mean(), 'std_rmse': perf_df['rmse'].std(), 'mean_mae': perf_df['mae'].mean(), 'mean_mape': perf_df['mape'].mean(), 'mean_r2': perf_df['r2'].mean(), 'mean_accuracy': perf_df['accuracy'].mean(), 'best_rmse_product': perf_df.loc[perf_df['rmse'].idxmin(), 'product'], 'worst_rmse_product': perf_df.loc[perf_df['rmse'].idxmax(), 'product'], 'rmse_percentiles': {'25th': perf_df['rmse'].quantile(0.25), '50th': perf_df['rmse'].quantile(0.5), '75th': perf_df['rmse'].quantile(0.75)}}
            win_counts = {}
            for perf in performances:
                pass
            overall_results[model_name] = overall_metrics
        if overall_results:
            best_model = min(overall_results.keys(), key=lambda x: overall_results[x]['mean_rmse'])
            overall_results['overall_best_model'] = best_model
        return overall_results

    def _create_overall_visualizations(self, overall_results, product_results):
        self._plot_overall_performance(overall_results)
        self._plot_performance_distribution(product_results)
        self._plot_win_rate_analysis(product_results)
        self._plot_consistency_analysis(product_results)
        self._plot_performance_heatmap(product_results)

    def _plot_overall_performance(self, overall_results):
        models = [model for model in overall_results.keys() if model != 'overall_best_model']
        if not models:
            return
        (fig, axes) = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        metrics = ['mean_rmse', 'mean_mae', 'mean_mape', 'mean_r2', 'mean_accuracy', 'std_rmse']
        metric_names = ['RMSE', 'MAE', 'MAPE (%)', 'R²', 'Accuracy', 'RMSE Std Dev']
        for (i, (metric, name)) in enumerate(zip(metrics, metric_names)):
            values = [overall_results[model][metric] for model in models]
            bars = axes[i].bar(models, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[i].set_title(f'Overall {name}', fontweight='bold')
            axes[i].set_ylabel(name)
            for (bar, value) in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            if metric in ['mean_rmse', 'mean_mae', 'mean_mape']:
                best_idx = np.argmin(values)
                bars[best_idx].set_color('#2ECC71')
            elif metric in ['mean_r2', 'mean_accuracy']:
                best_idx = np.argmax(values)
                bars[best_idx].set_color('#2ECC71')
        plt.suptitle('Overall Model Performance Comparison Across All Products', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/charts/overall_model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_distribution(self, product_results):
        models = ['ARIMA', 'Prophet', 'XGBoost']
        metrics_data = {model: {'rmse': [], 'mape': [], 'r2': []} for model in models}
        for (product_name, results) in product_results.items():
            for model_name in models:
                if model_name in results['models']:
                    model_results = results['models'][model_name]
                    metrics_data[model_name]['rmse'].append(model_results['regression_metrics']['RMSE'])
                    metrics_data[model_name]['mape'].append(model_results['regression_metrics']['MAPE'])
                    metrics_data[model_name]['r2'].append(model_results['regression_metrics']['R2'])
        (fig, axes) = plt.subplots(1, 3, figsize=(18, 6))
        rmse_data = [metrics_data[model]['rmse'] for model in models if metrics_data[model]['rmse']]
        axes[0].boxplot(rmse_data, labels=models)
        axes[0].set_title('RMSE Distribution Across Products', fontweight='bold')
        axes[0].set_ylabel('RMSE')
        axes[0].grid(True, alpha=0.3)
        mape_data = [metrics_data[model]['mape'] for model in models if metrics_data[model]['mape']]
        axes[1].boxplot(mape_data, labels=models)
        axes[1].set_title('MAPE Distribution Across Products', fontweight='bold')
        axes[1].set_ylabel('MAPE (%)')
        axes[1].grid(True, alpha=0.3)
        r2_data = [metrics_data[model]['r2'] for model in models if metrics_data[model]['r2']]
        axes[2].boxplot(r2_data, labels=models)
        axes[2].set_title('R² Distribution Across Products', fontweight='bold')
        axes[2].set_ylabel('R² Score')
        axes[2].grid(True, alpha=0.3)
        plt.suptitle('Model Performance Distribution Across All Products', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/charts/performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_win_rate_analysis(self, product_results):
        win_counts = {'ARIMA': 0, 'Prophet': 0, 'XGBoost': 0}
        total_products = len(product_results)
        for (product_name, results) in product_results.items():
            best_model = None
            best_rmse = float('inf')
            for (model_name, model_results) in results['models'].items():
                if model_name in win_counts:
                    rmse = model_results['regression_metrics']['RMSE']
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model_name
            if best_model:
                win_counts[best_model] += 1
        (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))
        models = list(win_counts.keys())
        wins = list(win_counts.values())
        bars = ax1.bar(models, wins, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Model Win Count (Best RMSE)', fontweight='bold')
        ax1.set_ylabel('Number of Products')
        for (bar, win) in zip(bars, wins):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f'{win}', ha='center', va='bottom')
        win_rates = {model: count / total_products * 100 for (model, count) in win_counts.items()}
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        (wedges, texts, autotexts) = ax2.pie(win_rates.values(), labels=win_rates.keys(), autopct='%1.1f%%', colors=colors)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax2.set_title('Model Win Rate Distribution', fontweight='bold')
        plt.suptitle(f'Model Win Rate Analysis (Total Products: {total_products})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/charts/win_rate_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_consistency_analysis(self, product_results):
        models = ['ARIMA', 'Prophet', 'XGBoost']
        consistency_data = []
        for (product_name, results) in product_results.items():
            for model_name in models:
                if model_name in results['models']:
                    rmse = results['models'][model_name]['regression_metrics']['RMSE']
                    consistency_data.append({'Product': product_name, 'Model': model_name, 'RMSE': rmse})
        if not consistency_data:
            return
        consistency_df = pd.DataFrame(consistency_data)
        pivot_table = consistency_df.pivot_table(values='RMSE', index='Product', columns='Model')
        plt.figure(figsize=(12, max(8, len(pivot_table) * 0.3)))
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd_r', cbar_kws={'label': 'RMSE (Lower is Better)'})
        plt.title('Model Performance Consistency Across Products', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/charts/consistency_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_heatmap(self, product_results):
        models = ['ARIMA', 'Prophet', 'XGBoost']
        ranking_data = []
        for (product_name, results) in product_results.items():
            product_rankings = {}
            for model_name in models:
                if model_name in results['models']:
                    rmse = results['models'][model_name]['regression_metrics']['RMSE']
                    product_rankings[model_name] = rmse
            if product_rankings:
                sorted_models = sorted(product_rankings.keys(), key=lambda x: product_rankings[x])
                for (rank, model_name) in enumerate(sorted_models, 1):
                    ranking_data.append({'Product': product_name, 'Model': model_name, 'Rank': rank, 'RMSE': product_rankings[model_name]})
        if not ranking_data:
            return
        ranking_df = pd.DataFrame(ranking_data)
        rank_pivot = ranking_df.pivot_table(values='Rank', index='Product', columns='Model')
        plt.figure(figsize=(10, max(6, len(rank_pivot) * 0.3)))
        sns.heatmap(rank_pivot, annot=True, fmt='d', cmap='RdYlGn_r', cbar_kws={'label': 'Rank (1 = Best)'})
        plt.title('Model Ranking Across Products (1 = Best Performance)', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/charts/ranking_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _save_comprehensive_reports(self, overall_results, product_results):
        import json
        with open(f'{self.output_dir}/reports/overall_model_performance.json', 'w') as f:
            json.dump(overall_results, f, indent=2)
        summary_data = []
        for (product_name, results) in product_results.items():
            for (model_name, model_results) in results['models'].items():
                summary_data.append({'product': product_name, 'model': model_name, 'rmse': model_results['regression_metrics']['RMSE'], 'mae': model_results['regression_metrics']['MAE'], 'mape': model_results['regression_metrics']['MAPE'], 'r2': model_results['regression_metrics']['R2'], 'accuracy': model_results['classification_metrics']['accuracy']})
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{self.output_dir}/reports/detailed_performance_report.csv', index=False)
        self._create_executive_summary(overall_results, summary_df)

    def _create_executive_summary(self, overall_results, summary_df):
        if 'overall_best_model' not in overall_results:
            return
        best_model = overall_results['overall_best_model']
        best_model_stats = overall_results[best_model]
        summary_text = f"\n        MODEL EVALUATION EXECUTIVE SUMMARY\n        =================================\n        \n        Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n        Total Products Evaluated: {best_model_stats['products_evaluated']}\n        \n        OVERALL BEST MODEL: {best_model}\n        -------------------------------\n        • Mean RMSE: {best_model_stats['mean_rmse']:.2f}\n        • Mean MAE: {best_model_stats['mean_mae']:.2f}\n        • Mean MAPE: {best_model_stats['mean_mape']:.2f}%\n        • Mean R²: {best_model_stats['mean_r2']:.3f}\n        • Mean Accuracy: {best_model_stats['mean_accuracy']:.3f}\n        \n        MODEL COMPARISON:\n        ----------------\n        "
        for (model_name, stats) in overall_results.items():
            if model_name != 'overall_best_model':
                summary_text += f"\n        {model_name}:\n          • Mean RMSE: {stats['mean_rmse']:.2f}\n          • Mean MAPE: {stats['mean_mape']:.2f}%\n          • Best Product: {stats['best_rmse_product']}\n        "
        with open(f'{self.output_dir}/reports/executive_summary.txt', 'w') as f:
            f.write(summary_text)

    def _calculate_classification_metrics(self, actual, predictions):
        errors = np.abs(actual - predictions)
        threshold = np.percentile(errors, 50)
        actual_class = (errors <= threshold).astype(int)
        predicted_class = (errors <= threshold).astype(int)
        accuracy = accuracy_score(actual_class, predicted_class)
        cm = confusion_matrix(actual_class, predicted_class)
        return {'accuracy': accuracy, 'confusion_matrix': cm.tolist(), 'threshold': threshold}

    def _calculate_r2(self, actual, predictions):
        ss_res = np.sum((actual - predictions) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0

    def _calculate_mase(self, actual, predictions, train):
        naive_errors = np.abs(np.diff(train))
        if len(naive_errors) == 0:
            return float('inf')
        mae_naive = np.mean(naive_errors)
        mae_model = mean_absolute_error(actual, predictions)
        return mae_model / mae_naive if mae_naive != 0 else float('inf')

    def _get_feature_importance(self, model, model_name):
        if model_name == 'XGBoost' and hasattr(model, 'model'):
            try:
                importance = model.model.feature_importances_
                feature_names = getattr(model, 'feature_columns', [f'feature_{i}' for i in range(len(importance))])
                return dict(zip(feature_names, importance.tolist()))
            except:
                return {}
        return {}
if __name__ == '__main__':
    evaluator = ModelEvaluator()
    df = pd.read_csv('data/THESIS-DATASET.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    (overall_results, product_results) = evaluator.evaluate_models_comprehensive(df)
    print('🎉 Comprehensive evaluation complete!')