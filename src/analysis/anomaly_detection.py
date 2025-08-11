"""
تشخیص ناهنجاری در رفتار مشتریان بانکی با استفاده از SHAP برای تفسیر
"""

import polars as pl
import pandas as pd  # Keep for sklearn compatibility
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

from ..database.sqlite_manager import SQLiteManager
from ..feature_engineering.extractors import BankingFeatureExtractor
from ..feature_engineering.transformers import FeatureTransformer
from ..utils.config import get_config

logger = logging.getLogger(__name__)

class BankingAnomalyDetector:
    """کلاس تشخیص ناهنجاری در رفتار مشتریان بانکی"""
    
    def __init__(self, db_manager: SQLiteManager = None):
        """مقداردهی اولیه"""
        self.db_manager = db_manager or SQLiteManager()
        self.config = get_config()
        
        # مدل‌های anomaly detection
        self.models = {}
        self.anomaly_results = {}
        self.shap_explainers = {}
        
        # ابزارهای کمکی
        self.feature_extractor = BankingFeatureExtractor(self.db_manager)
        self.feature_transformer = FeatureTransformer()
        
        # آمار تشخیص ناهنجاری
        self.detection_stats = {
            'models_fitted': 0,
            'anomalies_detected': 0,
            'best_model': None,
            'feature_importance': {}
        }
        
        logger.info("BankingAnomalyDetector initialized")
    
    def prepare_features_for_anomaly_detection(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """آماده‌سازی ویژگی‌ها برای تشخیص ناهنجاری"""
        logger.info("Preparing features for anomaly detection...")
        
        # دریافت ویژگی‌ها
        try:
            features_df = self.db_manager.execute_query("SELECT * FROM user_features")
            if features_df.empty:
                raise ValueError("No features found")
            logger.info(f"Loaded {len(features_df)} features from database")
        except:
            logger.info("Extracting features...")
            features_df = self.feature_extractor.extract_all_user_features()
            if not features_df.empty:
                self.feature_extractor.save_features_to_database(features_df)
        
        if features_df.empty:
            raise ValueError("Could not prepare features")
        
        # نمونه‌گیری
        if sample_size and len(features_df) > sample_size:
            features_df = features_df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} users")
        
        # تبدیل ویژگی‌ها
        transformed_features = self.feature_transformer.transform_pipeline(
            features_df,
            fit=True,
            steps=['missing_values', 'categorical_encoding', 'scaling']
        )
        
        # انتخاب ویژگی‌های مناسب برای anomaly detection
        detection_features = self._select_anomaly_features(transformed_features)
        
        logger.info(f"Prepared {len(detection_features.columns)} features for anomaly detection")
        
        return transformed_features, detection_features
    
    def _select_anomaly_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """انتخاب ویژگی‌های مناسب برای تشخیص ناهنجاری"""
        # ویژگی‌های کلیدی برای تشخیص ناهنجاری
        important_features = [
            'total_amount', 'avg_transaction_amount', 'total_transactions',
            'transaction_frequency', 'age', 'geographical_diversity',
            'night_transaction_ratio', 'weekend_transaction_ratio',
            'card_type_diversity', 'device_type_diversity',
            'amount_skewness', 'amount_iqr', 'regularity_score'
        ]
        
        # انتخاب ویژگی‌های موجود
        available_features = ['user_id'] + [f for f in important_features if f in features_df.columns]
        
        return features_df[available_features]
    
    def fit_isolation_forest(self, X: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """اجرای Isolation Forest"""
        logger.info(f"Fitting Isolation Forest (contamination={contamination})...")
        
        # حذف user_id
        X_clean = X.drop(columns=['user_id'], errors='ignore')
        
        # اجرای مدل
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        anomaly_labels = iso_forest.fit_predict(X_clean)
        anomaly_scores = iso_forest.decision_function(X_clean)
        
        # تبدیل labels (-1: anomaly, 1: normal) به (1: anomaly, 0: normal)
        is_anomaly = (anomaly_labels == -1).astype(int)
        
        results = {
            'model': iso_forest,
            'anomaly_labels': anomaly_labels,
            'is_anomaly': is_anomaly,
            'anomaly_scores': anomaly_scores,
            'contamination': contamination,
            'n_anomalies': np.sum(is_anomaly),
            'anomaly_percentage': np.mean(is_anomaly) * 100
        }
        
        self.models['isolation_forest'] = iso_forest
        self.anomaly_results['isolation_forest'] = results
        
        logger.info(f"Isolation Forest: {np.sum(is_anomaly)} anomalies ({np.mean(is_anomaly)*100:.2f}%)")
        
        return results
    
    def fit_one_class_svm(self, X: pd.DataFrame, nu: float = 0.1) -> Dict[str, Any]:
        """اجرای One-Class SVM"""
        logger.info(f"Fitting One-Class SVM (nu={nu})...")
        
        # حذف user_id
        X_clean = X.drop(columns=['user_id'], errors='ignore')
        
        # اجرای مدل
        oc_svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        
        anomaly_labels = oc_svm.fit_predict(X_clean)
        anomaly_scores = oc_svm.decision_function(X_clean)
        
        # تبدیل labels
        is_anomaly = (anomaly_labels == -1).astype(int)
        
        results = {
            'model': oc_svm,
            'anomaly_labels': anomaly_labels,
            'is_anomaly': is_anomaly,
            'anomaly_scores': anomaly_scores,
            'nu': nu,
            'n_anomalies': np.sum(is_anomaly),
            'anomaly_percentage': np.mean(is_anomaly) * 100
        }
        
        self.models['one_class_svm'] = oc_svm
        self.anomaly_results['one_class_svm'] = results
        
        logger.info(f"One-Class SVM: {np.sum(is_anomaly)} anomalies ({np.mean(is_anomaly)*100:.2f}%)")
        
        return results
    
    def create_ground_truth_labels(self, features_df: pd.DataFrame) -> np.ndarray:
        """ایجاد ground truth labels براساس نویزهای شناخته‌شده"""
        logger.info("Creating ground truth labels based on known noise...")
        
        # دریافت اطلاعات نویز از تراکنش‌ها
        noise_query = """
        SELECT user_id, COUNT(*) as noise_count, COUNT(*) * 1.0 / 
               (SELECT COUNT(*) FROM transactions t2 WHERE t2.user_id = t1.user_id) as noise_ratio
        FROM transactions t1 
        WHERE is_noise = 1 
        GROUP BY user_id
        """
        
        noise_df = self.db_manager.execute_query(noise_query)
        
        # ایجاد ground truth
        ground_truth = np.zeros(len(features_df))
        
        if not noise_df.empty:
            # کاربرانی که بیش از 10% تراکنش‌هایشان نویز است
            high_noise_users = noise_df[noise_df['noise_ratio'] > 0.1]['user_id'].values
            
            for i, user_id in enumerate(features_df['user_id']):
                if user_id in high_noise_users:
                    ground_truth[i] = 1
        
        logger.info(f"Created ground truth: {np.sum(ground_truth)} anomalous users")
        
        return ground_truth
    
    def fit_supervised_anomaly_detection(self, X: pd.DataFrame) -> Dict[str, Any]:
        """اجرای supervised anomaly detection"""
        logger.info("Fitting supervised anomaly detection...")
        
        # ایجاد ground truth
        ground_truth = self.create_ground_truth_labels(X)
        
        if np.sum(ground_truth) < 5:
            logger.warning("Not enough anomalous samples for supervised learning")
            return {}
        
        # حذف user_id
        X_clean = X.drop(columns=['user_id'], errors='ignore')
        
        # اجرای Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        rf.fit(X_clean, ground_truth)
        
        # پیش‌بینی
        anomaly_proba = rf.predict_proba(X_clean)[:, 1]
        is_anomaly = rf.predict(X_clean)
        
        results = {
            'model': rf,
            'is_anomaly': is_anomaly,
            'anomaly_scores': anomaly_proba,
            'ground_truth': ground_truth,
            'n_anomalies': np.sum(is_anomaly),
            'anomaly_percentage': np.mean(is_anomaly) * 100,
            'feature_importance': dict(zip(X_clean.columns, rf.feature_importances_))
        }
        
        self.models['supervised_rf'] = rf
        self.anomaly_results['supervised_rf'] = results
        
        logger.info(f"Supervised RF: {np.sum(is_anomaly)} anomalies ({np.mean(is_anomaly)*100:.2f}%)")
        
        return results
    
    def explain_anomalies_with_shap(self, X: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """تفسیر ناهنجاری‌ها با SHAP"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available for explanation")
            return {}
        
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found")
            return {}
        
        logger.info(f"Explaining anomalies with SHAP for {model_name}...")
        
        model = self.models[model_name]
        X_clean = X.drop(columns=['user_id'], errors='ignore')
        
        try:
            # انتخاب explainer مناسب
            if model_name == 'supervised_rf':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_clean)
                
                # برای classification، SHAP values کلاس anomaly
                if len(shap_values) == 2:  # binary classification
                    shap_values = shap_values[1]
            else:
                # برای unsupervised methods
                explainer = shap.KernelExplainer(model.decision_function, 
                                               shap.sample(X_clean, 100))
                shap_values = explainer.shap_values(X_clean[:500])  # محدود به 500 نمونه
            
            self.shap_explainers[model_name] = explainer
            
            # محاسبه feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            feature_names = X_clean.columns.tolist()
            
            importance_dict = dict(zip(feature_names, feature_importance))
            
            # شناسایی کاربران anomaly
            if model_name in self.anomaly_results:
                anomaly_indices = np.where(self.anomaly_results[model_name]['is_anomaly'] == 1)[0]
                
                if len(anomaly_indices) > 0:
                    # SHAP values برای کاربران anomaly
                    anomaly_shap_values = shap_values[anomaly_indices]
                    
                    explanation_results = {
                        'shap_values': shap_values,
                        'feature_importance': importance_dict,
                        'anomaly_shap_values': anomaly_shap_values,
                        'anomaly_indices': anomaly_indices,
                        'top_features': sorted(importance_dict.items(), 
                                             key=lambda x: x[1], reverse=True)[:10]
                    }
                    
                    logger.info(f"SHAP explanation completed for {model_name}")
                    return explanation_results
            
        except Exception as e:
            logger.error(f"Failed to explain with SHAP: {e}")
        
        return {}
    
    def visualize_anomaly_results(self, X: pd.DataFrame, model_name: str, 
                                save_path: str = None) -> plt.Figure:
        """تصویرسازی نتایج تشخیص ناهنجاری"""
        if model_name not in self.anomaly_results:
            logger.warning(f"No results for model: {model_name}")
            return None
        
        results = self.anomaly_results[model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. توزیع anomaly scores
        axes[0, 0].hist(results['anomaly_scores'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title(f'{model_name}: Anomaly Score Distribution')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. نسبت anomaly vs normal
        anomaly_counts = [np.sum(results['is_anomaly'] == 0), np.sum(results['is_anomaly'] == 1)]
        axes[0, 1].pie(anomaly_counts, labels=['Normal', 'Anomaly'], autopct='%1.1f%%',
                      colors=['lightgreen', 'red'], startangle=90)
        axes[0, 1].set_title('Normal vs Anomaly Distribution')
        
        # 3. Feature importance (اگر موجود باشد)
        if 'feature_importance' in results:
            importance = results['feature_importance']
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            features, scores = zip(*top_features)
            axes[1, 0].barh(range(len(features)), scores)
            axes[1, 0].set_yticks(range(len(features)))
            axes[1, 0].set_yticklabels(features)
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].set_xlabel('Importance Score')
        
        # 4. Anomaly scores vs feature
        X_clean = X.drop(columns=['user_id'], errors='ignore')
        if 'total_amount' in X_clean.columns:
            axes[1, 1].scatter(X_clean['total_amount'], results['anomaly_scores'], 
                              c=results['is_anomaly'], cmap='coolwarm', alpha=0.6)
            axes[1, 1].set_xlabel('Total Amount')
            axes[1, 1].set_ylabel('Anomaly Score')
            axes[1, 1].set_title('Anomaly Score vs Total Amount')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved anomaly visualization to {save_path}")
        
        return fig
    
    def save_anomaly_results(self, features_df: pd.DataFrame, model_name: str):
        """ذخیره نتایج تشخیص ناهنجاری در دیتابیس"""
        if model_name not in self.anomaly_results:
            logger.warning(f"No results for model: {model_name}")
            return
        
        results = self.anomaly_results[model_name]
        
        # آماده‌سازی داده
        anomaly_data = []
        
        for i, user_id in enumerate(features_df['user_id']):
            # SHAP values (اگر موجود باشد)
            shap_values_json = None
            if model_name in self.shap_explainers and 'shap_values' in results:
                shap_values_json = json.dumps(results['shap_values'][i].tolist())
            
            # Feature importance
            feature_importance_json = None
            if 'feature_importance' in results:
                feature_importance_json = json.dumps(results['feature_importance'])
            
            record = {
                'user_id': int(user_id),
                'anomaly_method': model_name,
                'anomaly_score': float(results['anomaly_scores'][i]),
                'is_anomaly': bool(results['is_anomaly'][i]),
                'shap_values': shap_values_json,
                'feature_importance': feature_importance_json
            }
            anomaly_data.append(record)
        
        anomaly_df = pd.DataFrame(anomaly_data)
        
        # ذخیره در دیتابیس
        with self.db_manager.get_connection() as conn:
            # حذف نتایج قبلی
            conn.execute("DELETE FROM anomaly_results WHERE anomaly_method = ?", (model_name,))
            
            # درج نتایج جدید
            anomaly_df.to_sql('anomaly_results', conn, if_exists='append', index=False)
            conn.commit()
        
        logger.info(f"Saved {len(anomaly_df)} anomaly results for {model_name}")
    
    def run_complete_anomaly_analysis(self, sample_size: Optional[int] = 10000) -> Dict[str, Any]:
        """اجرای تحلیل کامل تشخیص ناهنجاری"""
        logger.info("Starting complete anomaly detection analysis...")
        
        # آماده‌سازی ویژگی‌ها
        features_df, detection_features = self.prepare_features_for_anomaly_detection(sample_size)
        
        # اجرای روش‌های مختلف
        methods_results = {}
        
        # Isolation Forest
        iso_results = self.fit_isolation_forest(detection_features)
        if iso_results:
            methods_results['isolation_forest'] = iso_results
            self.detection_stats['models_fitted'] += 1
        
        # One-Class SVM
        svm_results = self.fit_one_class_svm(detection_features)
        if svm_results:
            methods_results['one_class_svm'] = svm_results
            self.detection_stats['models_fitted'] += 1
        
        # Supervised method
        supervised_results = self.fit_supervised_anomaly_detection(detection_features)
        if supervised_results:
            methods_results['supervised_rf'] = supervised_results
            self.detection_stats['models_fitted'] += 1
        
        # SHAP explanations
        shap_explanations = {}
        for method_name in methods_results.keys():
            explanation = self.explain_anomalies_with_shap(detection_features, method_name)
            if explanation:
                shap_explanations[method_name] = explanation
        
        # انتخاب بهترین روش
        best_method = self._select_best_method(methods_results)
        self.detection_stats['best_model'] = best_method
        
        # ذخیره نتایج بهترین روش
        if best_method:
            self.save_anomaly_results(features_df, best_method)
            
            # تصویرسازی
            self.visualize_anomaly_results(detection_features, best_method,
                                         f'output/plots/anomaly_{best_method}.png')
        
        # آمار کلی
        total_anomalies = 0
        if best_method and best_method in methods_results:
            total_anomalies = methods_results[best_method]['n_anomalies']
        
        self.detection_stats['anomalies_detected'] = total_anomalies
        
        # خلاصه نتایج
        analysis_summary = {
            'methods_results': methods_results,
            'shap_explanations': shap_explanations,
            'best_method': best_method,
            'total_anomalies_detected': total_anomalies,
            'detection_stats': self.detection_stats,
            'features_used': list(detection_features.columns),
            'sample_size': len(features_df)
        }
        
        logger.info("Complete anomaly detection analysis finished")
        
        return analysis_summary
    
    def _select_best_method(self, methods_results: Dict[str, Any]) -> str:
        """انتخاب بهترین روش تشخیص ناهنجاری"""
        if not methods_results:
            return None
        
        # اولویت با supervised method (اگر موجود باشد)
        if 'supervised_rf' in methods_results:
            return 'supervised_rf'
        
        # در غیر این صورت، روشی با بیشترین تعداد ناهنجاری
        best_method = max(methods_results.keys(), 
                         key=lambda x: methods_results[x]['n_anomalies'])
        
        return best_method 