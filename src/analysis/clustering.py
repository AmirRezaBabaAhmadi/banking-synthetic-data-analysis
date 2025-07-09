"""
دسته‌بندی کاربران بانکی با الگوریتم‌های مختلف clustering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    from hdbscan import HDBSCAN
    ADVANCED_CLUSTERING_AVAILABLE = True
except ImportError:
    ADVANCED_CLUSTERING_AVAILABLE = False
    logging.warning("UMAP/HDBSCAN not available. Install with: pip install umap-learn hdbscan")

from ..database.sqlite_manager import SQLiteManager
from ..feature_engineering.extractors import BankingFeatureExtractor
from ..feature_engineering.transformers import FeatureTransformer
from ..utils.config import get_config

logger = logging.getLogger(__name__)

class BankingCustomerClustering:
    """کلاس دسته‌بندی مشتریان بانکی"""
    
    def __init__(self, db_manager: SQLiteManager = None):
        """مقداردهی اولیه"""
        self.db_manager = db_manager or SQLiteManager()
        self.config = get_config()
        
        # ابزارهای clustering
        self.models = {}
        self.cluster_results = {}
        self.feature_extractor = BankingFeatureExtractor(self.db_manager)
        self.feature_transformer = FeatureTransformer()
        
        # آمار clustering
        self.clustering_stats = {
            'models_fitted': 0,
            'best_model': None,
            'best_score': -1,
            'cluster_profiles': {}
        }
        
        logger.info("BankingCustomerClustering initialized")
    
    def prepare_features_for_clustering(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """آماده‌سازی ویژگی‌ها برای clustering"""
        logger.info("Preparing features for clustering...")
        
        # دریافت یا استخراج ویژگی‌ها
        try:
            features_df = self.db_manager.execute_query("SELECT * FROM user_features")
            if features_df.empty:
                raise ValueError("No features found in database")
            logger.info(f"Loaded {len(features_df)} features from database")
        except:
            logger.info("Extracting features from scratch...")
            features_df = self.feature_extractor.extract_all_user_features()
            if not features_df.empty:
                self.feature_extractor.save_features_to_database(features_df)
        
        if features_df.empty:
            raise ValueError("Could not prepare features for clustering")
        
        # نمونه‌گیری در صورت لزوم
        if sample_size and len(features_df) > sample_size:
            features_df = features_df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} users for clustering")
        
        # تبدیل ویژگی‌ها
        transformed_features = self.feature_transformer.transform_pipeline(
            features_df, 
            fit=True,
            steps=['missing_values', 'categorical_encoding', 'scaling']
        )
        
        # حذف user_id برای clustering
        clustering_features = transformed_features.drop(columns=['user_id'], errors='ignore')
        
        logger.info(f"Prepared {len(clustering_features.columns)} features for clustering")
        
        return transformed_features, clustering_features
    
    def find_optimal_k_means(self, X: pd.DataFrame, max_k: int = 10) -> int:
        """یافتن تعداد بهینه cluster برای K-means"""
        logger.info(f"Finding optimal number of clusters (max_k={max_k})")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(X) // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            
            if len(np.unique(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(X, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(-1)
        
        # انتخاب بهترین k براساس silhouette score
        best_k = k_range[np.argmax(silhouette_scores)]
        
        logger.info(f"Optimal k found: {best_k} (silhouette score: {max(silhouette_scores):.3f})")
        
        return best_k
    
    def fit_kmeans_clustering(self, X: pd.DataFrame, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """اجرای K-means clustering"""
        logger.info("Fitting K-means clustering...")
        
        if n_clusters is None:
            n_clusters = self.find_optimal_k_means(X)
        
        # اجرای K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # محاسبه metrics
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X, cluster_labels)
        davies_bouldin = davies_bouldin_score(X, cluster_labels)
        
        results = {
            'model': kmeans,
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'metrics': {
                'silhouette_score': silhouette_avg,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'inertia': kmeans.inertia_
            }
        }
        
        self.models['kmeans'] = kmeans
        self.cluster_results['kmeans'] = results
        
        logger.info(f"K-means completed: {n_clusters} clusters, silhouette={silhouette_avg:.3f}")
        
        return results
    
    def fit_hierarchical_clustering(self, X: pd.DataFrame, n_clusters: int = 5) -> Dict[str, Any]:
        """اجرای Hierarchical clustering"""
        logger.info("Fitting Hierarchical clustering...")
        
        # اجرای Agglomerative clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = hierarchical.fit_predict(X)
        
        # محاسبه metrics
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X, cluster_labels)
        davies_bouldin = davies_bouldin_score(X, cluster_labels)
        
        results = {
            'model': hierarchical,
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'metrics': {
                'silhouette_score': silhouette_avg,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin
            }
        }
        
        self.models['hierarchical'] = hierarchical
        self.cluster_results['hierarchical'] = results
        
        logger.info(f"Hierarchical completed: {n_clusters} clusters, silhouette={silhouette_avg:.3f}")
        
        return results
    
    def fit_dbscan_clustering(self, X: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
        """اجرای DBSCAN clustering"""
        logger.info("Fitting DBSCAN clustering...")
        
        # اجرای DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # محاسبه metrics (فقط اگر بیش از یک cluster وجود داشته باشد)
        metrics = {}
        if n_clusters > 1:
            # حذف noise points برای محاسبه silhouette
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) > 1:
                silhouette_avg = silhouette_score(X[non_noise_mask], cluster_labels[non_noise_mask])
                metrics['silhouette_score'] = silhouette_avg
        
        results = {
            'model': dbscan,
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'metrics': metrics
        }
        
        self.models['dbscan'] = dbscan
        self.cluster_results['dbscan'] = results
        
        logger.info(f"DBSCAN completed: {n_clusters} clusters, {n_noise} noise points")
        
        return results
    
    def fit_hdbscan_clustering(self, X: pd.DataFrame, min_cluster_size: int = 10) -> Dict[str, Any]:
        """اجرای HDBSCAN clustering (اگر در دسترس باشد)"""
        if not ADVANCED_CLUSTERING_AVAILABLE:
            logger.warning("HDBSCAN not available")
            return {}
        
        logger.info("Fitting HDBSCAN clustering...")
        
        # اجرای HDBSCAN
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5)
        cluster_labels = hdbscan.fit_predict(X)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # محاسبه metrics
        metrics = {}
        if n_clusters > 1:
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) > 1:
                silhouette_avg = silhouette_score(X[non_noise_mask], cluster_labels[non_noise_mask])
                metrics['silhouette_score'] = silhouette_avg
        
        results = {
            'model': hdbscan,
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'metrics': metrics
        }
        
        self.models['hdbscan'] = hdbscan
        self.cluster_results['hdbscan'] = results
        
        logger.info(f"HDBSCAN completed: {n_clusters} clusters, {n_noise} noise points")
        
        return results
    
    def compare_clustering_methods(self, X: pd.DataFrame) -> Dict[str, Any]:
        """مقایسه روش‌های مختلف clustering"""
        logger.info("Comparing different clustering methods...")
        
        # اجرای روش‌های مختلف
        methods = {
            'kmeans': lambda: self.fit_kmeans_clustering(X),
            'hierarchical': lambda: self.fit_hierarchical_clustering(X),
            'dbscan': lambda: self.fit_dbscan_clustering(X),
        }
        
        if ADVANCED_CLUSTERING_AVAILABLE:
            methods['hdbscan'] = lambda: self.fit_hdbscan_clustering(X)
        
        comparison_results = {}
        
        for method_name, method_func in methods.items():
            try:
                result = method_func()
                comparison_results[method_name] = result
                self.clustering_stats['models_fitted'] += 1
            except Exception as e:
                logger.warning(f"Failed to fit {method_name}: {e}")
        
        # یافتن بهترین روش براساس silhouette score
        best_method = None
        best_score = -1
        
        for method_name, result in comparison_results.items():
            if 'metrics' in result and 'silhouette_score' in result['metrics']:
                score = result['metrics']['silhouette_score']
                if score > best_score:
                    best_score = score
                    best_method = method_name
        
        self.clustering_stats['best_model'] = best_method
        self.clustering_stats['best_score'] = best_score
        
        logger.info(f"Best clustering method: {best_method} (silhouette={best_score:.3f})")
        
        return comparison_results
    
    def analyze_cluster_profiles(self, features_df: pd.DataFrame, 
                                cluster_labels: np.ndarray,
                                method_name: str) -> Dict[str, Any]:
        """تحلیل profile های cluster ها"""
        logger.info(f"Analyzing cluster profiles for {method_name}...")
        
        # اضافه کردن cluster labels به features
        df_with_clusters = features_df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # حذف user_id برای تحلیل
        analysis_features = [col for col in df_with_clusters.columns 
                           if col not in ['user_id', 'cluster']]
        
        cluster_profiles = {}
        
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # noise cluster در DBSCAN
                continue
                
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_with_clusters) * 100,
                'feature_means': {},
                'dominant_characteristics': []
            }
            
            # محاسبه میانگین ویژگی‌ها برای این cluster
            for feature in analysis_features:
                if pd.api.types.is_numeric_dtype(df_with_clusters[feature]):
                    cluster_mean = cluster_data[feature].mean()
                    overall_mean = df_with_clusters[feature].mean()
                    profile['feature_means'][feature] = {
                        'cluster_mean': cluster_mean,
                        'overall_mean': overall_mean,
                        'difference_ratio': cluster_mean / overall_mean if overall_mean != 0 else 1
                    }
            
            # شناسایی ویژگی‌های مشخصه
            for feature, stats in profile['feature_means'].items():
                if abs(stats['difference_ratio'] - 1) > 0.3:  # 30% اختلاف
                    characteristic = {
                        'feature': feature,
                        'ratio': stats['difference_ratio'],
                        'direction': 'high' if stats['difference_ratio'] > 1 else 'low'
                    }
                    profile['dominant_characteristics'].append(characteristic)
            
            cluster_profiles[f'cluster_{cluster_id}'] = profile
        
        self.clustering_stats['cluster_profiles'][method_name] = cluster_profiles
        
        logger.info(f"Analyzed {len(cluster_profiles)} cluster profiles")
        
        return cluster_profiles
    
    def save_clustering_results(self, features_df: pd.DataFrame, method_name: str):
        """ذخیره نتایج clustering در دیتابیس"""
        if method_name not in self.cluster_results:
            logger.warning(f"No results found for method: {method_name}")
            return
        
        result = self.cluster_results[method_name]
        cluster_labels = result['labels']
        
        # آماده‌سازی داده برای ذخیره
        clustering_data = []
        
        for i, (user_id, cluster_label) in enumerate(zip(features_df['user_id'], cluster_labels)):
            record = {
                'user_id': int(user_id),
                'cluster_id': int(cluster_label),
                'cluster_method': method_name,
                'distance_to_center': None,  # می‌توان محاسبه کرد
                'cluster_probability': None  # برای روش‌هایی که احتمال می‌دهند
            }
            clustering_data.append(record)
        
        clustering_df = pd.DataFrame(clustering_data)
        
        # ذخیره در دیتابیس
        with self.db_manager.get_connection() as conn:
            # حذف نتایج قبلی برای این روش
            conn.execute("DELETE FROM clustering_results WHERE cluster_method = ?", (method_name,))
            
            # درج نتایج جدید
            clustering_df.to_sql('clustering_results', conn, if_exists='append', index=False)
            conn.commit()
        
        logger.info(f"Saved {len(clustering_df)} clustering results for {method_name}")
    
    def visualize_clusters(self, X: pd.DataFrame, cluster_labels: np.ndarray, 
                          method_name: str, save_path: str = None) -> plt.Figure:
        """تصویرسازی cluster ها"""
        # کاهش ابعاد برای visualization
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X)
            explained_variance = pca.explained_variance_ratio_.sum()
        else:
            X_reduced = X.values
            explained_variance = 1.0
        
        # ایجاد نمودار
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # نمودار scatter clusters
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:  # noise points
                ax1.scatter(X_reduced[cluster_labels == label, 0], 
                           X_reduced[cluster_labels == label, 1],
                           c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                ax1.scatter(X_reduced[cluster_labels == label, 0], 
                           X_reduced[cluster_labels == label, 1],
                           c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
        
        ax1.set_title(f'{method_name} Clustering\n(PCA: {explained_variance:.2f} variance explained)')
        ax1.set_xlabel('First Component')
        ax1.set_ylabel('Second Component')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # نمودار اندازه cluster ها
        cluster_sizes = []
        cluster_ids = []
        
        for label in unique_labels:
            if label != -1:
                cluster_sizes.append(np.sum(cluster_labels == label))
                cluster_ids.append(f'Cluster {label}')
        
        ax2.pie(cluster_sizes, labels=cluster_ids, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Cluster Size Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster visualization to {save_path}")
        
        return fig
    
    def run_complete_clustering_analysis(self, sample_size: Optional[int] = 10000) -> Dict[str, Any]:
        """اجرای تحلیل کامل clustering"""
        logger.info("Starting complete clustering analysis...")
        
        # آماده‌سازی ویژگی‌ها
        features_df, clustering_features = self.prepare_features_for_clustering(sample_size)
        
        # مقایسه روش‌های clustering
        comparison_results = self.compare_clustering_methods(clustering_features)
        
        # تحلیل profiles برای بهترین روش
        best_method = self.clustering_stats['best_model']
        if best_method and best_method in comparison_results:
            best_labels = comparison_results[best_method]['labels']
            
            # تحلیل cluster profiles
            profiles = self.analyze_cluster_profiles(features_df, best_labels, best_method)
            
            # ذخیره نتایج
            self.save_clustering_results(features_df, best_method)
            
            # تصویرسازی
            self.visualize_clusters(clustering_features, best_labels, best_method,
                                  f'output/plots/clustering_{best_method}.png')
        
        # خلاصه نتایج
        analysis_summary = {
            'comparison_results': comparison_results,
            'best_method': best_method,
            'best_score': self.clustering_stats['best_score'],
            'cluster_profiles': self.clustering_stats['cluster_profiles'],
            'features_used': list(clustering_features.columns),
            'sample_size': len(features_df)
        }
        
        logger.info("Complete clustering analysis finished")
        
        return analysis_summary 