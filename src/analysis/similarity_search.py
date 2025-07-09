"""
جستجوی تشابه کاربران بانکی و پیدا کردن الگوهای مشابه
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.manifold import TSNE
    import umap
    MANIFOLD_AVAILABLE = True
except ImportError:
    MANIFOLD_AVAILABLE = False
    logging.warning("TSNE/UMAP not available for visualization")

from ..database.sqlite_manager import SQLiteManager
from ..feature_engineering.extractors import BankingFeatureExtractor
from ..feature_engineering.transformers import FeatureTransformer
from ..data_generation.generators import NewUserGenerator
from ..utils.config import get_config

logger = logging.getLogger(__name__)

class BankingSimilaritySearch:
    """کلاس جستجوی تشابه کاربران بانکی"""
    
    def __init__(self, db_manager: SQLiteManager = None):
        """مقداردهی اولیه"""
        self.db_manager = db_manager or SQLiteManager()
        self.config = get_config()
        
        # مدل‌های similarity search
        self.similarity_models = {}
        self.similarity_results = {}
        
        # ابزارهای کمکی
        self.feature_extractor = BankingFeatureExtractor(self.db_manager)
        self.feature_transformer = FeatureTransformer()
        self.new_user_generator = NewUserGenerator()
        
        # آمار جستجوی تشابه
        self.search_stats = {
            'total_users': 0,
            'new_test_users': 0,
            'similarities_computed': 0,
            'avg_similarity_score': 0.0
        }
        
        logger.info("BankingSimilaritySearch initialized")
    
    def prepare_similarity_features(self, include_new_users: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """آماده‌سازی ویژگی‌ها برای جستجوی تشابه"""
        logger.info("Preparing features for similarity search...")
        
        # دریافت ویژگی‌های کاربران موجود
        try:
            existing_features = self.db_manager.execute_query("SELECT * FROM user_features")
            if existing_features.empty:
                raise ValueError("No existing features found")
            logger.info(f"Loaded {len(existing_features)} existing user features")
        except:
            logger.info("Extracting features for existing users...")
            existing_features = self.feature_extractor.extract_all_user_features()
            if not existing_features.empty:
                self.feature_extractor.save_features_to_database(existing_features)
        
        # تولید کاربران جدید برای تست (اگر درخواست شده)
        new_users_features = pd.DataFrame()
        if include_new_users:
            new_users_features = self._generate_test_users()
        
        # ترکیب ویژگی‌ها
        if not new_users_features.empty:
            all_features = pd.concat([existing_features, new_users_features], ignore_index=True)
            all_features['is_new_user'] = [False] * len(existing_features) + [True] * len(new_users_features)
        else:
            all_features = existing_features.copy()
            all_features['is_new_user'] = False
        
        # تبدیل ویژگی‌ها
        transformed_features = self.feature_transformer.transform_pipeline(
            all_features,
            fit=True,
            steps=['missing_values', 'categorical_encoding', 'scaling']
        )
        
        # انتخاب ویژگی‌های مناسب برای similarity
        similarity_features = self._select_similarity_features(transformed_features)
        
        logger.info(f"Prepared features: {len(existing_features)} existing + {len(new_users_features)} new users")
        
        return transformed_features, similarity_features
    
    def _generate_test_users(self, n_users: int = 100) -> pd.DataFrame:
        """تولید کاربران جدید برای تست similarity"""
        logger.info(f"Generating {n_users} new test users...")
        
        # تولید کاربران جدید
        new_users_data = self.new_user_generator.generate_new_users(n_users)
        
        if new_users_data.empty:
            logger.warning("Failed to generate new users")
            return pd.DataFrame()
        
        # تولید ویژگی‌های synthetic مستقیماً
        logger.info("Generating synthetic features for new users...")
        
        # ایجاد ویژگی‌های synthetic مشابه کاربران موجود
        new_features_list = []
        
        for _, user_row in new_users_data.iterrows():
            # تولید ویژگی‌های synthetic با توزیع‌های مشابه
            synthetic_features = {
                'user_id': user_row['user_id'],
                'age': user_row['age'],
                'birth_year': user_row['birth_year'],
                'total_amount': np.random.gamma(2, 50000),  # مبلغ کل تراکنش‌ها
                'avg_transaction_amount': np.random.lognormal(10, 1),  # میانگین مبلغ
                'total_transactions': np.random.poisson(100),  # تعداد تراکنش‌ها
                'transaction_frequency': np.random.exponential(2),  # فرکانس تراکنش
                'geographical_diversity': np.random.uniform(0, 1),  # تنوع جغرافیایی
                'night_transaction_ratio': np.random.beta(2, 5),  # نسبت تراکنش شبانه
                'weekend_transaction_ratio': np.random.beta(3, 4),  # نسبت تراکنش آخر هفته
                'card_type_diversity': np.random.uniform(0, 1),  # تنوع نوع کارت
                'device_type_diversity': np.random.uniform(0, 1),  # تنوع نوع دستگاه
                'regularity_score': np.random.beta(4, 2),  # امتیاز منظمی
                'amount_skewness': np.random.normal(1, 0.5),  # چولگی مبلغ
                'median_transaction_amount': np.random.lognormal(9.5, 1),
                'std_amount': np.random.gamma(1.5, 10000),
                'min_amount': np.random.uniform(1000, 10000),
                'max_amount': np.random.gamma(3, 100000),
                'amount_range': 0,  # محاسبه می‌شود
                'std_hour': np.random.uniform(0, 12),
                'avg_daily_transactions': np.random.poisson(3),
                'daily_std': np.random.exponential(1),
                'hourly_entropy': np.random.uniform(0, 4),
                'day_of_week_entropy': np.random.uniform(0, 3),
                'monthly_growth_rate': np.random.normal(0, 0.1),
                'days_since_registration': np.random.randint(30, 365),
                'last_transaction_days_ago': np.random.randint(1, 30)
            }
            
            # محاسبه amount_range
            synthetic_features['amount_range'] = synthetic_features['max_amount'] - synthetic_features['min_amount']
            
            new_features_list.append(synthetic_features)
        
        if new_features_list:
            new_features_df = pd.DataFrame(new_features_list)
            self.search_stats['new_test_users'] = len(new_features_df)
            logger.info(f"Generated synthetic features for {len(new_features_df)} new users")
            return new_features_df
        
        return pd.DataFrame()
    
    def _select_similarity_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """انتخاب ویژگی‌های مناسب برای محاسبه تشابه"""
        # ویژگی‌های کلیدی برای similarity
        similarity_features = [
            'total_amount', 'avg_transaction_amount', 'total_transactions',
            'transaction_frequency', 'age', 'geographical_diversity',
            'night_transaction_ratio', 'weekend_transaction_ratio',
            'card_type_diversity', 'device_type_diversity',
            'regularity_score', 'amount_skewness'
        ]
        
        # انتخاب ویژگی‌های موجود + شناسه‌ها
        available_features = ['user_id', 'is_new_user'] + [f for f in similarity_features if f in features_df.columns]
        
        return features_df[available_features]
    
    def build_knn_index(self, X: pd.DataFrame, n_neighbors: int = 10, 
                       metric: str = 'cosine') -> NearestNeighbors:
        """ساخت index برای جستجوی سریع k-nearest neighbors"""
        logger.info(f"Building KNN index with {n_neighbors} neighbors, metric={metric}")
        
        # حذف ستون‌های غیرعددی
        X_clean = X.drop(columns=['user_id', 'is_new_user'], errors='ignore')
        
        # ساخت مدل KNN
        knn_model = NearestNeighbors(
            n_neighbors=min(n_neighbors, len(X_clean)),
            metric=metric,
            n_jobs=-1
        )
        
        knn_model.fit(X_clean)
        
        self.similarity_models['knn'] = {
            'model': knn_model,
            'features': X_clean,
            'user_ids': X['user_id'].values,
            'is_new_user': X['is_new_user'].values if 'is_new_user' in X.columns else None
        }
        
        logger.info("KNN index built successfully")
        
        return knn_model
    
    def find_similar_users(self, target_user_id: int, n_similar: int = 10) -> Dict[str, Any]:
        """یافتن کاربران مشابه برای یک کاربر خاص"""
        if 'knn' not in self.similarity_models:
            logger.error("KNN model not built. Call build_knn_index first.")
            return {}
        
        knn_data = self.similarity_models['knn']
        knn_model = knn_data['model']
        user_ids = knn_data['user_ids']
        features = knn_data['features']
        
        # یافتن index کاربر هدف
        target_indices = np.where(user_ids == target_user_id)[0]
        
        if len(target_indices) == 0:
            logger.warning(f"User {target_user_id} not found in the dataset")
            return {}
        
        target_index = target_indices[0]
        target_features = features.iloc[target_index:target_index+1]
        
        # جستجوی کاربران مشابه
        distances, indices = knn_model.kneighbors(target_features, n_neighbors=n_similar+1)
        
        # حذف خود کاربر از نتایج
        similar_indices = indices[0][1:]  # حذف اولین (خود کاربر)
        similar_distances = distances[0][1:]
        
        # آماده‌سازی نتایج
        similar_users = []
        for i, (idx, distance) in enumerate(zip(similar_indices, similar_distances)):
            similar_user = {
                'rank': i + 1,
                'user_id': int(user_ids[idx]),
                'similarity_score': float(1 - distance),  # تبدیل distance به similarity
                'distance': float(distance),
                'is_new_user': bool(knn_data['is_new_user'][idx]) if knn_data['is_new_user'] is not None else False
            }
            similar_users.append(similar_user)
        
        result = {
            'target_user_id': target_user_id,
            'similar_users': similar_users,
            'avg_similarity': float(np.mean([u['similarity_score'] for u in similar_users])),
            'search_method': 'knn_cosine'
        }
        
        logger.info(f"Found {len(similar_users)} similar users for user {target_user_id}")
        
        return result
    
    def find_similar_users_for_new_users(self, n_similar: int = 10) -> List[Dict[str, Any]]:
        """یافتن کاربران مشابه برای تمام کاربران جدید"""
        if 'knn' not in self.similarity_models:
            logger.error("KNN model not built")
            return []
        
        knn_data = self.similarity_models['knn']
        user_ids = knn_data['user_ids']
        is_new_user = knn_data['is_new_user']
        
        if is_new_user is None:
            logger.warning("No new users identified")
            return []
        
        # تبدیل به boolean array و یافتن کاربران جدید
        is_new_user_bool = np.array(is_new_user, dtype=bool)
        new_user_ids = user_ids[is_new_user_bool]
        
        logger.info(f"Finding similar users for {len(new_user_ids)} new users...")
        
        all_results = []
        
        for user_id in new_user_ids:
            similarity_result = self.find_similar_users(user_id, n_similar)
            if similarity_result:
                all_results.append(similarity_result)
        
        logger.info(f"Completed similarity search for {len(all_results)} new users")
        
        return all_results
    
    def compute_pairwise_similarities(self, X: pd.DataFrame, method: str = 'cosine') -> np.ndarray:
        """محاسبه ماتریس تشابه برای همه جفت کاربران"""
        logger.info(f"Computing pairwise similarities using {method}...")
        
        X_clean = X.drop(columns=['user_id', 'is_new_user'], errors='ignore')
        
        if method == 'cosine':
            similarity_matrix = cosine_similarity(X_clean)
        elif method == 'euclidean':
            distances = euclidean_distances(X_clean)
            # تبدیل distance به similarity
            similarity_matrix = 1 / (1 + distances)
        else:
            logger.warning(f"Unknown similarity method: {method}")
            similarity_matrix = cosine_similarity(X_clean)
        
        logger.info(f"Computed {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} similarity matrix")
        
        return similarity_matrix
    
    def analyze_similarity_patterns(self, similarity_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """تحلیل الگوهای تشابه"""
        if not similarity_results:
            return {}
        
        logger.info("Analyzing similarity patterns...")
        
        # جمع‌آوری آمار
        all_similarities = []
        new_to_existing_similarities = []
        new_to_new_similarities = []
        
        for result in similarity_results:
            target_is_new = True  # کاربران جدید را بررسی می‌کنیم
            
            for similar_user in result['similar_users']:
                similarity_score = similar_user['similarity_score']
                all_similarities.append(similarity_score)
                
                if similar_user['is_new_user']:
                    new_to_new_similarities.append(similarity_score)
                else:
                    new_to_existing_similarities.append(similarity_score)
        
        # محاسبه آمار
        analysis = {
            'total_comparisons': len(all_similarities),
            'overall_stats': {
                'mean_similarity': float(np.mean(all_similarities)),
                'std_similarity': float(np.std(all_similarities)),
                'min_similarity': float(np.min(all_similarities)),
                'max_similarity': float(np.max(all_similarities))
            },
            'new_to_existing': {
                'count': len(new_to_existing_similarities),
                'mean_similarity': float(np.mean(new_to_existing_similarities)) if new_to_existing_similarities else 0,
                'percentage': len(new_to_existing_similarities) / len(all_similarities) * 100 if all_similarities else 0
            },
            'new_to_new': {
                'count': len(new_to_new_similarities),
                'mean_similarity': float(np.mean(new_to_new_similarities)) if new_to_new_similarities else 0,
                'percentage': len(new_to_new_similarities) / len(all_similarities) * 100 if all_similarities else 0
            }
        }
        
        # شناسایی الگوهای جالب
        high_similarity_threshold = 0.8
        low_similarity_threshold = 0.3
        
        high_similarities = [s for s in all_similarities if s > high_similarity_threshold]
        low_similarities = [s for s in all_similarities if s < low_similarity_threshold]
        
        analysis['patterns'] = {
            'high_similarity_count': len(high_similarities),
            'low_similarity_count': len(low_similarities),
            'similarity_distribution': {
                'very_high (>0.8)': len([s for s in all_similarities if s > 0.8]),
                'high (0.6-0.8)': len([s for s in all_similarities if 0.6 <= s <= 0.8]),
                'medium (0.4-0.6)': len([s for s in all_similarities if 0.4 <= s <= 0.6]),
                'low (<0.4)': len([s for s in all_similarities if s < 0.4])
            }
        }
        
        logger.info("Similarity pattern analysis completed")
        
        return analysis
    
    def visualize_similarity_results(self, X: pd.DataFrame, similarity_results: List[Dict[str, Any]], 
                                   save_path: str = None) -> plt.Figure:
        """تصویرسازی نتایج جستجوی تشابه"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. توزیع امتیازات تشابه
        all_similarities = []
        for result in similarity_results:
            for similar_user in result['similar_users']:
                all_similarities.append(similar_user['similarity_score'])
        
        if all_similarities:
            axes[0, 0].hist(all_similarities, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Distribution of Similarity Scores')
            axes[0, 0].set_xlabel('Similarity Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. مقایسه تشابه بین گروه‌های مختلف
        if similarity_results:
            new_to_existing = []
            new_to_new = []
            
            for result in similarity_results:
                for similar_user in result['similar_users']:
                    if similar_user['is_new_user']:
                        new_to_new.append(similar_user['similarity_score'])
                    else:
                        new_to_existing.append(similar_user['similarity_score'])
            
            if new_to_existing and new_to_new:
                axes[0, 1].boxplot([new_to_existing, new_to_new], 
                                 labels=['New to Existing', 'New to New'])
                axes[0, 1].set_title('Similarity Score Comparison')
                axes[0, 1].set_ylabel('Similarity Score')
        
        # 3. نمودار scatter similarity vs rank
        if similarity_results:
            ranks = []
            similarities = []
            
            for result in similarity_results:
                for similar_user in result['similar_users']:
                    ranks.append(similar_user['rank'])
                    similarities.append(similar_user['similarity_score'])
            
            if ranks and similarities:
                axes[1, 0].scatter(ranks, similarities, alpha=0.6, color='green')
                axes[1, 0].set_title('Similarity Score vs Rank')
                axes[1, 0].set_xlabel('Rank')
                axes[1, 0].set_ylabel('Similarity Score')
                axes[1, 0].grid(True, alpha=0.3)
        
        # 4. نمایش ماتریس تشابه (نمونه)
        if len(X) <= 100:  # فقط برای dataset های کوچک
            similarity_matrix = self.compute_pairwise_similarities(X)
            im = axes[1, 1].imshow(similarity_matrix, cmap='coolwarm', aspect='auto')
            axes[1, 1].set_title('Similarity Matrix (Sample)')
            plt.colorbar(im, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, 'Similarity Matrix\n(Too large to display)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Similarity Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved similarity visualization to {save_path}")
        
        return fig
    
    def save_similarity_results(self, similarity_results: List[Dict[str, Any]]):
        """ذخیره نتایج جستجوی تشابه در دیتابیس"""
        if not similarity_results:
            logger.warning("No similarity results to save")
            return
        
        logger.info(f"Saving {len(similarity_results)} similarity results...")
        
        # آماده‌سازی داده
        similarity_data = []
        
        for result in similarity_results:
            target_user_id = result['target_user_id']
            
            for similar_user in result['similar_users']:
                record = {
                    'query_user_id': target_user_id,
                    'similar_user_id': similar_user['user_id'],
                    'similarity_score': similar_user['similarity_score'],
                    'similarity_method': result['search_method']
                }
                similarity_data.append(record)
        
        similarity_df = pd.DataFrame(similarity_data)
        
        # ذخیره در دیتابیس
        with self.db_manager.get_connection() as conn:
            # حذف نتایج قبلی
            conn.execute("DELETE FROM similarity_results")
            
            # درج نتایج جدید
            similarity_df.to_sql('similarity_results', conn, if_exists='append', index=False)
            conn.commit()
        
        logger.info(f"Saved {len(similarity_df)} similarity records")
    
    def run_complete_similarity_analysis(self, n_similar: int = 10) -> Dict[str, Any]:
        """اجرای تحلیل کامل جستجوی تشابه"""
        logger.info("Starting complete similarity search analysis...")
        
        # آماده‌سازی ویژگی‌ها
        all_features, similarity_features = self.prepare_similarity_features(include_new_users=True)
        
        # ساخت KNN index
        knn_model = self.build_knn_index(similarity_features, n_neighbors=n_similar)
        
        # یافتن کاربران مشابه برای کاربران جدید
        similarity_results = self.find_similar_users_for_new_users(n_similar)
        
        # تحلیل الگوهای تشابه
        pattern_analysis = self.analyze_similarity_patterns(similarity_results)
        
        # ذخیره نتایج
        self.save_similarity_results(similarity_results)
        
        # تصویرسازی
        self.visualize_similarity_results(similarity_features, similarity_results,
                                        'output/plots/similarity_analysis.png')
        
        # بروزرسانی آمار
        self.search_stats['total_users'] = len(all_features)
        self.search_stats['similarities_computed'] = len(similarity_results) * n_similar
        if pattern_analysis and 'overall_stats' in pattern_analysis:
            self.search_stats['avg_similarity_score'] = pattern_analysis['overall_stats']['mean_similarity']
        
        # خلاصه نتایج
        analysis_summary = {
            'similarity_results': similarity_results,
            'pattern_analysis': pattern_analysis,
            'search_stats': self.search_stats,
            'total_users': len(all_features),
            'new_test_users': self.search_stats['new_test_users'],
            'features_used': list(similarity_features.columns),
            'knn_model_info': {
                'n_neighbors': n_similar,
                'metric': 'cosine',
                'n_samples': len(similarity_features)
            }
        }
        
        logger.info("Complete similarity search analysis finished")
        
        return analysis_summary 