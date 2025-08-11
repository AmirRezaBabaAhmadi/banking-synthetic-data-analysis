"""
استخراج ویژگی‌های پیشرفته از داده‌های بانکی برای تحلیل و مدل‌سازی
"""

import polars as pl
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..database.sqlite_manager import SQLiteManager
from ..utils.config import get_config

logger = logging.getLogger(__name__)

class BankingFeatureExtractor:
    """کلاس استخراج ویژگی‌های پیشرفته از داده‌های بانکی"""
    
    def __init__(self, db_manager: SQLiteManager = None):
        """مقداردهی اولیه"""
        self.db_manager = db_manager or SQLiteManager()
        self.config = get_config()
        
        # آمار استخراج ویژگی
        self.extraction_stats = {
            'features_extracted': 0,
            'users_processed': 0,
            'feature_names': []
        }
        
        logger.info("BankingFeatureExtractor initialized")
    
    def extract_transaction_volume_features(self, user_transactions: pl.DataFrame) -> Dict[str, float]:
        """استخراج ویژگی‌های مربوط به حجم تراکنش"""
        if user_transactions.is_empty():
            return {
                'total_transactions': 0,
                'total_amount': 0.0,
                'avg_transaction_amount': 0.0,
                'std_transaction_amount': 0.0,
                'min_transaction_amount': 0.0,
                'max_transaction_amount': 0.0,
                'median_transaction_amount': 0.0
            }
        
        amounts = user_transactions.get_column('amount')
        
        # Handle NaN values
        std_amount = amounts.std() or 0.0
        
        return {
            'total_transactions': len(user_transactions),
            'total_amount': float(amounts.sum()),
            'avg_transaction_amount': float(amounts.mean()),
            'std_transaction_amount': float(std_amount),
            'min_transaction_amount': float(amounts.min()),
            'max_transaction_amount': float(amounts.max()),
            'median_transaction_amount': float(amounts.median())
        }
    
    def extract_temporal_features(self, user_transactions: pl.DataFrame) -> Dict[str, float]:
        """استخراج ویژگی‌های زمانی"""
        if user_transactions.is_empty():
            return {
                'avg_hour_of_day': 12.0,
                'std_hour_of_day': 0.0,
                'weekend_transaction_ratio': 0.0,
                'night_transaction_ratio': 0.0,
                'transaction_frequency': 0.0
            }
        
        hours = user_transactions.get_column('hour_of_day')
        is_weekend = user_transactions.get_column('is_weekend')
        
        # محاسبه نسبت تراکنش شبانه
        night_mask = (hours >= 22) | (hours <= 6)
        night_ratio = night_mask.mean()
        
        # محاسبه فرکانس تراکنش
        unique_dates = user_transactions.get_column('transaction_date').n_unique()
        transaction_frequency = len(user_transactions) / max(unique_dates, 1)
        
        # Handle NaN values
        std_hour = hours.std() or 0.0
        
        return {
            'avg_hour_of_day': float(hours.mean()),
            'std_hour_of_day': float(std_hour),
            'weekend_transaction_ratio': float(is_weekend.mean()),
            'night_transaction_ratio': float(night_ratio),
            'transaction_frequency': float(transaction_frequency)
        }
    
    def extract_geographical_features(self, user_transactions: pl.DataFrame) -> Dict[str, Any]:
        """استخراج ویژگی‌های جغرافیایی"""
        if user_transactions.is_empty():
            return {
                'unique_cities_count': 0,
                'unique_provinces_count': 0,
                'most_frequent_city': 'تهران',
                'most_frequent_province': 'تهران',
                'geographical_diversity': 0.0
            }
        
        cities = user_transactions.get_column('city')
        provinces = user_transactions.get_column('province')
        
        # تنوع جغرافیایی
        unique_cities_count = cities.n_unique()
        geo_diversity = unique_cities_count / len(cities)
        
        # پیدا کردن پرتکرارترین شهر و استان
        city_counts = user_transactions.group_by('city').agg(pl.len().alias('count')).sort('count', descending=True)
        province_counts = user_transactions.group_by('province').agg(pl.len().alias('count')).sort('count', descending=True)
        
        most_frequent_city = city_counts.get_column('city')[0] if len(city_counts) > 0 else 'تهران'
        most_frequent_province = province_counts.get_column('province')[0] if len(province_counts) > 0 else 'تهران'
        
        return {
            'unique_cities_count': unique_cities_count,
            'unique_provinces_count': provinces.n_unique(),
            'most_frequent_city': most_frequent_city,
            'most_frequent_province': most_frequent_province,
            'geographical_diversity': float(geo_diversity)
        }
    
    def extract_amount_distribution_features(self, user_transactions: pl.DataFrame) -> Dict[str, float]:
        """استخراج ویژگی‌های توزیع مبلغ"""
        if user_transactions.is_empty():
            return {
                'amount_percentile_25': 0.0,
                'amount_percentile_75': 0.0,
                'amount_iqr': 0.0,
                'amount_skewness': 0.0
            }
        
        amounts = user_transactions.get_column('amount')
        
        # اگر فقط یک مقدار داشته باشیم
        if len(amounts) == 1:
            amount_value = float(amounts[0])
            return {
                'amount_percentile_25': amount_value,
                'amount_percentile_75': amount_value,
                'amount_iqr': 0.0,
                'amount_skewness': 0.0  # برای یک مقدار، skewness صفر است
            }
        
        q25 = amounts.quantile(0.25)
        q75 = amounts.quantile(0.75)
        iqr = q75 - q25
        
        # محاسبه skewness با handle کردن NaN
        try:
            amounts_np = amounts.to_numpy()
            skewness = stats.skew(amounts_np)
            if np.isnan(skewness) or np.isinf(skewness):
                skewness = 0.0
        except:
            skewness = 0.0
        
        return {
            'amount_percentile_25': float(q25),
            'amount_percentile_75': float(q75),
            'amount_iqr': float(iqr),
            'amount_skewness': float(skewness)
        }
    
    def extract_behavioral_features(self, user_transactions: pl.DataFrame) -> Dict[str, float]:
        """استخراج ویژگی‌های رفتاری"""
        if user_transactions.is_empty():
            return {
                'card_type_diversity': 0.0,
                'device_type_diversity': 0.0,
                'regularity_score': 0.0
            }
        
        # تنوع در استفاده از انواع کارت و دستگاه
        card_types = user_transactions.get_column('card_type').n_unique()
        device_types = user_transactions.get_column('device_type').n_unique()
        
        card_diversity = card_types / len(user_transactions)
        device_diversity = device_types / len(user_transactions)
        
        # امتیاز منظمی
        daily_counts = user_transactions.group_by('transaction_date').agg(pl.len().alias('count'))
        daily_std = daily_counts.get_column('count').std() or 0.0
        
        # Handle NaN values
        if daily_std == 0:
            regularity_score = 1.0  # کاملاً منظم
        else:
            regularity_score = 1 / (daily_std + 1)
        
        return {
            'card_type_diversity': float(card_diversity),
            'device_type_diversity': float(device_diversity),
            'regularity_score': float(regularity_score)
        }
    
    def extract_user_features(self, user_id: int) -> Dict[str, Any]:
        """استخراج تمام ویژگی‌های یک کاربر"""
        # دریافت اطلاعات کاربر
        user_info = self.db_manager.execute_query(
            "SELECT * FROM users WHERE user_id = ?", [user_id]
        )
        
        if user_info.is_empty():
            logger.warning(f"User {user_id} not found")
            return {}
        
        user_age = user_info.get_column('age')[0]
        
        # دریافت تراکنش‌های کاربر
        user_transactions = self.db_manager.get_user_transactions(user_id)
        
        # استخراج انواع ویژگی
        features = {'user_id': user_id}
        
        features.update(self.extract_transaction_volume_features(user_transactions))
        features.update(self.extract_temporal_features(user_transactions))
        features.update(self.extract_geographical_features(user_transactions))
        features.update(self.extract_behavioral_features(user_transactions))
        features.update(self.extract_amount_distribution_features(user_transactions))
        
        # اضافه کردن ویژگی‌های دموگرافیک
        features.update({
            'age': user_age,
            'birth_year': int(user_info.get_column('birth_year')[0])
        })
        
        return features
    
    def extract_features_batch(self, user_ids: List[int]) -> pl.DataFrame:
        """استخراج ویژگی برای مجموعه‌ای از کاربران"""
        logger.info(f"Extracting features for {len(user_ids)} users...")
        
        features_list = []
        
        for user_id in user_ids:
            user_features = self.extract_user_features(user_id)
            if user_features:
                features_list.append(user_features)
        
        if not features_list:
            logger.warning("No features extracted")
            return pl.DataFrame()
        
        features_df = pl.DataFrame(features_list)
        
        # ذخیره آمار
        self.extraction_stats['feature_names'] = list(features_df.columns)
        self.extraction_stats['features_extracted'] = len(features_df.columns)
        self.extraction_stats['users_processed'] = len(features_df)
        
        logger.info(f"Extracted {len(features_df.columns)} features for {len(features_df)} users")
        
        return features_df
    
    def extract_all_user_features(self, batch_size: int = 1000) -> pl.DataFrame:
        """استخراج ویژگی برای تمام کاربران"""
        logger.info("Starting feature extraction for all users...")
        
        # دریافت تمام user IDs
        all_users = self.db_manager.execute_query("SELECT user_id FROM users ORDER BY user_id")
        all_user_ids = all_users.get_column('user_id').to_list()
        
        logger.info(f"Found {len(all_user_ids)} users to process")
        
        all_features = []
        
        # پردازش batch به batch
        for i in range(0, len(all_user_ids), batch_size):
            batch_user_ids = all_user_ids[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}")
            
            batch_features = self.extract_features_batch(batch_user_ids)
            
            if not batch_features.is_empty():
                all_features.append(batch_features)
        
        # ترکیب تمام batch ها
        if all_features:
            final_features_df = pl.concat(all_features)
            logger.info(f"Feature extraction completed: {len(final_features_df)} users")
            return final_features_df
        else:
            logger.warning("No features extracted")
            return pl.DataFrame()
    
    def save_features_to_database(self, features_df: pl.DataFrame):
        """ذخیره ویژگی‌ها در دیتابیس"""
        if features_df.is_empty():
            logger.warning("No features to save")
            return
        
        logger.info(f"Saving {len(features_df)} user features to database...")
        
        # تبدیل geographical features به string
        if 'most_frequent_city' in features_df.columns:
            features_df = features_df.with_columns(
                pl.col('most_frequent_city').cast(pl.Utf8)
            )
        if 'most_frequent_province' in features_df.columns:
            features_df = features_df.with_columns(
                pl.col('most_frequent_province').cast(pl.Utf8)
            )
        
        # تبدیل به pandas برای ذخیره در SQLite (موقتی)
        pandas_df = features_df.to_pandas()
        
        # ذخیره در دیتابیس
        with self.db_manager.get_connection() as conn:
            conn.execute("DELETE FROM user_features")
            pandas_df.to_sql('user_features', conn, if_exists='append', index=False)
            conn.commit()
        
        logger.info("Features saved to database successfully") 