"""
تبدیل و نرمال‌سازی ویژگی‌ها برای مدل‌سازی
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import get_config

logger = logging.getLogger(__name__)

class FeatureTransformer:
    """کلاس تبدیل و پردازش ویژگی‌های استخراج‌شده"""
    
    def __init__(self):
        """مقداردهی اولیه"""
        self.config = get_config()
        
        # ابزارهای تبدیل
        self.scalers = {}
        self.label_encoders = {}
        self.feature_selector = None
        self.pca = None
        
        # آمار تبدیل
        self.transformation_stats = {
            'original_features': 0,
            'transformed_features': 0,
            'categorical_features': 0,
            'numerical_features': 0
        }
        
        logger.info("FeatureTransformer initialized")
    
    def identify_feature_types(self, features_df: pd.DataFrame) -> Dict[str, List[str]]:
        """شناسایی انواع ویژگی‌ها"""
        feature_types = {
            'numerical': [],
            'categorical': [],
            'identifier': []
        }
        
        for column in features_df.columns:
            if column in ['user_id']:
                feature_types['identifier'].append(column)
            elif features_df[column].dtype in ['object', 'category']:
                feature_types['categorical'].append(column)
            else:
                feature_types['numerical'].append(column)
        
        logger.info(f"Identified feature types: "
                   f"numerical={len(feature_types['numerical'])}, "
                   f"categorical={len(feature_types['categorical'])}, "
                   f"identifier={len(feature_types['identifier'])}")
        
        return feature_types
    
    def handle_missing_values(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """پردازش مقادیر گمشده"""
        df = features_df.copy()
        
        feature_types = self.identify_feature_types(df)
        
        # پر کردن مقادیر گمشده عددی با میانه
        for col in feature_types['numerical']:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                logger.debug(f"Filled {col} missing values with median: {median_value}")
        
        # پر کردن مقادیر گمشده categorical با mode
        for col in feature_types['categorical']:
            if df[col].isnull().any():
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
                df[col].fillna(mode_value, inplace=True)
                logger.debug(f"Filled {col} missing values with mode: {mode_value}")
        
        return df
    
    def encode_categorical_features(self, features_df: pd.DataFrame, 
                                  fit: bool = True) -> pd.DataFrame:
        """کدگذاری ویژگی‌های categorical"""
        df = features_df.copy()
        feature_types = self.identify_feature_types(df)
        
        for col in feature_types['categorical']:
            if fit:
                # ایجاد و fit کردن encoder جدید
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.label_encoders[col] = encoder
                logger.debug(f"Fitted and encoded {col}")
            else:
                # استفاده از encoder موجود
                if col in self.label_encoders:
                    try:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
                        logger.debug(f"Encoded {col} using existing encoder")
                    except ValueError as e:
                        logger.warning(f"Could not encode {col}: {e}")
                        # در صورت وجود label جدید، از مقدار پیش‌فرض استفاده کن
                        df[col] = 0
        
        self.transformation_stats['categorical_features'] = len(feature_types['categorical'])
        
        return df
    
    def scale_numerical_features(self, features_df: pd.DataFrame, 
                                method: str = 'standard',
                                fit: bool = True) -> pd.DataFrame:
        """نرمال‌سازی ویژگی‌های عددی"""
        df = features_df.copy()
        feature_types = self.identify_feature_types(df)
        
        # حذف identifier features از scaling
        numerical_features = [col for col in feature_types['numerical'] 
                            if col not in feature_types['identifier']]
        
        if not numerical_features:
            logger.warning("No numerical features to scale")
            return df
        
        # انتخاب روش scaling
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {method}. Using standard.")
            scaler = StandardScaler()
        
        if fit:
            # Fit و transform
            df[numerical_features] = scaler.fit_transform(df[numerical_features])
            self.scalers[method] = scaler
            logger.info(f"Fitted and scaled {len(numerical_features)} features using {method}")
        else:
            # فقط transform
            if method in self.scalers:
                df[numerical_features] = self.scalers[method].transform(df[numerical_features])
                logger.info(f"Scaled {len(numerical_features)} features using existing {method} scaler")
            else:
                logger.warning(f"No fitted scaler found for method: {method}")
        
        self.transformation_stats['numerical_features'] = len(numerical_features)
        
        return df
    
    def remove_outliers(self, features_df: pd.DataFrame, 
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """حذف outlier ها"""
        df = features_df.copy()
        feature_types = self.identify_feature_types(df)
        
        numerical_features = [col for col in feature_types['numerical'] 
                            if col not in ['user_id']]
        
        outlier_indices = set()
        
        for col in numerical_features:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                outlier_indices.update(col_outliers)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = df[z_scores > threshold].index
                outlier_indices.update(col_outliers)
        
        # حذف outlier ها
        clean_df = df.drop(index=list(outlier_indices))
        
        logger.info(f"Removed {len(outlier_indices)} outliers using {method} method")
        
        return clean_df
    
    def select_features(self, features_df: pd.DataFrame, 
                       target_col: Optional[str] = None,
                       k: int = 20,
                       method: str = 'f_classif') -> pd.DataFrame:
        """انتخاب بهترین ویژگی‌ها"""
        df = features_df.copy()
        
        if target_col is None or target_col not in df.columns:
            logger.warning("No valid target column for feature selection")
            return df
        
        # جدا کردن features و target
        feature_cols = [col for col in df.columns if col not in ['user_id', target_col]]
        X = df[feature_cols]
        y = df[target_col]
        
        # انتخاب روش
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(feature_cols)))
        else:
            logger.warning(f"Unknown feature selection method: {method}")
            return df
        
        # انتخاب ویژگی‌ها
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # ایجاد DataFrame جدید با ویژگی‌های انتخاب‌شده
        result_df = df[['user_id'] + selected_features + [target_col]]
        
        self.feature_selector = selector
        
        logger.info(f"Selected {len(selected_features)} features out of {len(feature_cols)}")
        
        return result_df
    
    def apply_pca(self, features_df: pd.DataFrame, 
                  n_components: Optional[int] = None,
                  variance_threshold: float = 0.95) -> pd.DataFrame:
        """اعمال PCA برای کاهش ابعاد"""
        df = features_df.copy()
        feature_types = self.identify_feature_types(df)
        
        # فقط ویژگی‌های عددی
        numerical_features = [col for col in feature_types['numerical'] 
                            if col not in ['user_id']]
        
        if len(numerical_features) < 2:
            logger.warning("Not enough numerical features for PCA")
            return df
        
        X = df[numerical_features]
        
        # تعیین تعداد components
        if n_components is None:
            # محاسبه تعداد component برای رسیدن به variance threshold
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        
        # اعمال PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # ایجاد DataFrame جدید
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)
        
        # ترکیب با identifier columns
        result_df = pd.concat([df[['user_id']], pca_df], axis=1)
        
        self.pca = pca
        
        logger.info(f"Applied PCA: {len(numerical_features)} -> {n_components} components")
        logger.info(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return result_df
    
    def create_interaction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """ایجاد ویژگی‌های تعاملی"""
        df = features_df.copy()
        
        # ویژگی‌های کلیدی برای تعامل
        key_features = [
            'total_amount', 'avg_transaction_amount', 'total_transactions',
            'transaction_frequency', 'age', 'geographical_diversity'
        ]
        
        available_features = [f for f in key_features if f in df.columns]
        
        if len(available_features) < 2:
            logger.warning("Not enough features for interaction")
            return df
        
        # ایجاد interaction features مهم
        interactions = [
            ('total_amount', 'age', 'amount_per_age'),
            ('avg_transaction_amount', 'transaction_frequency', 'amount_frequency_ratio'),
            ('total_transactions', 'geographical_diversity', 'transaction_geo_ratio')
        ]
        
        for feat1, feat2, new_name in interactions:
            if feat1 in df.columns and feat2 in df.columns:
                df[new_name] = df[feat1] * df[feat2]
                logger.debug(f"Created interaction feature: {new_name}")
        
        # ratio features
        if 'total_amount' in df.columns and 'total_transactions' in df.columns:
            df['amount_per_transaction'] = df['total_amount'] / (df['total_transactions'] + 1)
        
        if 'weekend_transaction_ratio' in df.columns:
            df['weekday_transaction_ratio'] = 1 - df['weekend_transaction_ratio']
        
        return df
    
    def transform_pipeline(self, features_df: pd.DataFrame, 
                          fit: bool = True,
                          steps: List[str] = None) -> pd.DataFrame:
        """پایپلاین کامل تبدیل ویژگی‌ها"""
        if steps is None:
            steps = ['missing_values', 'categorical_encoding', 'interaction_features', 'scaling']
        
        df = features_df.copy()
        
        logger.info(f"Starting transformation pipeline with steps: {steps}")
        
        # 1. پردازش مقادیر گمشده
        if 'missing_values' in steps:
            df = self.handle_missing_values(df)
            logger.info("✓ Handled missing values")
        
        # 2. کدگذاری categorical
        if 'categorical_encoding' in steps:
            df = self.encode_categorical_features(df, fit=fit)
            logger.info("✓ Encoded categorical features")
        
        # 3. ایجاد ویژگی‌های تعاملی
        if 'interaction_features' in steps:
            df = self.create_interaction_features(df)
            logger.info("✓ Created interaction features")
        
        # 4. scaling
        if 'scaling' in steps:
            df = self.scale_numerical_features(df, method='standard', fit=fit)
            logger.info("✓ Scaled numerical features")
        
        # 5. حذف outliers (فقط در fit mode)
        if 'outlier_removal' in steps and fit:
            df = self.remove_outliers(df, method='iqr', threshold=2.0)
            logger.info("✓ Removed outliers")
        
        # بروزرسانی آمار
        self.transformation_stats['original_features'] = len(features_df.columns)
        self.transformation_stats['transformed_features'] = len(df.columns)
        
        logger.info(f"Transformation completed: {len(features_df.columns)} -> {len(df.columns)} features")
        
        return df
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """خلاصه تبدیلات انجام‌شده"""
        return {
            'transformation_stats': self.transformation_stats,
            'fitted_scalers': list(self.scalers.keys()),
            'encoded_features': list(self.label_encoders.keys()),
            'has_feature_selector': self.feature_selector is not None,
            'has_pca': self.pca is not None
        } 