"""
سیستم تولید نویز کنترل‌شده برای شبیه‌سازی داده‌های واقعی
"""

import numpy as np
import polars as pl
import pandas as pd  # Keep for backward compatibility
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime

from ..utils.config import get_config

logger = logging.getLogger(__name__)

class NoiseInjector:
    """کلاس تولید و تزریق انواع نویز به داده‌ها"""
    
    def __init__(self):
        """مقداردهی اولیه"""
        self.config = get_config()
        self.data_config = self.config['data_generation']
        self.iran_config = self.config['iran_locations']
        
        # آمار نویزهای تولیدشده
        self.noise_stats = {
            'location_anomaly': 0,
            'age_amount_mismatch': 0,
            'time_pattern_anomaly': 0,
            'amount_outlier': 0,
            'total_noise_records': 0
        }
        
        logger.info("NoiseInjector initialized with controlled noise rates")
    
    def inject_location_noise(self, 
                            province: str, 
                            city: str, 
                            noise_probability: float = None) -> Tuple[str, str, bool]:
        """
        تزریق نویز موقعیت جغرافیایی
        
        توضیح: 2% از تراکنش‌ها در مکان‌های غیرعادی (دورافتاده) انجام می‌شوند
        دلیل: شبیه‌سازی سفرها، تراکنش‌های غیرعادی، یا خطای سیستم
        
        Args:
            province: استان اصلی
            city: شهر اصلی
            noise_probability: احتمال نویز (None برای استفاده از config)
            
        Returns:
            (province, city, is_noise)
        """
        noise_rate = noise_probability or self.data_config.location_noise_rate
        
        if np.random.random() < noise_rate:
            # انتخاب شهر دورافتاده
            remote_city = np.random.choice(self.iran_config.REMOTE_LOCATIONS)
            
            # تعیین استان براساس شهر
            if remote_city in ['چابهار', 'زاهدان', 'زابل']:
                remote_province = 'سیستان و بلوچستان'
            elif remote_city in ['بندر عباس', 'میناب', 'کنگان', 'بندر لنگه', 'جاسک']:
                remote_province = 'هرمزگان'
            else:
                remote_province = 'سیستان و بلوچستان'
            
            self.noise_stats['location_anomaly'] += 1
            logger.debug(f"Location noise: {province}/{city} -> {remote_province}/{remote_city}")
            
            return remote_province, remote_city, True
        
        return province, city, False
    
    def inject_age_amount_mismatch(self, 
                                 user_age: int, 
                                 amount: float,
                                 noise_probability: float = None) -> Tuple[float, bool]:
        """
        تزریق نویز عدم تطابق سن-مبلغ
        
        توضیح: 0.5% کاربران جوان (زیر 25 سال) تراکنش‌های بزرگ (>5 میلیون) دارند
        دلیل: ممکن است حساب خانوادگی، ارث، یا فعالیت مشکوک باشد
        
        Args:
            user_age: سن کاربر
            amount: مبلغ اصلی
            noise_probability: احتمال نویز
            
        Returns:
            (modified_amount, is_noise)
        """
        noise_rate = noise_probability or self.data_config.age_amount_noise_rate
        
        # فقط برای کاربران جوان
        if user_age < 25 and np.random.random() < noise_rate:
            # تولید مبلغ بزرگ (5-20 میلیون)
            large_amount = np.random.uniform(5_000_000, 20_000_000)
            
            self.noise_stats['age_amount_mismatch'] += 1
            logger.debug(f"Age-amount mismatch: age={user_age}, amount={amount:.0f} -> {large_amount:.0f}")
            
            return large_amount, True
        
        return amount, False
    
    def inject_time_pattern_noise(self, 
                                hour: int,
                                noise_probability: float = None) -> Tuple[int, bool]:
        """
        تزریق نویز الگوی زمانی
        
        توضیح: 1% تراکنش‌ها در ساعات غیرعادی (2-5 صبح) انجام می‌شوند
        دلیل: تراکنش‌های اورژانسی، کار شبانه، یا فعالیت مشکوک
        
        Args:
            hour: ساعت اصلی
            noise_probability: احتمال نویز
            
        Returns:
            (modified_hour, is_noise)
        """
        noise_rate = noise_probability or self.data_config.time_pattern_noise_rate
        
        if np.random.random() < noise_rate:
            # انتخاب ساعت غیرعادی (2-5 صبح)
            unusual_hour = np.random.randint(2, 6)
            
            self.noise_stats['time_pattern_anomaly'] += 1
            logger.debug(f"Time pattern noise: {hour} -> {unusual_hour}")
            
            return unusual_hour, True
        
        return hour, False
    
    def inject_amount_outlier(self, 
                            amount: float,
                            noise_probability: float = None) -> Tuple[float, bool]:
        """
        تزریق نویز مبلغ outlier
        
        توضیح: 0.3% تراکنش‌ها مبالغ خیلی بزرگ (>30 میلیون) یا خیلی کوچک (<500) دارند
        دلیل: خطای انسانی، تراکنش‌های خاص، یا مشکل سیستم
        
        Args:
            amount: مبلغ اصلی
            noise_probability: احتمال نویز
            
        Returns:
            (modified_amount, is_noise)
        """
        noise_rate = noise_probability or self.data_config.amount_outlier_rate
        
        if np.random.random() < noise_rate:
            # 70% احتمال مبلغ خیلی بزرگ، 30% احتمال مبلغ خیلی کوچک
            if np.random.random() < 0.7:
                # مبلغ خیلی بزرگ (30-100 میلیون)
                outlier_amount = np.random.uniform(30_000_000, 100_000_000)
            else:
                # مبلغ خیلی کوچک (100-800 تومان)
                outlier_amount = np.random.uniform(100, 800)
            
            self.noise_stats['amount_outlier'] += 1
            logger.debug(f"Amount outlier: {amount:.0f} -> {outlier_amount:.0f}")
            
            return outlier_amount, True
        
        return amount, False
    
    def apply_comprehensive_noise(self, 
                                transactions_df: pd.DataFrame,
                                users_df: pd.DataFrame) -> pd.DataFrame:
        """
        اعمال جامع تمام انواع نویز به DataFrame تراکنش‌ها
        
        Args:
            transactions_df: DataFrame تراکنش‌ها
            users_df: DataFrame کاربران (برای دسترسی به سن)
            
        Returns:
            DataFrame تراکنش‌ها با نویز اعمال‌شده
        """
        logger.info("Applying comprehensive noise to transactions...")
        
        # کپی DataFrame برای عدم تغییر اصلی
        df = transactions_df.copy()
        
        # اضافه کردن ستون‌های نویز
        df['is_noise'] = False
        df['noise_type'] = None
        
        # ایجاد دیکشنری نگاشت user_id به age
        user_ages = users_df.set_index('user_id')['age'].to_dict()
        
        # اعمال انواع نویز
        for idx, row in df.iterrows():
            user_age = user_ages.get(row['user_id'], 30)  # سن پیش‌فرض 30
            noise_applied = []
            
            # 1. نویز موقعیت
            new_province, new_city, location_noise = self.inject_location_noise(
                row['province'], row['city']
            )
            if location_noise:
                df.at[idx, 'province'] = new_province
                df.at[idx, 'city'] = new_city
                noise_applied.append('location_anomaly')
            
            # 2. نویز سن-مبلغ
            new_amount, age_amount_noise = self.inject_age_amount_mismatch(
                user_age, row['amount']
            )
            if age_amount_noise:
                df.at[idx, 'amount'] = new_amount
                noise_applied.append('age_amount_mismatch')
            
            # 3. نویز الگوی زمانی
            new_hour, time_noise = self.inject_time_pattern_noise(
                row['hour_of_day']
            )
            if time_noise:
                df.at[idx, 'hour_of_day'] = new_hour
                # بروزرسانی زمان کامل
                time_parts = row['transaction_time'].split(':')
                df.at[idx, 'transaction_time'] = f"{new_hour:02d}:{time_parts[1]}:{time_parts[2]}"
                noise_applied.append('time_pattern_anomaly')
            
            # 4. نویز مبلغ outlier (فقط اگر نویز سن-مبلغ اعمال نشده)
            if not age_amount_noise:
                final_amount, amount_noise = self.inject_amount_outlier(
                    df.at[idx, 'amount']
                )
                if amount_noise:
                    df.at[idx, 'amount'] = final_amount
                    noise_applied.append('amount_outlier')
            
            # ثبت نویز در DataFrame
            if noise_applied:
                df.at[idx, 'is_noise'] = True
                df.at[idx, 'noise_type'] = ','.join(noise_applied)
                self.noise_stats['total_noise_records'] += 1
        
        logger.info(f"Noise injection completed. Total noise records: {self.noise_stats['total_noise_records']}")
        return df
    
    def generate_controlled_user_noise(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """
        تولید نویز کنترل‌شده در داده‌های کاربران
        
        Args:
            users_df: DataFrame کاربران
            
        Returns:
            DataFrame کاربران با نویز اعمال‌شده
        """
        df = users_df.copy()
        
        # نویز سن: 1% کاربران سن غیرعادی دارند
        age_noise_mask = np.random.random(len(df)) < 0.01
        unusual_ages = np.random.choice([16, 17, 85, 90, 95], size=np.sum(age_noise_mask))
        df.loc[age_noise_mask, 'age'] = unusual_ages
        
        # نویز موقعیت: 2% کاربران در شهرهای دورافتاده ثبت‌نام کرده‌اند
        location_noise_mask = np.random.random(len(df)) < 0.02
        for idx in df[location_noise_mask].index:
            remote_city = np.random.choice(self.iran_config.REMOTE_LOCATIONS)
            if remote_city in ['چابهار', 'زاهدان', 'زابل']:
                df.at[idx, 'province'] = 'سیستان و بلوچستان'
            else:
                df.at[idx, 'province'] = 'هرمزگان'
            df.at[idx, 'city'] = remote_city
        
        return df
    
    def get_noise_statistics(self) -> Dict[str, Any]:
        """
        دریافت آمار کامل نویزهای تولیدشده
        
        Returns:
            دیکشنری حاوی آمار نویز
        """
        total_noise = self.noise_stats['total_noise_records']
        
        stats = {
            'noise_counts': self.noise_stats.copy(),
            'noise_percentages': {},
            'noise_descriptions': {
                'location_anomaly': {
                    'description': 'تراکنش در مکان‌های دورافتاده یا غیرعادی',
                    'reason': 'سفر، کار، یا خطای سیستم',
                    'detection_difficulty': 'متوسط'
                },
                'age_amount_mismatch': {
                    'description': 'کاربران جوان با تراکنش‌های بزرگ',
                    'reason': 'حساب خانوادگی، ارث، یا فعالیت مشکوک',
                    'detection_difficulty': 'آسان'
                },
                'time_pattern_anomaly': {
                    'description': 'تراکنش در ساعات غیرعادی (شب)',
                    'reason': 'اورژانس، کار شبانه، یا فعالیت مشکوک',
                    'detection_difficulty': 'آسان'
                },
                'amount_outlier': {
                    'description': 'مبالغ خیلی بزرگ یا خیلی کوچک',
                    'reason': 'خطای انسانی، تراکنش خاص، یا مشکل سیستم',
                    'detection_difficulty': 'آسان'
                }
            }
        }
        
        # محاسبه درصدها
        if total_noise > 0:
            for noise_type, count in self.noise_stats.items():
                if noise_type != 'total_noise_records':
                    stats['noise_percentages'][noise_type] = (count / total_noise) * 100
        
        return stats
    
    def get_noise_summary_report(self) -> str:
        """
        تولید گزارش خلاصه نویزها
        
        Returns:
            گزارش متنی نویزها
        """
        stats = self.get_noise_statistics()
        
        report = "=== گزارش نویزهای تولیدشده ===\n\n"
        
        report += f"کل رکوردهای نویز: {stats['noise_counts']['total_noise_records']:,}\n\n"
        
        report += "تفکیک انواع نویز:\n"
        for noise_type, description in stats['noise_descriptions'].items():
            count = stats['noise_counts'][noise_type]
            percentage = stats['noise_percentages'].get(noise_type, 0)
            
            report += f"\n{noise_type}:\n"
            report += f"  تعداد: {count:,}\n"
            report += f"  درصد: {percentage:.2f}%\n"
            report += f"  توضیح: {description['description']}\n"
            report += f"  دلیل: {description['reason']}\n"
            report += f"  سختی تشخیص: {description['detection_difficulty']}\n"
        
        report += "\n=== منطق تولید نویز ===\n"
        report += "1. نویزها به صورت کنترل‌شده و با نرخ‌های مشخص تولید شده‌اند\n"
        report += "2. هر نوع نویز الگوی مشخصی از ناهنجاری را شبیه‌سازی می‌کند\n"
        report += "3. نویزها برای تست الگوریتم‌های anomaly detection طراحی شده‌اند\n"
        report += "4. توزیع نویزها بر اساس تجربه واقعی سیستم‌های بانکی تنظیم شده\n"
        
        return report
    
    def reset_statistics(self):
        """ریست آمار نویزها"""
        self.noise_stats = {
            'location_anomaly': 0,
            'age_amount_mismatch': 0,
            'time_pattern_anomaly': 0,
            'amount_outlier': 0,
            'total_noise_records': 0
        }
        logger.info("Noise statistics reset")
    
    def export_noise_analysis(self, output_path: str):
        """
        صادرات تحلیل نویز به فایل
        
        Args:
            output_path: مسیر فایل خروجی
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.get_noise_summary_report())
        
        logger.info(f"Noise analysis exported to {output_path}") 