"""
سیستم تولید نویز کنترل‌شده برای شبیه‌سازی داده‌های واقعی (نسخه Polars)
"""

import numpy as np
import polars as pl
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime

from ..utils.config import get_config

logger = logging.getLogger(__name__)

class NoiseInjector:
    """کلاس تولید و تزریق انواع نویز به داده‌ها (تماماً با Polars)"""
    
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
                                transactions_df: pl.DataFrame,
                                users_df: pl.DataFrame) -> pl.DataFrame:
        """
        اعمال جامع تمام انواع نویز به DataFrame تراکنش‌ها
        
        Args:
            transactions_df: DataFrame تراکنش‌ها
            users_df: DataFrame کاربران (برای دسترسی به سن)
            
        Returns:
            DataFrame تراکنش‌ها با نویز اعمال‌شده
        """
        logger.info("Applying comprehensive noise to transactions...")

        # اطمینان از وجود ستون‌های لازم
        required_cols = {"user_id","amount","transaction_date","transaction_time","province","city","card_type","device_type","is_weekend","hour_of_day","day_of_month"}
        missing = required_cols - set(transactions_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in transactions: {missing}")

        # الحاق سن کاربر به تراکنش‌ها
        tx = transactions_df
        users_min = users_df.select(["user_id","age"]) if "age" in users_df.columns else users_df
        tx = tx.join(users_min, on="user_id", how="left")

        n = tx.height
        # ماسک‌ها و متغیرهای تصادفی
        rand_loc = np.random.random(n)
        rand_age = np.random.random(n)
        rand_time = np.random.random(n)
        rand_amount = np.random.random(n)
        rand_big_small = np.random.random(n)

        # 1) نویز موقعیت
        location_rate = self.data_config.location_noise_rate
        loc_mask = rand_loc < location_rate
        new_city_arr = tx["city"].to_numpy().copy()
        new_prov_arr = tx["province"].to_numpy().copy()
        if loc_mask.any():
            remote_choices = np.random.choice(self.iran_config.REMOTE_LOCATIONS, size=loc_mask.sum())
            # نگاشت شهر -> استان
            mapped_provinces = []
            for c in remote_choices:
                if c in ['چابهار', 'زاهدان', 'زابل']:
                    mapped_provinces.append('سیستان و بلوچستان')
                else:
                    mapped_provinces.append('هرمزگان')
            new_city_arr[loc_mask] = remote_choices
            new_prov_arr[loc_mask] = np.array(mapped_provinces, dtype=object)

        # 2) نویز سن-مبلغ
        age_rate = self.data_config.age_amount_noise_rate
        ages = tx["age"].fill_null(30).to_numpy() if "age" in tx.columns else np.full(n, 30)
        amt_arr = tx["amount"].to_numpy().copy()
        age_mask = (ages < 25) & (rand_age < age_rate)
        if age_mask.any():
            amt_arr[age_mask] = np.random.uniform(5_000_000, 20_000_000, size=age_mask.sum())

        # 3) نویز الگوی زمانی
        time_rate = self.data_config.time_pattern_noise_rate
        time_mask = rand_time < time_rate
        hour_arr = tx["hour_of_day"].to_numpy().astype(int).copy()
        if time_mask.any():
            hour_arr[time_mask] = np.random.randint(2, 6, size=time_mask.sum())
        # بروزرسانی رشته زمان
        time_str_arr = tx["transaction_time"].to_numpy().astype(object).copy()
        if time_mask.any():
            # فقط برای اندیس‌های time_mask، ساعت را جایگزین کنید
            idxs = np.where(time_mask)[0]
            for i in idxs:
                parts = str(time_str_arr[i]).split(":")
                if len(parts) == 3:
                    time_str_arr[i] = f"{int(hour_arr[i]):02d}:{parts[1]}:{parts[2]}"

        # 4) نویز مبلغ outlier (اگر نویز سن-مبلغ اعمال نشده)
        outlier_rate = self.data_config.amount_outlier_rate
        amount_mask = (~age_mask) & (rand_amount < outlier_rate)
        if amount_mask.any():
            big_mask = rand_big_small[amount_mask] < 0.7
            big_vals = np.random.uniform(30_000_000, 100_000_000, size=big_mask.sum())
            small_vals = np.random.uniform(100, 800, size=(~big_mask).sum())
            replacement = np.empty(amount_mask.sum())
            replacement[big_mask] = big_vals
            replacement[~big_mask] = small_vals
            amt_arr[amount_mask] = replacement

        # برچسب‌گذاری نویز و نوع آن
        is_noise_arr = loc_mask | age_mask | time_mask | amount_mask

        # تولید رشته noise_type با ترتیب منطقی
        def build_noise_type_strings() -> np.ndarray:
            parts = np.empty((n, 4), dtype=object)
            parts[:, 0] = np.where(loc_mask, 'location_anomaly', '')
            parts[:, 1] = np.where(age_mask, 'age_amount_mismatch', '')
            parts[:, 2] = np.where(time_mask, 'time_pattern_anomaly', '')
            parts[:, 3] = np.where(amount_mask, 'amount_outlier', '')
            # ترکیب با حذف رشته‌های خالی
            result = np.empty(n, dtype=object)
            for i in range(n):
                tokens = [p for p in parts[i] if p]
                result[i] = ",".join(tokens) if tokens else None
            return result

        noise_type_arr = build_noise_type_strings()

        # اعمال تغییرات به DataFrame
        tx = tx.with_columns([
            pl.Series("province", new_prov_arr),
            pl.Series("city", new_city_arr),
            pl.Series("amount", amt_arr),
            pl.Series("hour_of_day", hour_arr),
            pl.Series("transaction_time", time_str_arr),
            pl.Series("is_noise", is_noise_arr),
            pl.Series("noise_type", noise_type_arr),
        ])

        # بروزرسانی آمار نویز
        self.noise_stats['location_anomaly'] += int(loc_mask.sum())
        self.noise_stats['age_amount_mismatch'] += int(age_mask.sum())
        self.noise_stats['time_pattern_anomaly'] += int(time_mask.sum())
        self.noise_stats['amount_outlier'] += int(amount_mask.sum())
        self.noise_stats['total_noise_records'] += int(is_noise_arr.sum())

        # حذف ستون age الحاق‌شده
        if "age" in tx.columns:
            tx = tx.drop("age")

        logger.info(f"Noise injection completed. Total noise records: {self.noise_stats['total_noise_records']}")
        return tx
    
    def generate_controlled_user_noise(self, users_df: pl.DataFrame) -> pl.DataFrame:
        """
        تولید نویز کنترل‌شده در داده‌های کاربران
        
        Args:
            users_df: DataFrame کاربران
            
        Returns:
            DataFrame کاربران با نویز اعمال‌شده
        """
        df = users_df
        n = df.height

        # 1) نویز سن: 1%
        age_mask = np.random.random(n) < 0.01
        if age_mask.any():
            ages = df["age"].to_numpy().copy()
            ages[age_mask] = np.random.choice([16, 17, 85, 90, 95], size=int(age_mask.sum()))
            df = df.with_columns([pl.Series("age", ages)])

        # 2) نویز موقعیت: 2%
        loc_mask = np.random.random(n) < 0.02
        if loc_mask.any():
            prov = df["province"].to_numpy().astype(object).copy()
            city = df["city"].to_numpy().astype(object).copy()
            selected = np.random.choice(self.iran_config.REMOTE_LOCATIONS, size=int(loc_mask.sum()))
            mapped_prov = []
            for c in selected:
                if c in ['چابهار', 'زاهدان', 'زابل']:
                    mapped_prov.append('سیستان و بلوچستان')
                else:
                    mapped_prov.append('هرمزگان')
            prov[loc_mask] = np.array(mapped_prov, dtype=object)
            city[loc_mask] = selected
            df = df.with_columns([
                pl.Series("province", prov),
                pl.Series("city", city),
            ])

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