"""
تنظیمات و پیکربندی پروژه
"""
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class DatabaseConfig:
    """تنظیمات دیتابیس"""
    db_path: str = "database/banking_data.db"
    chunk_size: int = 10000  # تعداد رکورد در هر chunk
    batch_size: int = 1000   # تعداد رکورد برای batch processing
    enable_wal: bool = True  # فعال‌سازی WAL mode برای عملکرد بهتر

@dataclass
class DataGenerationConfig:
    """تنظیمات تولید داده"""
    total_users: int = 100_000
    min_transactions_per_user: int = 20
    max_transactions_per_user: int = 2000
    
    # بازه زمانی (یک ماه)
    start_date: datetime = datetime(2024, 1, 1)
    end_date: datetime = datetime(2024, 1, 31)
    
    # نرخ نویز
    location_noise_rate: float = 0.02      # 2% موقعیت غیرعادی
    age_amount_noise_rate: float = 0.005   # 0.5% سن-مبلغ غیرتطبیق
    time_pattern_noise_rate: float = 0.01  # 1% الگوی زمانی غیرعادی
    amount_outlier_rate: float = 0.003     # 0.3% مبلغ outlier

@dataclass
class DistributionConfig:
    """تنظیمات توزیع‌های آماری"""
    
    # توزیع سن (Beta scaled to 18-80)
    age_beta_alpha: float = 2.0
    age_beta_beta: float = 5.0
    age_min: int = 18
    age_max: int = 80
    
    # توزیع مبلغ تراکنش (Log-normal)
    amount_lognorm_mean: float = 4.5  # ln(مبلغ متوسط)
    amount_lognorm_sigma: float = 1.2
    amount_min: float = 1000.0        # حداقل 1000 تومان
    amount_max: float = 50_000_000.0  # حداکثر 50 میلیون تومان
    
    # توزیع تعداد تراکنش روزانه (Poisson)
    daily_transactions_lambda: float = 3.0
    
    # توزیع ساعت تراکنش (Beta برای شبیه‌سازی ساعات فعال)
    hour_beta_alpha: float = 3.0
    hour_beta_beta: float = 2.0

class IranLocationConfig:
    """تنظیمات موقعیت‌های جغرافیایی ایران"""
    
    # استان‌های ایران با جمعیت تقریبی (برای weighted sampling)
    PROVINCES = {
        'تهران': {'weight': 0.15, 'cities': ['تهران', 'ری', 'شهریار', 'ورامین']},
        'اصفهان': {'weight': 0.08, 'cities': ['اصفهان', 'کاشان', 'نجف‌آباد', 'خمینی‌شهر']},
        'فارس': {'weight': 0.07, 'cities': ['شیراز', 'مرودشت', 'کازرون', 'لامرد']},
        'خراسان رضوی': {'weight': 0.06, 'cities': ['مشهد', 'نیشابور', 'سبزوار', 'تربت حیدریه']},
        'خوزستان': {'weight': 0.06, 'cities': ['اهواز', 'آبادان', 'خرمشهر', 'دزفول']},
        'مازندران': {'weight': 0.05, 'cities': ['ساری', 'بابل', 'آمل', 'قائمشهر']},
        'گیلان': {'weight': 0.04, 'cities': ['رشت', 'بندر انزلی', 'لاهیجان', 'رودسر']},
        'آذربایجان شرقی': {'weight': 0.05, 'cities': ['تبریز', 'مراغه', 'اهر', 'بناب']},
        'کرمان': {'weight': 0.04, 'cities': ['کرمان', 'رفسنجان', 'بم', 'سیرجان']},
        'البرز': {'weight': 0.04, 'cities': ['کرج', 'نظرآباد', 'طالقان', 'اشتهارد']}
    }
    
    # شهرهای دورافتاده برای نویز
    REMOTE_LOCATIONS = [
        'چابهار', 'زاهدان', 'زابل', 'بندر عباس', 
        'میناب', 'کنگان', 'بندر لنگه', 'جاسک'
    ]

class CardTypeConfig:
    """تنظیمات انواع کارت"""
    
    CARD_TYPES = {
        'پیش‌پرداخت': {'weight': 0.15, 'avg_amount_multiplier': 0.6},
        'جاری': {'weight': 0.35, 'avg_amount_multiplier': 1.0},
        'قرض‌الحسنه': {'weight': 0.25, 'avg_amount_multiplier': 0.8},
        'اعتباری': {'weight': 0.20, 'avg_amount_multiplier': 1.3},
        'طلایی': {'weight': 0.05, 'avg_amount_multiplier': 2.0}
    }

class DeviceTypeConfig:
    """تنظیمات انواع دستگاه"""
    
    DEVICE_TYPES = {
        'موبایل_اندروید': {'weight': 0.45},
        'موبایل_iOS': {'weight': 0.25},
        'وب_دسکتاپ': {'weight': 0.15},
        'ATM': {'weight': 0.10},
        'POS': {'weight': 0.05}
    }

# تنظیمات کلی
CONFIG = {
    'database': DatabaseConfig(),
    'data_generation': DataGenerationConfig(),
    'distributions': DistributionConfig(),
    'iran_locations': IranLocationConfig(),
    'card_types': CardTypeConfig(),
    'device_types': DeviceTypeConfig()
}

def get_config():
    """دریافت تنظیمات کامل پروژه"""
    return CONFIG

def setup_directories():
    """ایجاد پوشه‌های مورد نیاز"""
    directories = [
        'database',
        'output/reports',
        'output/plots', 
        'output/summaries'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True) 