"""
کلاس‌های توزیع آماری برای تولید داده‌های واقعی‌تر
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Any
import logging
from datetime import datetime, timedelta

from ..utils.config import get_config

logger = logging.getLogger(__name__)

class StatisticalDistributions:
    """کلاس اصلی توزیع‌های آماری"""
    
    def __init__(self):
        """مقداردهی اولیه با تنظیمات پروژه"""
        self.config = get_config()
        self.dist_config = self.config['distributions']
        self.iran_config = self.config['iran_locations']
        self.card_config = self.config['card_types']
        self.device_config = self.config['device_types']
        
        # تنظیم seed برای reproducibility
        np.random.seed(42)
        
        # آماده‌سازی داده‌های جغرافیایی
        self._prepare_location_weights()
        self._prepare_card_weights()
        self._prepare_device_weights()
    
    def _prepare_location_weights(self):
        """آماده‌سازی وزن‌های استان‌ها و شهرها"""
        self.provinces = list(self.iran_config.PROVINCES.keys())
        self.province_weights = [self.iran_config.PROVINCES[p]['weight'] 
                               for p in self.provinces]
        
        # نرمال‌سازی وزن‌های استان‌ها
        total_province_weight = sum(self.province_weights)
        self.province_weights = [w / total_province_weight for w in self.province_weights]
        
        # ایجاد لیست تمام شهرها با وزن‌هایشان
        self.all_cities = []
        self.city_weights = []
        self.city_to_province = {}
        
        for province, data in self.iran_config.PROVINCES.items():
            cities = data['cities']
            province_weight = data['weight']
            city_weight = province_weight / len(cities)  # تقسیم مساوی وزن استان
            
            for city in cities:
                self.all_cities.append(city)
                self.city_weights.append(city_weight)
                self.city_to_province[city] = province
        
        # نرمال‌سازی وزن‌های شهرها
        total_city_weight = sum(self.city_weights)
        self.city_weights = [w / total_city_weight for w in self.city_weights]
    
    def _prepare_card_weights(self):
        """آماده‌سازی وزن‌های انواع کارت"""
        self.card_types = list(self.card_config.CARD_TYPES.keys())
        self.card_weights = [self.card_config.CARD_TYPES[c]['weight'] 
                           for c in self.card_types]
        
        # نرمال‌سازی وزن‌ها
        total_weight = sum(self.card_weights)
        self.card_weights = [w / total_weight for w in self.card_weights]
    
    def _prepare_device_weights(self):
        """آماده‌سازی وزن‌های انواع دستگاه"""
        self.device_types = list(self.device_config.DEVICE_TYPES.keys())
        self.device_weights = [self.device_config.DEVICE_TYPES[d]['weight'] 
                             for d in self.device_types]
        
        # نرمال‌سازی وزن‌ها
        total_weight = sum(self.device_weights)
        self.device_weights = [w / total_weight for w in self.device_weights]
    
    def generate_ages(self, n: int) -> np.ndarray:
        """
        تولید سن‌ها با توزیع Beta
        
        توضیح: توزیع Beta برای شبیه‌سازی سن‌ها مناسب است چون:
        - می‌توان بازه آن را محدود کرد (18-80)
        - توزیع آن قابل تنظیم است (بیشتر جوان یا مسن)
        - شکل واقعی‌تری نسبت به uniform دارد
        
        Args:
            n: تعداد نمونه
            
        Returns:
            آرایه سن‌ها
        """
        # توزیع Beta با پارامترهای α=2, β=5 (بیشتر جوان)
        beta_samples = stats.beta.rvs(
            self.dist_config.age_beta_alpha,
            self.dist_config.age_beta_beta,
            size=n
        )
        
        # تبدیل بازه [0,1] به [18,80]
        ages = (beta_samples * (self.dist_config.age_max - self.dist_config.age_min) + 
                self.dist_config.age_min)
        
        return np.round(ages).astype(int)
    
    def generate_transaction_amounts(self, n: int, 
                                   card_type: str = None, 
                                   user_age: int = None) -> np.ndarray:
        """
        تولید مبلغ تراکنش‌ها با توزیع Log-Normal
        
        توضیح: توزیع Log-Normal برای مبالغ مناسب است چون:
        - اکثر تراکنش‌ها مبلغ کمی دارند
        - تعداد کمی تراکنش مبلغ بزرگ دارند
        - شکل واقعی‌تری از رفتار مالی است
        
        Args:
            n: تعداد نمونه
            card_type: نوع کارت (برای تعدیل مبلغ)
            user_age: سن کاربر (برای تعدیل مبلغ)
            
        Returns:
            آرایه مبالغ
        """
        # پارامترهای پایه توزیع Log-Normal
        mu = self.dist_config.amount_lognorm_mean
        sigma = self.dist_config.amount_lognorm_sigma
        
        # تعدیل براساس نوع کارت
        if card_type and card_type in self.card_config.CARD_TYPES:
            multiplier = self.card_config.CARD_TYPES[card_type]['avg_amount_multiplier']
            mu = mu + np.log(multiplier)
        
        # تعدیل براساس سن (کاربران مسن‌تر معمولاً تراکنش بیشتری دارند)
        if user_age:
            age_factor = 1 + (user_age - 30) * 0.02  # 2% افزایش در هر سال
            age_factor = max(0.5, min(2.0, age_factor))  # محدود به بازه [0.5, 2.0]
            mu = mu + np.log(age_factor)
        
        # تولید نمونه‌ها
        amounts = stats.lognorm.rvs(sigma, scale=np.exp(mu), size=n)
        
        # اعمال حدود کمینه و بیشینه
        amounts = np.clip(amounts, 
                         self.dist_config.amount_min, 
                         self.dist_config.amount_max)
        
        return np.round(amounts).astype(int)
    
    def generate_daily_transaction_counts(self, n: int) -> np.ndarray:
        """
        تولید تعداد تراکنش روزانه با توزیع Poisson
        
        توضیح: توزیع Poisson برای شمارش رویدادها مناسب است چون:
        - مدل می‌کند که رویدادها به صورت تصادفی اتفاق می‌افتند
        - پارامتر λ متوسط و واریانس را کنترل می‌کند
        - برای تعداد تراکنش روزانه واقعی است
        
        Args:
            n: تعداد روز
            
        Returns:
            آرایه تعداد تراکنش‌ها در هر روز
        """
        return stats.poisson.rvs(
            self.dist_config.daily_transactions_lambda, 
            size=n
        )
    
    def generate_transaction_hours(self, n: int) -> np.ndarray:
        """
        تولید ساعت تراکنش با توزیع Beta
        
        توضیح: توزیع Beta برای ساعات مناسب است چون:
        - می‌توان شکل توزیع را کنترل کرد
        - ساعات فعال (8-20) احتمال بیشتری دارند
        - شبیه‌سازی الگوی واقعی فعالیت
        
        Args:
            n: تعداد نمونه
            
        Returns:
            آرایه ساعت‌ها (0-23)
        """
        # توزیع Beta برای ساعات فعال (بیشتر 8-20)
        beta_samples = stats.beta.rvs(
            self.dist_config.hour_beta_alpha,
            self.dist_config.hour_beta_beta,
            size=n
        )
        
        # تبدیل به ساعات 6-22 (اصلی‌ترین ساعات)
        base_hours = beta_samples * 16 + 6  # بازه [6, 22]
        
        # اضافه کردن کمی تراکنش در ساعات دیگر
        random_factor = np.random.uniform(0, 1, n)
        off_hours_mask = random_factor < 0.1  # 10% در ساعات غیرعادی
        
        off_hours = np.random.uniform(0, 6, np.sum(off_hours_mask))  # 0-6 یا 22-24
        off_hours = np.where(np.random.random(len(off_hours)) < 0.5, 
                           off_hours, off_hours + 22)
        
        base_hours[off_hours_mask] = off_hours
        
        return np.clip(np.round(base_hours), 0, 23).astype(int)
    
    def generate_locations(self, n: int, noise_rate: float = 0.0) -> List[Tuple[str, str]]:
        """
        تولید موقعیت‌های جغرافیایی (استان، شهر)
        
        Args:
            n: تعداد نمونه
            noise_rate: نرخ نویز (موقعیت‌های غیرعادی)
            
        Returns:
            لیست تاپل‌های (استان، شهر)
        """
        locations = []
        
        for _ in range(n):
            if np.random.random() < noise_rate:
                # انتخاب شهر دورافتاده برای نویز
                city = np.random.choice(self.iran_config.REMOTE_LOCATIONS)
                province = "سیستان و بلوچستان"  # اکثر شهرهای دورافتاده
                if city in ["بندر عباس", "میناب", "کنگان", "بندر لنگه", "جاسک"]:
                    province = "هرمزگان"
            else:
                # انتخاب عادی با وزن
                city = np.random.choice(self.all_cities, p=self.city_weights)
                province = self.city_to_province[city]
            
            locations.append((province, city))
        
        return locations
    
    def generate_card_types(self, n: int) -> List[str]:
        """
        تولید انواع کارت با توزیع وزن‌دار
        
        Args:
            n: تعداد نمونه
            
        Returns:
            لیست انواع کارت
        """
        return np.random.choice(
            self.card_types, 
            size=n, 
            p=self.card_weights
        ).tolist()
    
    def generate_device_types(self, n: int) -> List[str]:
        """
        تولید انواع دستگاه با توزیع وزن‌دار
        
        Args:
            n: تعداد نمونه
            
        Returns:
            لیست انواع دستگاه
        """
        return np.random.choice(
            self.device_types, 
            size=n, 
            p=self.device_weights
        ).tolist()
    
    def generate_transaction_times_in_day(self, hours: np.ndarray) -> List[str]:
        """
        تولید زمان دقیق تراکنش در روز
        
        Args:
            hours: آرایه ساعت‌ها
            
        Returns:
            لیست زمان‌ها در فرمت HH:MM:SS
        """
        times = []
        for hour in hours:
            # دقیقه و ثانیه تصادفی
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            times.append(f"{hour:02d}:{minute:02d}:{second:02d}")
        
        return times
    
    def generate_transaction_dates(self, start_date: datetime, 
                                 end_date: datetime, 
                                 n: int) -> List[str]:
        """
        تولید تاریخ‌های تراکنش در بازه زمانی
        
        Args:
            start_date: تاریخ شروع
            end_date: تاریخ پایان
            n: تعداد نمونه
            
        Returns:
            لیست تاریخ‌ها در فرمت YYYY-MM-DD
        """
        # محاسبه تعداد روزها
        total_days = (end_date - start_date).days + 1
        
        # تولید تاریخ‌های تصادفی
        random_days = np.random.randint(0, total_days, n)
        dates = []
        
        for day_offset in random_days:
            date = start_date + timedelta(days=int(day_offset))
            dates.append(date.strftime("%Y-%m-%d"))
        
        return dates
    
    def is_weekend(self, date_str: str) -> bool:
        """
        تشخیص آخر هفته (جمعه در ایران)
        
        Args:
            date_str: تاریخ در فرمت YYYY-MM-DD
            
        Returns:
            True اگر آخر هفته باشد
        """
        date = datetime.strptime(date_str, "%Y-%m-%d")
        # در ایران جمعه آخر هفته است (weekday=4)
        return date.weekday() == 4
    
    def get_distribution_summary(self) -> Dict[str, Any]:
        """
        خلاصه‌ای از توزیع‌های استفاده‌شده
        
        Returns:
            دیکشنری حاوی اطلاعات توزیع‌ها
        """
        return {
            'age_distribution': {
                'type': 'Beta',
                'parameters': {
                    'alpha': self.dist_config.age_beta_alpha,
                    'beta': self.dist_config.age_beta_beta,
                    'range': f"[{self.dist_config.age_min}, {self.dist_config.age_max}]"
                },
                'reasoning': 'توزیع Beta برای سن مناسب است چون شکل کنترل‌پذیر دارد و می‌توان گروه‌های سنی فعال را مدل کرد'
            },
            'amount_distribution': {
                'type': 'Log-Normal',
                'parameters': {
                    'mu': self.dist_config.amount_lognorm_mean,
                    'sigma': self.dist_config.amount_lognorm_sigma,
                    'range': f"[{self.dist_config.amount_min}, {self.dist_config.amount_max}]"
                },
                'reasoning': 'توزیع Log-Normal برای مبالغ مالی طبیعی است: اکثر تراکنش‌ها کم، تعداد کمی بزرگ'
            },
            'daily_transactions': {
                'type': 'Poisson',
                'parameters': {
                    'lambda': self.dist_config.daily_transactions_lambda
                },
                'reasoning': 'Poisson برای شمارش رویدادهای تصادفی (تعداد تراکنش روزانه) ایده‌آل است'
            },
            'hour_distribution': {
                'type': 'Beta (scaled)',
                'parameters': {
                    'alpha': self.dist_config.hour_beta_alpha,
                    'beta': self.dist_config.hour_beta_beta,
                    'active_hours': '[6, 22]'
                },
                'reasoning': 'Beta برای ساعات روز امکان شبیه‌سازی ساعات فعال (روز) و غیرفعال (شب) را فراهم می‌کند'
            },
            'location_distribution': {
                'type': 'Weighted Sampling',
                'parameters': {
                    'provinces': len(self.provinces),
                    'total_cities': len(self.all_cities)
                },
                'reasoning': 'نمونه‌گیری وزن‌دار براساس جمعیت واقعی استان‌ها و شهرهای ایران'
            }
        } 