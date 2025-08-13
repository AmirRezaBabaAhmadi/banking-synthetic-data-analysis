"""
کلاس‌های اصلی تولید داده‌های synthetic بانکی
"""

import numpy as np
import polars as pl
import pandas as pd  # Keep for backward compatibility
from typing import List, Dict, Any, Tuple, Iterator
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

from .distributions import StatisticalDistributions
from .noise_injection import NoiseInjector
from ..database.sqlite_manager import SQLiteManager
from ..utils.config import get_config
import os
import json

logger = logging.getLogger(__name__)

class BankingDataGenerator:
    """کلاس اصلی تولید داده‌های بانکی synthetic"""
    
    def __init__(self, db_manager: SQLiteManager = None):
        """
        مقداردهی اولیه
        
        Args:
            db_manager: مدیر دیتابیس (اختیاری)
        """
        self.config = get_config()
        self.data_config = self.config['data_generation']
        
        # ابزارهای تولید داده
        self.distributions = StatisticalDistributions()
        self.noise_injector = NoiseInjector()
        
        # مدیر دیتابیس
        self.db_manager = db_manager or SQLiteManager()
        
        # آمار تولید
        self.generation_stats = {
            'users_generated': 0,
            'transactions_generated': 0,
            'chunks_processed': 0,
            'total_time': 0
        }
        
        logger.info(f"BankingDataGenerator initialized for {self.data_config.total_users:,} users")
    
    def generate_users_chunk(self, chunk_size: int, start_user_id: int = 1) -> pl.DataFrame:
        """
        تولید chunk از کاربران
        
        Args:
            chunk_size: تعداد کاربران در chunk
            start_user_id: شناسه شروع کاربران
            
        Returns:
            DataFrame کاربران
        """
        logger.debug(f"Generating {chunk_size} users starting from ID {start_user_id}")
        
        # تولید user IDs
        user_ids = np.arange(start_user_id, start_user_id + chunk_size)
        
        # تولید سن‌ها
        ages = self.distributions.generate_ages(chunk_size)
        
        # محاسبه سال تولد (فرض: سال جاری 2024)
        current_year = 2024
        birth_years = current_year - ages
        
        # تولید تاریخ ثبت‌نام (6 ماه تا 5 سال پیش)
        days_ago = np.random.randint(180, 1825, chunk_size)  # 6 ماه تا 5 سال
        registration_dates = []
        base_date = datetime(2024, 1, 1)
        
        for days in days_ago:
            reg_date = base_date - timedelta(days=int(days))
            registration_dates.append(reg_date.strftime("%Y-%m-%d"))
        
        # تولید موقعیت‌ها
        locations = self.distributions.generate_locations(chunk_size)
        provinces, cities = zip(*locations)
        
        # تولید انواع کارت و دستگاه
        card_types = self.distributions.generate_card_types(chunk_size)
        device_types = self.distributions.generate_device_types(chunk_size)
        
        # ایجاد Polars DataFrame
        users_df = pl.DataFrame({
            'user_id': user_ids,
            'age': ages,
            'birth_year': birth_years,
            'registration_date': registration_dates,
            'province': provinces,
            'city': cities,
            'preferred_card_type': card_types,
            'primary_device': device_types
        })
        
        # اعمال نویز به کاربران
        users_df = self.noise_injector.generate_controlled_user_noise(users_df)
        
        self.generation_stats['users_generated'] += chunk_size
        
        return users_df
    
    def generate_user_transactions(self, user_data: Dict[str, Any]) -> pl.DataFrame:
        """
        تولید تراکنش‌های یک کاربر
        
        Args:
            user_data: اطلاعات کاربر (دیکشنری)
            
        Returns:
            DataFrame تراکنش‌های کاربر
        """
        user_id = user_data['user_id']
        user_age = user_data['age']
        card_type = user_data['preferred_card_type']
        device_type = user_data['primary_device']
        user_province = user_data['province']
        user_city = user_data['city']
        
        # تعداد تراکنش برای این کاربر
        num_transactions = np.random.randint(
            self.data_config.min_transactions_per_user,
            self.data_config.max_transactions_per_user + 1
        )
        
        # تولید تاریخ‌های تراکنش
        transaction_dates = self.distributions.generate_transaction_dates(
            self.data_config.start_date,
            self.data_config.end_date,
            num_transactions
        )
        
        # تولید ساعات تراکنش
        hours = self.distributions.generate_transaction_hours(num_transactions)
        
        # تولید زمان‌های کامل
        transaction_times = self.distributions.generate_transaction_times_in_day(hours)
        
        # تولید مبالغ تراکنش
        amounts = self.distributions.generate_transaction_amounts(
            num_transactions, 
            card_type=card_type,
            user_age=user_age
        )
        
        # تولید موقعیت‌های تراکنش (اکثر در شهر خود کاربر)
        transaction_locations = []
        for _ in range(num_transactions):
            # 80% احتمال در همان شهر کاربر
            if np.random.random() < 0.8:
                transaction_locations.append((user_province, user_city))
            else:
                # 20% احتمال در شهر دیگر
                locations = self.distributions.generate_locations(1)
                transaction_locations.append(locations[0])
        
        trans_provinces, trans_cities = zip(*transaction_locations)
        
        # تولید انواع کارت (معمولاً همان کارت ترجیحی)
        trans_card_types = []
        for _ in range(num_transactions):
            if np.random.random() < 0.7:  # 70% همان کارت ترجیحی
                trans_card_types.append(card_type)
            else:  # 30% کارت دیگر
                trans_card_types.extend(self.distributions.generate_card_types(1))
        
        # تولید انواع دستگاه
        trans_device_types = []
        for _ in range(num_transactions):
            if np.random.random() < 0.6:  # 60% همان دستگاه اصلی
                trans_device_types.append(device_type)
            else:  # 40% دستگاه دیگر
                trans_device_types.extend(self.distributions.generate_device_types(1))
        
        # محاسبه فیلدهای اضافی
        is_weekends = [self.distributions.is_weekend(date) for date in transaction_dates]
        days_of_month = [int(date.split('-')[2]) for date in transaction_dates]
        
        # ایجاد Polars DataFrame تراکنش‌ها
        transactions_df = pl.DataFrame({
            'user_id': [user_id] * num_transactions,
            'amount': amounts,
            'transaction_date': transaction_dates,
            'transaction_time': transaction_times,
            'province': trans_provinces,
            'city': trans_cities,
            'card_type': trans_card_types,
            'device_type': trans_device_types,
            'is_weekend': is_weekends,
            'hour_of_day': hours,
            'day_of_month': days_of_month
        })
        
        return transactions_df
    
    def generate_and_save_chunk(self, chunk_id: int, chunk_size: int) -> Dict[str, int]:
        """
        تولید و ذخیره یک chunk کامل (کاربران + تراکنش‌ها)
        
        Args:
            chunk_id: شناسه chunk
            chunk_size: اندازه chunk
            
        Returns:
            آمار chunk تولیدشده
        """
        start_time = datetime.now()
        start_user_id = chunk_id * chunk_size + 1
        
        logger.info(f"Processing chunk {chunk_id + 1}: users {start_user_id} to {start_user_id + chunk_size - 1}")
        
        # تولید کاربران
        users_df = self.generate_users_chunk(chunk_size, start_user_id)
        
        # ذخیره کاربران در دیتابیس
        users_inserted = self.db_manager.insert_users_chunk(users_df)
        
        # تولید تراکنش‌ها برای هر کاربر
        all_transactions = []
        
        # تبدیل Polars به دیکشنری برای iteration
        users_dicts = users_df.to_dicts()
        
        for user_dict in tqdm(users_dicts, 
                              desc=f"Generating transactions for chunk {chunk_id + 1}",
                              leave=False):
            
            user_transactions = self.generate_user_transactions(user_dict)
            all_transactions.append(user_transactions)
        
        # ترکیب تمام تراکنش‌ها
        if all_transactions:
            chunk_transactions_df = pl.concat(all_transactions)
        else:
            chunk_transactions_df = pl.DataFrame()
        
        # اعمال نویز به تراکنش‌ها
        if not chunk_transactions_df.is_empty():
            chunk_transactions_df = self.noise_injector.apply_comprehensive_noise(
                chunk_transactions_df, users_df
            )
            
            # ذخیره تراکنش‌ها در دیتابیس
            transactions_inserted = self.db_manager.insert_transactions_chunk(chunk_transactions_df)
        else:
            transactions_inserted = 0
        
        # محاسبه زمان پردازش
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # بروزرسانی آمار
        self.generation_stats['chunks_processed'] += 1
        self.generation_stats['transactions_generated'] += transactions_inserted
        self.generation_stats['total_time'] += processing_time
        
        chunk_stats = {
            'chunk_id': chunk_id,
            'users_inserted': users_inserted,
            'transactions_inserted': transactions_inserted,
            'processing_time_seconds': processing_time
        }
        
        logger.info(f"Chunk {chunk_id + 1} completed: "
                   f"{users_inserted} users, {transactions_inserted} transactions "
                   f"in {processing_time:.2f}s")

        # Checkpoint پس از هر 10,000 کاربر (یا پایان هر chunk)
        try:
            os.makedirs("output/checkpoints", exist_ok=True)
            checkpoint = {
                "timestamp": datetime.now().isoformat(),
                "chunk_id": chunk_id,
                "start_user_id": start_user_id,
                "users_inserted": users_inserted,
                "transactions_inserted": transactions_inserted,
                "total_users_generated": self.generation_stats['users_generated'],
                "total_transactions_generated": self.generation_stats['transactions_generated'],
                "processing_time_seconds": processing_time
            }
            # نام فایل: هر 10000 کاربر یک checkpoint
            checkpoint_idx = self.generation_stats['users_generated'] // 10000
            checkpoint_path = os.path.join("output", "checkpoints", f"checkpoint_{checkpoint_idx:05d}.json")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
        
        return chunk_stats
    
    def generate_complete_dataset(self) -> Dict[str, Any]:
        """
        تولید dataset کامل به صورت chunk-based
        
        Returns:
            آمار کلی تولید داده
        """
        start_time = datetime.now()
        
        logger.info(f"Starting complete dataset generation for {self.data_config.total_users:,} users")
        
        # محاسبه تعداد chunks
        chunk_size = self.config['database'].chunk_size
        total_chunks = (self.data_config.total_users + chunk_size - 1) // chunk_size
        
        logger.info(f"Processing {total_chunks} chunks of size {chunk_size}")
        
        # پردازش هر chunk
        chunk_stats_list = []
        
        for chunk_id in tqdm(range(total_chunks), desc="Processing chunks"):
            # محاسبه اندازه chunk فعلی (chunk آخر ممکن است کوچکتر باشد)
            current_chunk_size = min(chunk_size, 
                                   self.data_config.total_users - chunk_id * chunk_size)
            
            chunk_stats = self.generate_and_save_chunk(chunk_id, current_chunk_size)
            chunk_stats_list.append(chunk_stats)
            
            # نمایش پیشرفت
            if (chunk_id + 1) % 10 == 0:
                logger.info(f"Processed {chunk_id + 1}/{total_chunks} chunks")
        
        # محاسبه آمار نهایی
        total_time = (datetime.now() - start_time).total_seconds()
        
        final_stats = {
            'generation_completed_at': datetime.now().isoformat(),
            'total_processing_time_seconds': total_time,
            'total_users_generated': self.generation_stats['users_generated'],
            'total_transactions_generated': self.generation_stats['transactions_generated'],
            'chunks_processed': self.generation_stats['chunks_processed'],
            'chunk_stats': chunk_stats_list,
            'average_chunk_time': total_time / total_chunks if total_chunks > 0 else 0,
            'transactions_per_second': self.generation_stats['transactions_generated'] / total_time if total_time > 0 else 0,
            'database_stats': self.db_manager.get_database_stats(),
            'noise_statistics': self.noise_injector.get_noise_statistics()
        }
        
        logger.info(f"Dataset generation completed in {total_time:.2f}s")
        logger.info(f"Generated {final_stats['total_users_generated']:,} users and "
                   f"{final_stats['total_transactions_generated']:,} transactions")
        
        return final_stats
    
    def generate_sample_data(self, num_users: int = 1000) -> Dict[str, Any]:
        """
        تولید داده نمونه برای تست
        
        Args:
            num_users: تعداد کاربران نمونه
            
        Returns:
            آمار داده نمونه
        """
        logger.info(f"Generating sample data for {num_users} users")
        
        # تنظیم موقت تعداد کاربران
        original_total = self.data_config.total_users
        self.data_config.total_users = num_users
        
        try:
            # تولید داده
            stats = self.generate_complete_dataset()
            
            logger.info("Sample data generation completed")
            return stats
            
        finally:
            # بازگرداندن تنظیمات اصلی
            self.data_config.total_users = original_total
    
    def get_generation_report(self) -> str:
        """
        تولید گزارش کامل فرآیند تولید داده
        
        Returns:
            گزارش متنی
        """
        db_stats = self.db_manager.get_database_stats()
        noise_stats = self.noise_injector.get_noise_statistics()
        
        report = "=== گزارش تولید داده‌های بانکی ===\n\n"
        
        # آمار کلی
        report += "آمار کلی:\n"
        report += f"  کل کاربران: {db_stats['total_users']:,}\n"
        report += f"  کل تراکنش‌ها: {db_stats['total_transactions']:,}\n"
        report += f"  میانگین تراکنش در کاربر: {db_stats['avg_transactions_per_user']:.1f}\n"
        report += f"  مجموع مبلغ تراکنش‌ها: {db_stats['total_amount']:,.0f} تومان\n"
        report += f"  اندازه دیتابیس: {db_stats['database_size_mb']:.1f} MB\n\n"
        
        # آمار نویز
        report += "آمار نویز:\n"
        report += f"  کل رکوردهای نویز: {noise_stats['noise_counts']['total_noise_records']:,}\n"
        report += f"  درصد نویز: {db_stats['noise_percentage']:.2f}%\n\n"
        
        # توزیع‌های استفاده‌شده
        dist_summary = self.distributions.get_distribution_summary()
        report += "توزیع‌های آماری استفاده‌شده:\n"
        for dist_name, dist_info in dist_summary.items():
            report += f"  {dist_name}:\n"
            report += f"    نوع: {dist_info['type']}\n"
            report += f"    دلیل انتخاب: {dist_info['reasoning']}\n\n"
        
        # تنظیمات کلیدی
        report += "تنظیمات کلیدی:\n"
        report += f"  بازه سنی: {self.config['distributions'].age_min}-{self.config['distributions'].age_max} سال\n"
        report += f"  بازه مبلغ: {self.config['distributions'].amount_min:,.0f}-{self.config['distributions'].amount_max:,.0f} تومان\n"
        report += f"  بازه تراکنش در کاربر: {self.data_config.min_transactions_per_user}-{self.data_config.max_transactions_per_user}\n"
        report += f"  بازه زمانی: {self.data_config.start_date.date()} تا {self.data_config.end_date.date()}\n"
        
        return report
    
    def export_generation_report(self, output_path: str):
        """
        صادرات گزارش تولید به فایل
        
        Args:
            output_path: مسیر فایل خروجی
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.get_generation_report())
        
        logger.info(f"Generation report exported to {output_path}")
    
    def close(self):
        """بستن منابع"""
        if self.db_manager:
            self.db_manager.close()
        logger.info("BankingDataGenerator closed")


class NewUserGenerator:
    """کلاس تولید کاربران جدید برای similarity search"""
    
    def __init__(self):
        """مقداردهی اولیه"""
        self.distributions = StatisticalDistributions()
        self.config = get_config()
    
    def generate_new_users(self, count: int = 100) -> pl.DataFrame:
        """
        تولید کاربران جدید برای similarity search
        
        Args:
            count: تعداد کاربران جدید
            
        Returns:
            Polars DataFrame کاربران جدید
        """
        logger.info(f"Generating {count} new users for similarity search")
        
        # تولید user IDs موقت (منفی برای جدا کردن از کاربران اصلی)
        user_ids = np.arange(-1, -count - 1, -1)
        
        # تولید سایر اطلاعات مشابه کاربران اصلی
        ages = self.distributions.generate_ages(count)
        birth_years = 2024 - ages
        
        # تاریخ ثبت‌نام جدید
        registration_dates = [datetime.now().strftime("%Y-%m-%d")] * count
        
        # موقعیت‌ها
        locations = self.distributions.generate_locations(count)
        provinces, cities = zip(*locations)
        
        # انواع کارت و دستگاه
        card_types = self.distributions.generate_card_types(count)
        device_types = self.distributions.generate_device_types(count)
        
        # ایجاد Polars DataFrame
        new_users_df = pl.DataFrame({
            'user_id': user_ids,
            'age': ages,
            'birth_year': birth_years,
            'registration_date': registration_dates,
            'province': provinces,
            'city': cities,
            'preferred_card_type': card_types,
            'primary_device': device_types
        })
        
        logger.info(f"Generated {count} new users successfully")
        return new_users_df 