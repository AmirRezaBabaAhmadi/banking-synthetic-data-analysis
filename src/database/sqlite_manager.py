"""
کلاس مدیریت دیتابیس SQLite با پشتیبانی از chunk-based operations
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Iterator, Tuple
from contextlib import contextmanager
import os
from tqdm import tqdm

from .schema import ALL_SCHEMAS, get_create_index_queries
from ..utils.config import get_config

# تنظیم logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLiteManager:
    """کلاس مدیریت دیتابیس SQLite با قابلیت‌های پیشرفته"""
    
    def __init__(self, db_path: str = None):
        """
        مقداردهی اولیه
        
        Args:
            db_path: مسیر فایل دیتابیس (اختیاری)
        """
        config = get_config()
        self.db_path = db_path or config['database'].db_path
        self.chunk_size = config['database'].chunk_size
        self.batch_size = config['database'].batch_size
        self.enable_wal = config['database'].enable_wal
        
        # ایجاد دیرکتوری دیتابیس در صورت عدم وجود
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # بهینه‌سازی دیتابیس
        self._optimize_database()
        
        # ایجاد جداول
        self.create_tables()
        
    def _optimize_database(self):
        """بهینه‌سازی تنظیمات دیتابیس"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # فعال‌سازی WAL mode برای عملکرد بهتر
            if self.enable_wal:
                cursor.execute("PRAGMA journal_mode=WAL;")
            
            # تنظیمات بهینه‌سازی
            optimizations = [
                "PRAGMA synchronous=NORMAL;",        # کاهش sync برای سرعت بیشتر
                "PRAGMA cache_size=10000;",          # افزایش cache
                "PRAGMA temp_store=MEMORY;",         # استفاده از RAM برای temp
                "PRAGMA mmap_size=268435456;",       # 256MB memory map
                "PRAGMA optimize;",                  # بهینه‌سازی عمومی
            ]
            
            for pragma in optimizations:
                try:
                    cursor.execute(pragma)
                    logger.debug(f"Applied optimization: {pragma}")
                except Exception as e:
                    logger.warning(f"Failed to apply {pragma}: {e}")
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Context manager برای اتصال ایمن به دیتابیس"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # برای دسترسی dict-like
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def create_tables(self):
        """ایجاد تمام جداول با schema تعریف‌شده"""
        logger.info("Creating database tables...")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # ایجاد جداول اصلی
            for schema in ALL_SCHEMAS:
                try:
                    cursor.execute(schema)
                    logger.debug("Created table from schema")
                except Exception as e:
                    logger.error(f"Failed to create table: {e}")
                    raise
            
            # ایجاد ایندکس‌های اضافی
            for index_query in get_create_index_queries():
                try:
                    cursor.execute(index_query)
                    logger.debug(f"Created index: {index_query[:50]}...")
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")
            
            conn.commit()
            logger.info("Database tables created successfully")
    
    def insert_users_chunk(self, users_df: pd.DataFrame) -> int:
        """
        درج chunk از کاربران
        
        Args:
            users_df: DataFrame کاربران
            
        Returns:
            تعداد رکوردهای درج‌شده
        """
        with self.get_connection() as conn:
            try:
                # تبدیل DataFrame به records
                records = users_df.to_dict('records')
                
                # تعریف کوئری INSERT
                insert_query = """
                INSERT OR REPLACE INTO users 
                (user_id, age, birth_year, registration_date, province, city, 
                 preferred_card_type, primary_device)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                # آماده‌سازی داده‌ها
                data_tuples = [
                    (
                        record['user_id'],
                        record['age'],
                        record['birth_year'],
                        record['registration_date'],
                        record['province'],
                        record['city'],
                        record['preferred_card_type'],
                        record['primary_device']
                    )
                    for record in records
                ]
                
                # درج batch‌وار
                cursor = conn.cursor()
                cursor.executemany(insert_query, data_tuples)
                conn.commit()
                
                logger.debug(f"Inserted {len(data_tuples)} users")
                return len(data_tuples)
                
            except Exception as e:
                logger.error(f"Error inserting users: {e}")
                raise
    
    def insert_transactions_chunk(self, transactions_df: pd.DataFrame) -> int:
        """
        درج chunk از تراکنش‌ها
        
        Args:
            transactions_df: DataFrame تراکنش‌ها
            
        Returns:
            تعداد رکوردهای درج‌شده
        """
        with self.get_connection() as conn:
            try:
                # تبدیل DataFrame به records
                records = transactions_df.to_dict('records')
                
                # تعریف کوئری INSERT
                insert_query = """
                INSERT INTO transactions 
                (user_id, amount, transaction_date, transaction_time, province, city,
                 card_type, device_type, is_weekend, hour_of_day, day_of_month,
                 is_noise, noise_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                # آماده‌سازی داده‌ها
                data_tuples = [
                    (
                        record['user_id'],
                        record['amount'],
                        record['transaction_date'],
                        record['transaction_time'],
                        record['province'],
                        record['city'],
                        record['card_type'],
                        record['device_type'],
                        record['is_weekend'],
                        record['hour_of_day'],
                        record['day_of_month'],
                        record.get('is_noise', False),
                        record.get('noise_type', None)
                    )
                    for record in records
                ]
                
                # درج batch‌وار
                cursor = conn.cursor()
                cursor.executemany(insert_query, data_tuples)
                conn.commit()
                
                logger.debug(f"Inserted {len(data_tuples)} transactions")
                return len(data_tuples)
                
            except Exception as e:
                logger.error(f"Error inserting transactions: {e}")
                raise
    
    def get_users_batch(self, batch_size: int = None, offset: int = 0) -> pd.DataFrame:
        """
        دریافت batch از کاربران
        
        Args:
            batch_size: اندازه batch
            offset: شروع batch
            
        Returns:
            DataFrame کاربران
        """
        batch_size = batch_size or self.batch_size
        
        query = """
        SELECT * FROM users 
        ORDER BY user_id 
        LIMIT ? OFFSET ?
        """
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=[batch_size, offset])
    
    def get_transactions_batch(self, batch_size: int = None, offset: int = 0) -> pd.DataFrame:
        """
        دریافت batch از تراکنش‌ها
        
        Args:
            batch_size: اندازه batch
            offset: شروع batch
            
        Returns:
            DataFrame تراکنش‌ها
        """
        batch_size = batch_size or self.batch_size
        
        query = """
        SELECT * FROM transactions 
        ORDER BY transaction_id 
        LIMIT ? OFFSET ?
        """
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=[batch_size, offset])
    
    def get_user_transactions(self, user_id: int) -> pd.DataFrame:
        """
        دریافت تمام تراکنش‌های یک کاربر
        
        Args:
            user_id: شناسه کاربر
            
        Returns:
            DataFrame تراکنش‌های کاربر
        """
        query = """
        SELECT * FROM transactions 
        WHERE user_id = ? 
        ORDER BY transaction_date, transaction_time
        """
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=[user_id])
    
    def execute_query(self, query: str, params: List = None) -> pd.DataFrame:
        """
        اجرای کوئری SQL دلخواه
        
        Args:
            query: کوئری SQL
            params: پارامترهای کوئری
            
        Returns:
            نتیجه کوئری
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params or [])
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        دریافت آمار کلی دیتابیس
        
        Returns:
            دیکشنری حاوی آمارهای دیتابیس
        """
        stats = {}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # تعداد کاربران
            cursor.execute("SELECT COUNT(*) FROM users")
            stats['total_users'] = cursor.fetchone()[0]
            
            # تعداد تراکنش‌ها
            cursor.execute("SELECT COUNT(*) FROM transactions")
            stats['total_transactions'] = cursor.fetchone()[0]
            
            # مجموع مبلغ تراکنش‌ها
            cursor.execute("SELECT SUM(amount) FROM transactions")
            stats['total_amount'] = cursor.fetchone()[0] or 0
            
            # میانگین تراکنش در کاربر
            if stats['total_users'] > 0:
                stats['avg_transactions_per_user'] = stats['total_transactions'] / stats['total_users']
            else:
                stats['avg_transactions_per_user'] = 0
            
            # اندازه فایل دیتابیس
            stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
            
            # تعداد نویزها
            cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_noise = TRUE")
            stats['noise_transactions'] = cursor.fetchone()[0]
            
            if stats['total_transactions'] > 0:
                stats['noise_percentage'] = (stats['noise_transactions'] / stats['total_transactions']) * 100
            else:
                stats['noise_percentage'] = 0
        
        return stats
    
    def iter_users_batches(self, batch_size: int = None) -> Iterator[pd.DataFrame]:
        """
        Iterator برای پردازش batch‌وار کاربران
        
        Args:
            batch_size: اندازه هر batch
            
        Yields:
            DataFrame هر batch از کاربران
        """
        batch_size = batch_size or self.batch_size
        offset = 0
        
        while True:
            batch = self.get_users_batch(batch_size, offset)
            if batch.empty:
                break
            yield batch
            offset += batch_size
    
    def iter_transactions_batches(self, batch_size: int = None) -> Iterator[pd.DataFrame]:
        """
        Iterator برای پردازش batch‌وار تراکنش‌ها
        
        Args:
            batch_size: اندازه هر batch
            
        Yields:
            DataFrame هر batch از تراکنش‌ها
        """
        batch_size = batch_size or self.batch_size
        offset = 0
        
        while True:
            batch = self.get_transactions_batch(batch_size, offset)
            if batch.empty:
                break
            yield batch
            offset += batch_size
    
    def vacuum_database(self):
        """پاکسازی و بهینه‌سازی دیتابیس"""
        logger.info("Vacuuming database...")
        with self.get_connection() as conn:
            conn.execute("VACUUM;")
            conn.execute("ANALYZE;")
        logger.info("Database vacuumed successfully")
    
    def backup_database(self, backup_path: str):
        """پشتیبان‌گیری از دیتابیس"""
        import shutil
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")
    
    def close(self):
        """بستن اتصالات و پاکسازی منابع"""
        # در SQLite معمولاً نیازی به cleanup خاص نیست
        # اما می‌توان VACUUM اجرا کرد
        self.vacuum_database()
        logger.info("Database manager closed") 