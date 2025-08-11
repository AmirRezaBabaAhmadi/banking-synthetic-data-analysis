"""
کلاس مدیریت async دیتابیس SQLite با پشتیبانی از Polars و aiosqlite
"""

import aiosqlite
import polars as pl
import numpy as np
import logging
import asyncio
from typing import List, Dict, Any, Optional, Iterator, Tuple, AsyncIterator
from contextlib import asynccontextmanager
import os
from tqdm.asyncio import tqdm

from .schema import ALL_SCHEMAS, get_create_index_queries
from ..utils.config import get_config

# تنظیم logging
logger = logging.getLogger(__name__)

class AsyncSQLiteManager:
    """کلاس async مدیریت دیتابیس SQLite با قابلیت‌های پیشرفته و Polars"""
    
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
        
        # متغیر برای نگهداری connection pool
        self._connection_pool = None
        
    async def initialize(self):
        """مقداردهی async - باید بعد از ایجاد object صدا زده شود"""
        await self._optimize_database()
        await self.create_tables()
        
    async def _optimize_database(self):
        """بهینه‌سازی async تنظیمات دیتابیس"""
        async with self.get_connection() as conn:
            # فعال‌سازی WAL mode برای عملکرد بهتر
            if self.enable_wal:
                await conn.execute("PRAGMA journal_mode=WAL;")
            
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
                    await conn.execute(pragma)
                    logger.debug(f"Applied optimization: {pragma}")
                except Exception as e:
                    logger.warning(f"Failed to apply {pragma}: {e}")
            
            await conn.commit()
    
    @asynccontextmanager
    async def get_connection(self):
        """Context manager async برای اتصال ایمن به دیتابیس"""
        conn = None
        try:
            conn = await aiosqlite.connect(self.db_path, timeout=30.0)
            conn.row_factory = aiosqlite.Row  # برای دسترسی dict-like
            yield conn
        except Exception as e:
            if conn:
                await conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                await conn.close()
    
    async def create_tables(self):
        """ایجاد async تمام جداول با schema تعریف‌شده"""
        logger.info("Creating database tables asynchronously...")
        
        async with self.get_connection() as conn:
            # ایجاد جداول اصلی
            for schema in ALL_SCHEMAS:
                try:
                    await conn.execute(schema)
                    logger.debug("Created table from schema")
                except Exception as e:
                    logger.error(f"Failed to create table: {e}")
                    raise
            
            # ایجاد ایندکس‌های اضافی
            for index_query in get_create_index_queries():
                try:
                    await conn.execute(index_query)
                    logger.debug(f"Created index: {index_query[:50]}...")
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")
            
            await conn.commit()
            logger.info("Database tables created successfully")
    
    async def insert_users_chunk(self, users_df: pl.DataFrame) -> int:
        """
        درج async chunk از کاربران
        
        Args:
            users_df: Polars DataFrame کاربران
            
        Returns:
            تعداد رکوردهای درج‌شده
        """
        async with self.get_connection() as conn:
            try:
                # تبدیل Polars DataFrame به records
                records = users_df.to_dicts()
                
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
                
                # درج async batch‌وار
                await conn.executemany(insert_query, data_tuples)
                await conn.commit()
                
                logger.debug(f"Inserted {len(data_tuples)} users asynchronously")
                return len(data_tuples)
                
            except Exception as e:
                logger.error(f"Error inserting users: {e}")
                raise
    
    async def insert_transactions_chunk(self, transactions_df: pl.DataFrame) -> int:
        """
        درج async chunk از تراکنش‌ها
        
        Args:
            transactions_df: Polars DataFrame تراکنش‌ها
            
        Returns:
            تعداد رکوردهای درج‌شده
        """
        async with self.get_connection() as conn:
            try:
                # تبدیل Polars DataFrame به records
                records = transactions_df.to_dicts()
                
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
                
                # درج async batch‌وار
                await conn.executemany(insert_query, data_tuples)
                await conn.commit()
                
                logger.debug(f"Inserted {len(data_tuples)} transactions asynchronously")
                return len(data_tuples)
                
            except Exception as e:
                logger.error(f"Error inserting transactions: {e}")
                raise
    
    async def get_users_batch(self, batch_size: int = None, offset: int = 0) -> pl.DataFrame:
        """
        دریافت async batch از کاربران
        
        Args:
            batch_size: اندازه batch
            offset: شروع batch
            
        Returns:
            Polars DataFrame کاربران
        """
        batch_size = batch_size or self.batch_size
        
        query = """
        SELECT * FROM users 
        ORDER BY user_id 
        LIMIT ? OFFSET ?
        """
        
        async with self.get_connection() as conn:
            cursor = await conn.execute(query, [batch_size, offset])
            rows = await cursor.fetchall()
            
            if not rows:
                return pl.DataFrame()
            
            # تبدیل به Polars DataFrame
            columns = [description[0] for description in cursor.description]
            data = [dict(zip(columns, row)) for row in rows]
            
            return pl.DataFrame(data)
    
    async def get_transactions_batch(self, batch_size: int = None, offset: int = 0) -> pl.DataFrame:
        """
        دریافت async batch از تراکنش‌ها
        
        Args:
            batch_size: اندازه batch
            offset: شروع batch
            
        Returns:
            Polars DataFrame تراکنش‌ها
        """
        batch_size = batch_size or self.batch_size
        
        query = """
        SELECT * FROM transactions 
        ORDER BY transaction_id 
        LIMIT ? OFFSET ?
        """
        
        async with self.get_connection() as conn:
            cursor = await conn.execute(query, [batch_size, offset])
            rows = await cursor.fetchall()
            
            if not rows:
                return pl.DataFrame()
            
            # تبدیل به Polars DataFrame
            columns = [description[0] for description in cursor.description]
            data = [dict(zip(columns, row)) for row in rows]
            
            return pl.DataFrame(data)
    
    async def get_user_transactions(self, user_id: int) -> pl.DataFrame:
        """
        دریافت async تمام تراکنش‌های یک کاربر
        
        Args:
            user_id: شناسه کاربر
            
        Returns:
            Polars DataFrame تراکنش‌های کاربر
        """
        query = """
        SELECT * FROM transactions 
        WHERE user_id = ? 
        ORDER BY transaction_date, transaction_time
        """
        
        async with self.get_connection() as conn:
            cursor = await conn.execute(query, [user_id])
            rows = await cursor.fetchall()
            
            if not rows:
                return pl.DataFrame()
            
            # تبدیل به Polars DataFrame
            columns = [description[0] for description in cursor.description]
            data = [dict(zip(columns, row)) for row in rows]
            
            return pl.DataFrame(data)
    
    async def execute_query(self, query: str, params: List = None) -> pl.DataFrame:
        """
        اجرای async کوئری SQL دلخواه
        
        Args:
            query: کوئری SQL
            params: پارامترهای کوئری
            
        Returns:
            Polars DataFrame نتیجه کوئری
        """
        async with self.get_connection() as conn:
            cursor = await conn.execute(query, params or [])
            rows = await cursor.fetchall()
            
            if not rows:
                return pl.DataFrame()
            
            # تبدیل به Polars DataFrame
            columns = [description[0] for description in cursor.description]
            data = [dict(zip(columns, row)) for row in rows]
            
            return pl.DataFrame(data)
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """
        دریافت async آمار کلی دیتابیس
        
        Returns:
            دیکشنری حاوی آمارهای دیتابیس
        """
        stats = {}
        
        async with self.get_connection() as conn:
            # تعداد کاربران
            cursor = await conn.execute("SELECT COUNT(*) FROM users")
            result = await cursor.fetchone()
            stats['total_users'] = result[0] if result else 0
            
            # تعداد تراکنش‌ها
            cursor = await conn.execute("SELECT COUNT(*) FROM transactions")
            result = await cursor.fetchone()
            stats['total_transactions'] = result[0] if result else 0
            
            # مجموع مبلغ تراکنش‌ها
            cursor = await conn.execute("SELECT SUM(amount) FROM transactions")
            result = await cursor.fetchone()
            stats['total_amount'] = result[0] if result and result[0] else 0
            
            # میانگین تراکنش در کاربر
            if stats['total_users'] > 0:
                stats['avg_transactions_per_user'] = stats['total_transactions'] / stats['total_users']
            else:
                stats['avg_transactions_per_user'] = 0
            
            # اندازه فایل دیتابیس
            stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
            
            # تعداد نویزها
            cursor = await conn.execute("SELECT COUNT(*) FROM transactions WHERE is_noise = 1")
            result = await cursor.fetchone()
            stats['noise_transactions'] = result[0] if result else 0
            
            if stats['total_transactions'] > 0:
                stats['noise_percentage'] = (stats['noise_transactions'] / stats['total_transactions']) * 100
            else:
                stats['noise_percentage'] = 0
        
        return stats
    
    async def iter_users_batches(self, batch_size: int = None) -> AsyncIterator[pl.DataFrame]:
        """
        Async Iterator برای پردازش batch‌وار کاربران
        
        Args:
            batch_size: اندازه هر batch
            
        Yields:
            Polars DataFrame هر batch از کاربران
        """
        batch_size = batch_size or self.batch_size
        offset = 0
        
        while True:
            batch = await self.get_users_batch(batch_size, offset)
            if batch.is_empty():
                break
            yield batch
            offset += batch_size
    
    async def iter_transactions_batches(self, batch_size: int = None) -> AsyncIterator[pl.DataFrame]:
        """
        Async Iterator برای پردازش batch‌وار تراکنش‌ها
        
        Args:
            batch_size: اندازه هر batch
            
        Yields:
            Polars DataFrame هر batch از تراکنش‌ها
        """
        batch_size = batch_size or self.batch_size
        offset = 0
        
        while True:
            batch = await self.get_transactions_batch(batch_size, offset)
            if batch.is_empty():
                break
            yield batch
            offset += batch_size
    
    async def vacuum_database(self):
        """پاکسازی و بهینه‌سازی async دیتابیس"""
        logger.info("Vacuuming database asynchronously...")
        async with self.get_connection() as conn:
            await conn.execute("VACUUM;")
            await conn.execute("ANALYZE;")
        logger.info("Database vacuumed successfully")
    
    async def backup_database(self, backup_path: str):
        """پشتیبان‌گیری async از دیتابیس"""
        import shutil
        # اجرای backup در thread pool برای جلوگیری از blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, shutil.copy2, self.db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")
    
    async def close(self):
        """بستن async اتصالات و پاکسازی منابع"""
        # اجرای VACUUM برای بهینه‌سازی نهایی
        await self.vacuum_database()
        logger.info("Async database manager closed")

    # متدهای سازگاری برای کد موجود (sync wrappers)
    def execute_query_sync(self, query: str, params: List = None) -> pl.DataFrame:
        """Sync wrapper برای execute_query - برای سازگاری با کد موجود"""
        return asyncio.run(self.execute_query(query, params))
    
    def get_database_stats_sync(self) -> Dict[str, Any]:
        """Sync wrapper برای get_database_stats - برای سازگاری با کد موجود"""
        return asyncio.run(self.get_database_stats())


# کلاس hybrid برای سازگاری با کد موجود
class HybridSQLiteManager:
    """کلاس hybrid که هم sync و هم async عملیات را پشتیبانی می‌کند"""
    
    def __init__(self, db_path: str = None):
        self.async_manager = AsyncSQLiteManager(db_path)
        self.db_path = self.async_manager.db_path
        self.chunk_size = self.async_manager.chunk_size
        self.batch_size = self.async_manager.batch_size
        self.enable_wal = self.async_manager.enable_wal
        
        # مقداردهی sync
        asyncio.run(self.async_manager.initialize())
    
    # Sync methods برای سازگاری
    def execute_query(self, query: str, params: List = None) -> pl.DataFrame:
        """اجرای sync کوئری SQL"""
        return asyncio.run(self.async_manager.execute_query(query, params))
    
    def get_database_stats(self) -> Dict[str, Any]:
        """دریافت sync آمار دیتابیس"""
        return asyncio.run(self.async_manager.get_database_stats())
    
    def get_users_batch(self, batch_size: int = None, offset: int = 0) -> pl.DataFrame:
        """دریافت sync batch از کاربران"""
        return asyncio.run(self.async_manager.get_users_batch(batch_size, offset))
    
    def get_transactions_batch(self, batch_size: int = None, offset: int = 0) -> pl.DataFrame:
        """دریافت sync batch از تراکنش‌ها"""
        return asyncio.run(self.async_manager.get_transactions_batch(batch_size, offset))
    
    def get_user_transactions(self, user_id: int) -> pl.DataFrame:
        """دریافت sync تراکنش‌های کاربر"""
        return asyncio.run(self.async_manager.get_user_transactions(user_id))
    
    def insert_users_chunk(self, users_df: pl.DataFrame) -> int:
        """درج sync chunk از کاربران"""
        return asyncio.run(self.async_manager.insert_users_chunk(users_df))
    
    def insert_transactions_chunk(self, transactions_df: pl.DataFrame) -> int:
        """درج sync chunk از تراکنش‌ها"""
        return asyncio.run(self.async_manager.insert_transactions_chunk(transactions_df))
    
    def close(self):
        """بستن sync manager"""
        asyncio.run(self.async_manager.close())
    
    def vacuum_database(self):
        """پاکسازی sync دیتابیس"""
        asyncio.run(self.async_manager.vacuum_database())
    
    # Async methods
    async def execute_query_async(self, query: str, params: List = None) -> pl.DataFrame:
        """اجرای async کوئری SQL"""
        return await self.async_manager.execute_query(query, params)
    
    async def get_database_stats_async(self) -> Dict[str, Any]:
        """دریافت async آمار دیتابیس"""
        return await self.async_manager.get_database_stats()
    
    async def insert_users_chunk_async(self, users_df: pl.DataFrame) -> int:
        """درج async chunk از کاربران"""
        return await self.async_manager.insert_users_chunk(users_df)
    
    async def insert_transactions_chunk_async(self, transactions_df: pl.DataFrame) -> int:
        """درج async chunk از تراکنش‌ها"""
        return await self.async_manager.insert_transactions_chunk(transactions_df)


# برای سازگاری با کد موجود - alias
SQLiteManager = HybridSQLiteManager