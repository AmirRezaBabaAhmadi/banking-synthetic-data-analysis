"""
استخراج ویژگی با پردازش async برای سرعت بالاتر
"""

import asyncio
import concurrent.futures
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

from .extractors import BankingFeatureExtractor

logger = logging.getLogger(__name__)

class AsyncBankingFeatureExtractor(BankingFeatureExtractor):
    """استخراج ویژگی async برای پردازش موازی"""
    
    def __init__(self, db_manager, max_workers: int = None):
        """
        مقداردهی اولیه
        
        Args:
            db_manager: مدیر دیتابیس
            max_workers: حداکثر تعداد worker ها برای parallel processing
        """
        super().__init__(db_manager)
        self.max_workers = max_workers or min(32, (len(os.sched_getaffinity(0)) or 1) + 4)
        logger.info(f"AsyncFeatureExtractor initialized with {self.max_workers} workers")
    
    async def extract_user_features_async(self, user_id: int) -> Dict[str, Any]:
        """استخراج ویژگی async برای یک کاربر"""
        loop = asyncio.get_event_loop()
        
        # اجرای extract_user_features در thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.extract_user_features, user_id)
            result = await loop.run_in_executor(None, lambda: future.result())
        
        return result
    
    async def extract_features_batch_async(self, user_ids: List[int], 
                                         batch_size: int = 100) -> pd.DataFrame:
        """
        استخراج ویژگی async برای مجموعه کاربران
        
        Args:
            user_ids: لیست شناسه کاربران
            batch_size: اندازه هر batch برای پردازش
        """
        logger.info(f"Starting async feature extraction for {len(user_ids)} users...")
        
        all_features = []
        
        # تقسیم به batch های کوچکتر
        for i in range(0, len(user_ids), batch_size):
            batch_user_ids = user_ids[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(user_ids) + batch_size - 1)//batch_size}")
            
            # پردازش موازی در هر batch
            batch_features = await self._process_batch_async(batch_user_ids)
            
            if batch_features:
                all_features.extend(batch_features)
        
        if not all_features:
            logger.warning("No features extracted")
            return pd.DataFrame()
        
        features_df = pd.DataFrame(all_features)
        
        # ذخیره آمار
        self.extraction_stats['feature_names'] = list(features_df.columns)
        self.extraction_stats['features_extracted'] = len(features_df.columns)
        self.extraction_stats['users_processed'] = len(features_df)
        
        logger.info(f"Async extraction completed: {len(features_df.columns)} features for {len(features_df)} users")
        
        return features_df
    
    async def _process_batch_async(self, user_ids: List[int]) -> List[Dict[str, Any]]:
        """پردازش موازی یک batch از کاربران"""
        
        # ایجاد tasks برای پردازش موازی
        tasks = []
        
        # محدود کردن تعداد concurrent tasks
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def extract_with_semaphore(user_id: int):
            async with semaphore:
                return await self.extract_user_features_async(user_id)
        
        # ایجاد tasks
        for user_id in user_ids:
            task = extract_with_semaphore(user_id)
            tasks.append(task)
        
        # اجرای موازی تمام tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # فیلتر کردن نتایج معتبر
        valid_features = []
        for result in results:
            if isinstance(result, dict) and result:
                valid_features.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Error in feature extraction: {result}")
        
        return valid_features
    
    async def extract_all_user_features_async(self, batch_size: int = 1000,
                                            concurrent_batches: int = 2) -> pd.DataFrame:
        """
        استخراج ویژگی async برای تمام کاربران
        
        Args:
            batch_size: اندازه هر batch
            concurrent_batches: تعداد batch های همزمان
        """
        logger.info("Starting async feature extraction for all users...")
        
        # دریافت تمام user IDs
        all_users = self.db_manager.execute_query("SELECT user_id FROM users ORDER BY user_id")
        all_user_ids = all_users['user_id'].tolist()
        
        logger.info(f"Found {len(all_user_ids)} users to process with async extraction")
        
        all_features = []
        
        # تقسیم به batch های بزرگتر برای پردازش concurrent
        big_batch_size = batch_size * concurrent_batches
        
        for i in range(0, len(all_user_ids), big_batch_size):
            big_batch_user_ids = all_user_ids[i:i + big_batch_size]
            
            logger.info(f"Processing big batch {i//big_batch_size + 1}")
            
            # تقسیم big batch به batch های کوچکتر
            batch_tasks = []
            for j in range(0, len(big_batch_user_ids), batch_size):
                batch_user_ids = big_batch_user_ids[j:j + batch_size]
                task = self.extract_features_batch_async(batch_user_ids, batch_size=100)
                batch_tasks.append(task)
            
            # اجرای موازی batch ها
            batch_results = await asyncio.gather(*batch_tasks)
            
            # ترکیب نتایج
            for batch_df in batch_results:
                if not batch_df.empty:
                    all_features.append(batch_df)
        
        # ترکیب تمام نتایج
        if all_features:
            final_features_df = pd.concat(all_features, ignore_index=True)
            logger.info(f"Async feature extraction completed: {len(final_features_df)} users")
            return final_features_df
        else:
            logger.warning("No features extracted")
            return pd.DataFrame()

# وظایف کمکی برای async processing
import os

async def run_async_feature_extraction(db_manager, max_workers: int = None) -> pd.DataFrame:
    """
    اجرای کامل async feature extraction
    
    Args:
        db_manager: مدیر دیتابیس
        max_workers: حداکثر worker ها
    
    Returns:
        DataFrame ویژگی‌های استخراج شده
    """
    extractor = AsyncBankingFeatureExtractor(db_manager, max_workers)
    
    start_time = datetime.now()
    
    # استخراج ویژگی async
    features_df = await extractor.extract_all_user_features_async(
        batch_size=1000,
        concurrent_batches=3
    )
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Async feature extraction completed in {processing_time:.2f} seconds")
    logger.info(f"Speed: {len(features_df)/processing_time:.1f} users/second")
    
    return features_df 