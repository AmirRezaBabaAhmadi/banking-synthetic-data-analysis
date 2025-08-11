"""
تحلیل clustering با پردازش async برای سرعت بالاتر
"""

import asyncio
import concurrent.futures
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .clustering import BankingCustomerClustering

logger = logging.getLogger(__name__)

class AsyncBankingCustomerClustering(BankingCustomerClustering):
    """تحلیل clustering async برای پردازش موازی"""
    
    def __init__(self, db_manager, max_workers: int = None):
        """
        مقداردهی اولیه
        
        Args:
            db_manager: مدیر دیتابیس
            max_workers: حداکثر تعداد worker ها
        """
        super().__init__(db_manager)
        import os
        self.max_workers = max_workers or min(8, (len(os.sched_getaffinity(0)) or 1) + 2)
        logger.info(f"AsyncClustering initialized with {self.max_workers} workers")
    
    async def run_clustering_method_async(self, method_name: str, 
                                        clustering_features: pd.DataFrame) -> Dict[str, Any]:
        """اجرای async یک روش clustering"""
        loop = asyncio.get_event_loop()
        
        logger.info(f"Starting async clustering with {method_name}")
        
        # انتخاب روش clustering
        method_map = {
            'kmeans': self.apply_kmeans_clustering,
            'dbscan': self.apply_dbscan_clustering,
            'gaussian_mixture': self.apply_gaussian_mixture_clustering,
            'hdbscan': self.apply_hdbscan_clustering
        }
        
        if method_name not in method_map:
            logger.error(f"Unknown clustering method: {method_name}")
            return {}
        
        # اجرای روش clustering در thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(method_map[method_name], clustering_features)
            result = await loop.run_in_executor(None, lambda: future.result())
        
        logger.info(f"Async clustering {method_name} completed")
        return result
    
    async def run_all_clustering_methods_async(self, 
                                             clustering_features: pd.DataFrame) -> Dict[str, Any]:
        """اجرای async تمام روش‌های clustering به صورت موازی"""
        
        logger.info("Starting async clustering for all methods...")
        
        clustering_methods = ['kmeans', 'dbscan', 'gaussian_mixture', 'hdbscan']
        
        # ایجاد tasks برای تمام روش‌ها
        tasks = []
        for method in clustering_methods:
            task = self.run_clustering_method_async(method, clustering_features)
            tasks.append((method, task))
        
        # اجرای موازی تمام روش‌ها
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], 
                                             return_exceptions=True)
        
        # پردازش نتایج
        for i, result in enumerate(completed_tasks):
            method_name = tasks[i][0]
            
            if isinstance(result, Exception):
                logger.error(f"Error in {method_name}: {result}")
                results[method_name] = {'error': str(result)}
            else:
                results[method_name] = result
        
        logger.info("Async clustering for all methods completed")
        return results
    
    async def evaluate_clustering_methods_async(self, 
                                              clustering_results: Dict[str, Any],
                                              clustering_features: pd.DataFrame) -> Dict[str, Any]:
        """ارزیابی async نتایج clustering"""
        
        logger.info("Starting async evaluation of clustering methods...")
        
        # ایجاد tasks برای ارزیابی
        evaluation_tasks = []
        
        for method_name, method_result in clustering_results.items():
            if 'error' not in method_result and 'labels' in method_result:
                task = self.evaluate_single_method_async(
                    method_name, method_result, clustering_features
                )
                evaluation_tasks.append((method_name, task))
        
        # اجرای موازی ارزیابی‌ها
        evaluation_results = {}
        if evaluation_tasks:
            completed_evaluations = await asyncio.gather(
                *[task for _, task in evaluation_tasks], 
                return_exceptions=True
            )
            
            # پردازش نتایج ارزیابی
            for i, eval_result in enumerate(completed_evaluations):
                method_name = evaluation_tasks[i][0]
                
                if isinstance(eval_result, Exception):
                    logger.error(f"Error evaluating {method_name}: {eval_result}")
                    evaluation_results[method_name] = {'error': str(eval_result)}
                else:
                    evaluation_results[method_name] = eval_result
        
        logger.info("Async evaluation completed")
        return evaluation_results
    
    async def evaluate_single_method_async(self, method_name: str, 
                                         method_result: Dict[str, Any],
                                         clustering_features: pd.DataFrame) -> Dict[str, Any]:
        """ارزیابی async یک روش clustering"""
        
        loop = asyncio.get_event_loop()
        
        # اجرای ارزیابی در thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.evaluate_clustering_quality, 
                clustering_features, 
                method_result['labels']
            )
            metrics = await loop.run_in_executor(None, lambda: future.result())
        
        # اضافه کردن اطلاعات روش
        result = {
            'method': method_name,
            'metrics': metrics,
            'n_clusters': len(np.unique(method_result['labels'])),
            'algorithm_params': method_result.get('params', {})
        }
        
        return result
    
    async def generate_cluster_profiles_async(self, 
                                            clustering_results: Dict[str, Any],
                                            features_df: pd.DataFrame) -> Dict[str, Any]:
        """تولید async پروفایل خوشه‌ها"""
        
        logger.info("Starting async cluster profile generation...")
        
        # ایجاد tasks برای تولید پروفایل
        profile_tasks = []
        
        for method_name, method_result in clustering_results.items():
            if 'error' not in method_result and 'labels' in method_result:
                task = self.generate_single_method_profiles_async(
                    method_name, method_result, features_df
                )
                profile_tasks.append((method_name, task))
        
        # اجرای موازی تولید پروفایل
        cluster_profiles = {}
        if profile_tasks:
            completed_profiles = await asyncio.gather(
                *[task for _, task in profile_tasks], 
                return_exceptions=True
            )
            
            # پردازش نتایج
            for i, profile_result in enumerate(completed_profiles):
                method_name = profile_tasks[i][0]
                
                if isinstance(profile_result, Exception):
                    logger.error(f"Error generating profiles for {method_name}: {profile_result}")
                    cluster_profiles[method_name] = {'error': str(profile_result)}
                else:
                    cluster_profiles[method_name] = profile_result
        
        logger.info("Async cluster profile generation completed")
        return cluster_profiles
    
    async def generate_single_method_profiles_async(self, method_name: str,
                                                  method_result: Dict[str, Any],
                                                  features_df: pd.DataFrame) -> Dict[str, Any]:
        """تولید async پروفایل برای یک روش"""
        
        loop = asyncio.get_event_loop()
        
        # اجرای تولید پروفایل در thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # اضافه کردن labels به features
            features_with_labels = features_df.copy()
            features_with_labels['cluster'] = method_result['labels']
            
            future = executor.submit(self.analyze_cluster_profiles, features_with_labels)
            profiles = await loop.run_in_executor(None, lambda: future.result())
        
        return profiles
    
    async def run_complete_clustering_analysis_async(self, 
                                                   sample_size: Optional[int] = 10000) -> Dict[str, Any]:
        """اجرای کامل async تحلیل clustering"""
        
        logger.info("Starting complete async clustering analysis...")
        start_time = datetime.now()
        
        # آماده‌سازی ویژگی‌ها
        features_df, clustering_features = self.prepare_features_for_clustering(sample_size)
        
        if features_df.empty:
            logger.error("No features available for clustering")
            return {}
        
        # اجرای async تمام روش‌های clustering
        clustering_results = await self.run_all_clustering_methods_async(clustering_features)
        
        # ارزیابی async روش‌ها
        evaluation_results = await self.evaluate_clustering_methods_async(
            clustering_results, clustering_features
        )
        
        # تولید async پروفایل خوشه‌ها
        cluster_profiles = await self.generate_cluster_profiles_async(
            clustering_results, features_df
        )
        
        # انتخاب بهترین روش
        best_method, best_score = self.select_best_clustering_method(evaluation_results)
        
        # ذخیره نتایج
        if best_method and best_method in clustering_results:
            await self.save_clustering_results_async(
                features_df, clustering_results[best_method]['labels'], best_method
            )
        
        # محاسبه زمان کل
        total_time = (datetime.now() - start_time).total_seconds()
        
        # خلاصه نتایج
        analysis_summary = {
            'clustering_results': clustering_results,
            'evaluation_results': evaluation_results,
            'cluster_profiles': cluster_profiles,
            'best_method': best_method,
            'best_score': best_score,
            'sample_size': len(features_df),
            'processing_time': total_time,
            'async_processing': True
        }
        
        logger.info(f"Complete async clustering analysis finished in {total_time:.2f} seconds")
        
        return analysis_summary
    
    async def save_clustering_results_async(self, features_df: pd.DataFrame, 
                                          labels: np.ndarray, method: str):
        """ذخیره async نتایج clustering"""
        
        loop = asyncio.get_event_loop()
        
        # اجرای ذخیره در thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.save_clustering_results, features_df, labels, method)
            await loop.run_in_executor(None, lambda: future.result())
        
        logger.info(f"Async clustering results saved for method: {method}")

# تابع کمکی برای اجرای async clustering
async def run_async_clustering_analysis(db_manager, sample_size: int = 10000, 
                                      max_workers: int = None) -> Dict[str, Any]:
    """
    اجرای کامل async clustering analysis
    
    Args:
        db_manager: مدیر دیتابیس
        sample_size: اندازه نمونه
        max_workers: حداکثر worker ها
    
    Returns:
        نتایج کامل تحلیل clustering
    """
    analyzer = AsyncBankingCustomerClustering(db_manager, max_workers)
    
    # اجرای تحلیل async
    results = await analyzer.run_complete_clustering_analysis_async(sample_size)
    
    return results 