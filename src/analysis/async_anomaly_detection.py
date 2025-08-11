"""
تشخیص anomaly با پردازش async برای سرعت بالاتر
"""

import asyncio
import concurrent.futures
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .anomaly_detection import BankingAnomalyDetector

logger = logging.getLogger(__name__)

class AsyncBankingAnomalyDetection(BankingAnomalyDetector):
    """تشخیص anomaly async برای پردازش موازی"""
    
    def __init__(self, db_manager, max_workers: int = None):
        """
        مقداردهی اولیه
        
        Args:
            db_manager: مدیر دیتابیس
            max_workers: حداکثر تعداد worker ها
        """
        super().__init__(db_manager)
        import os
        self.max_workers = max_workers or min(6, (len(os.sched_getaffinity(0)) or 1) + 2)
        logger.info(f"AsyncAnomalyDetection initialized with {self.max_workers} workers")
    
    async def run_anomaly_method_async(self, method_name: str, 
                                     features_df: pd.DataFrame) -> Dict[str, Any]:
        """اجرای async یک روش anomaly detection"""
        loop = asyncio.get_event_loop()
        
        logger.info(f"Starting async anomaly detection with {method_name}")
        
        # انتخاب روش anomaly detection
        method_map = {
            'isolation_forest': self.fit_isolation_forest,
            'one_class_svm': self.fit_one_class_svm,
            'supervised': self.fit_supervised_anomaly_detection
        }
        
        if method_name not in method_map:
            logger.error(f"Unknown anomaly detection method: {method_name}")
            return {}
        
        # اجرای روش در thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(method_map[method_name], features_df)
            result = await loop.run_in_executor(None, lambda: future.result())
        
        logger.info(f"Async anomaly detection {method_name} completed")
        return result
    
    async def run_all_anomaly_methods_async(self, 
                                          features_df: pd.DataFrame) -> Dict[str, Any]:
        """اجرای async تمام روش‌های anomaly detection به صورت موازی"""
        
        logger.info("Starting async anomaly detection for all methods...")
        
        anomaly_methods = ['isolation_forest', 'one_class_svm', 'supervised']
        
        # ایجاد tasks برای تمام روش‌ها
        tasks = []
        for method in anomaly_methods:
            task = self.run_anomaly_method_async(method, features_df)
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
        
        logger.info("Async anomaly detection for all methods completed")
        return results
    
    async def analyze_anomalies_async(self, 
                                    anomaly_results: Dict[str, Any],
                                    features_df: pd.DataFrame) -> Dict[str, Any]:
        """تحلیل async anomaly ها"""
        
        logger.info("Starting async anomaly analysis...")
        
        # ایجاد tasks برای تحلیل
        analysis_tasks = []
        
        for method_name, method_result in anomaly_results.items():
            if 'error' not in method_result and 'anomaly_scores' in method_result:
                task = self.analyze_single_method_anomalies_async(
                    method_name, method_result, features_df
                )
                analysis_tasks.append((method_name, task))
        
        # اجرای موازی تحلیل‌ها
        analysis_results = {}
        if analysis_tasks:
            completed_analyses = await asyncio.gather(
                *[task for _, task in analysis_tasks], 
                return_exceptions=True
            )
            
            # پردازش نتایج تحلیل
            for i, analysis_result in enumerate(completed_analyses):
                method_name = analysis_tasks[i][0]
                
                if isinstance(analysis_result, Exception):
                    logger.error(f"Error analyzing {method_name}: {analysis_result}")
                    analysis_results[method_name] = {'error': str(analysis_result)}
                else:
                    analysis_results[method_name] = analysis_result
        
        logger.info("Async anomaly analysis completed")
        return analysis_results
    
    async def analyze_single_method_anomalies_async(self, method_name: str,
                                                  method_result: Dict[str, Any],
                                                  features_df: pd.DataFrame) -> Dict[str, Any]:
        """تحلیل async anomaly های یک روش"""
        
        loop = asyncio.get_event_loop()
        
        # اجرای تحلیل در thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # آماده‌سازی داده‌ها
            anomaly_indices = method_result.get('anomaly_indices', [])
            anomaly_scores = method_result.get('anomaly_scores', [])
            
            # تابع تحلیل
            def analyze_method():
                analysis = self.analyze_anomaly_patterns(
                    features_df, anomaly_indices, anomaly_scores
                )
                
                # اضافه کردن آمار روش
                analysis.update({
                    'method': method_name,
                    'total_anomalies': len(anomaly_indices),
                    'anomaly_rate': len(anomaly_indices) / len(features_df) if len(features_df) > 0 else 0,
                    'mean_anomaly_score': np.mean(anomaly_scores) if anomaly_scores else 0
                })
                
                return analysis
            
            future = executor.submit(analyze_method)
            result = await loop.run_in_executor(None, lambda: future.result())
        
        return result
    
    async def generate_shap_explanations_async(self, 
                                             anomaly_results: Dict[str, Any],
                                             features_df: pd.DataFrame,
                                             max_samples: int = 100) -> Dict[str, Any]:
        """تولید async توضیحات SHAP"""
        
        logger.info("Starting async SHAP explanation generation...")
        
        # ایجاد tasks برای SHAP
        shap_tasks = []
        
        for method_name, method_result in anomaly_results.items():
            if ('error' not in method_result and 
                'anomaly_indices' in method_result and 
                len(method_result['anomaly_indices']) > 0):
                
                task = self.generate_single_method_shap_async(
                    method_name, method_result, features_df, max_samples
                )
                shap_tasks.append((method_name, task))
        
        # اجرای موازی SHAP
        shap_results = {}
        if shap_tasks:
            completed_shap = await asyncio.gather(
                *[task for _, task in shap_tasks], 
                return_exceptions=True
            )
            
            # پردازش نتایج SHAP
            for i, shap_result in enumerate(completed_shap):
                method_name = shap_tasks[i][0]
                
                if isinstance(shap_result, Exception):
                    logger.error(f"Error generating SHAP for {method_name}: {shap_result}")
                    shap_results[method_name] = {'error': str(shap_result)}
                else:
                    shap_results[method_name] = shap_result
        
        logger.info("Async SHAP explanation generation completed")
        return shap_results
    
    async def generate_single_method_shap_async(self, method_name: str,
                                              method_result: Dict[str, Any],
                                              features_df: pd.DataFrame,
                                              max_samples: int) -> Dict[str, Any]:
        """تولید async SHAP برای یک روش"""
        
        loop = asyncio.get_event_loop()
        
        # اجرای SHAP در thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # انتخاب نمونه anomaly ها
            anomaly_indices = method_result['anomaly_indices']
            selected_indices = anomaly_indices[:max_samples] if len(anomaly_indices) > max_samples else anomaly_indices
            
            future = executor.submit(
                self.explain_anomalies_with_shap, 
                features_df, selected_indices
            )
            shap_explanations = await loop.run_in_executor(None, lambda: future.result())
        
        return {
            'method': method_name,
            'explanations': shap_explanations,
            'samples_analyzed': len(selected_indices)
        }
    
    async def run_complete_anomaly_analysis_async(self, 
                                                sample_size: Optional[int] = 30000) -> Dict[str, Any]:
        """اجرای کامل async تحلیل anomaly detection"""
        
        logger.info("Starting complete async anomaly analysis...")
        start_time = datetime.now()
        
        # آماده‌سازی ویژگی‌ها
        features_df = self.prepare_features_for_anomaly_detection(sample_size)
        
        if features_df.empty:
            logger.error("No features available for anomaly detection")
            return {}
        
        # اجرای async تمام روش‌های anomaly detection
        anomaly_results = await self.run_all_anomaly_methods_async(features_df)
        
        # تحلیل async anomaly ها
        analysis_results = await self.analyze_anomalies_async(anomaly_results, features_df)
        
        # تولید async توضیحات SHAP
        shap_results = await self.generate_shap_explanations_async(
            anomaly_results, features_df, max_samples=50
        )
        
        # ذخیره بهترین نتایج
        best_method = self.select_best_anomaly_method(anomaly_results)
        if best_method and best_method in anomaly_results:
            await self.save_anomaly_results_async(
                features_df, anomaly_results[best_method], best_method
            )
        
        # محاسبه زمان کل
        total_time = (datetime.now() - start_time).total_seconds()
        
        # خلاصه نتایج
        analysis_summary = {
            'anomaly_results': anomaly_results,
            'analysis_results': analysis_results,
            'shap_results': shap_results,
            'best_method': best_method,
            'sample_size': len(features_df),
            'processing_time': total_time,
            'async_processing': True,
            'total_anomalies_detected': sum([
                len(result.get('anomaly_indices', [])) 
                for result in anomaly_results.values()
                if 'error' not in result
            ])
        }
        
        logger.info(f"Complete async anomaly analysis finished in {total_time:.2f} seconds")
        
        return analysis_summary
    
    async def save_anomaly_results_async(self, features_df: pd.DataFrame,
                                       anomaly_result: Dict[str, Any], method: str):
        """ذخیره async نتایج anomaly detection"""
        
        loop = asyncio.get_event_loop()
        
        # اجرای ذخیره در thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.save_anomaly_detection_results, 
                features_df, anomaly_result['anomaly_indices'], 
                anomaly_result['anomaly_scores'], method
            )
            await loop.run_in_executor(None, lambda: future.result())
        
        logger.info(f"Async anomaly results saved for method: {method}")

# تابع کمکی برای اجرای async anomaly detection
async def run_async_anomaly_analysis(db_manager, sample_size: int = 30000, 
                                   max_workers: int = None) -> Dict[str, Any]:
    """
    اجرای کامل async anomaly detection analysis
    
    Args:
        db_manager: مدیر دیتابیس
        sample_size: اندازه نمونه
        max_workers: حداکثر worker ها
    
    Returns:
        نتایج کامل تحلیل anomaly detection
    """
    analyzer = AsyncBankingAnomalyDetection(db_manager, max_workers)
    
    # اجرای تحلیل async
    results = await analyzer.run_complete_anomaly_analysis_async(sample_size)
    
    return results 