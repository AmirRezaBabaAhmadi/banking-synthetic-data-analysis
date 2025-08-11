#!/usr/bin/env python3
"""
تست performance مقایسه‌ای Sync vs Async
"""

import sys
import time
import asyncio
import logging
import psutil
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# اضافه کردن مسیر src
sys.path.append(str(Path(__file__).parent / "src"))

from src.database.sqlite_manager import SQLiteManager
from src.feature_engineering.extractors import BankingFeatureExtractor
from src.feature_engineering.async_extractors import AsyncBankingFeatureExtractor
from src.analysis.clustering import BankingCustomerClustering  
from src.analysis.async_clustering import AsyncBankingCustomerClustering
from src.analysis.anomaly_detection import BankingAnomalyDetector
from src.analysis.async_anomaly_detection import AsyncBankingAnomalyDetection

# تنظیم logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AsyncPerformanceTester:
    """تست performance async vs sync"""
    
    def __init__(self, db_path: str = "database/banking_data.db"):
        self.db_path = db_path
        self.results = {}
        
    def monitor_resources(self, process_name: str):
        """مانیتورینگ منابع سیستم"""
        process = psutil.Process()
        
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'process_name': process_name,
            'timestamp': datetime.now()
        }
    
    def test_feature_extraction_sync(self, sample_size: int = 1000):
        """تست sync feature extraction"""
        logger.info(f"🔄 تست Sync Feature Extraction ({sample_size:,} users)...")
        
        start_time = time.time()
        start_resources = self.monitor_resources("sync_feature_extraction")
        
        try:
            db_manager = SQLiteManager(self.db_path)
            extractor = BankingFeatureExtractor(db_manager)
            
            # دریافت sample users
            users_sample = db_manager.execute_query(
                f"SELECT user_id FROM users ORDER BY user_id LIMIT {sample_size}"
            )
            user_ids = users_sample['user_id'].tolist()
            
            # استخراج ویژگی sync
            features_df = extractor.extract_features_batch(user_ids, batch_size=100)
            
            processing_time = time.time() - start_time
            end_resources = self.monitor_resources("sync_feature_extraction")
            
            result = {
                'processing_time': processing_time,
                'users_processed': len(features_df),
                'features_extracted': len(features_df.columns) if not features_df.empty else 0,
                'speed_users_per_sec': len(features_df) / processing_time if processing_time > 0 else 0,
                'start_resources': start_resources,
                'end_resources': end_resources,
                'memory_peak_mb': end_resources['memory_mb'] - start_resources['memory_mb']
            }
            
            logger.info(f"✅ Sync Feature Extraction: {processing_time:.2f}s, {result['speed_users_per_sec']:.1f} users/sec")
            
            db_manager.close()
            return result
            
        except Exception as e:
            logger.error(f"❌ خطا در sync feature extraction: {e}")
            return {'error': str(e)}
    
    async def test_feature_extraction_async(self, sample_size: int = 1000):
        """تست async feature extraction"""
        logger.info(f"🚀 تست Async Feature Extraction ({sample_size:,} users)...")
        
        start_time = time.time()
        start_resources = self.monitor_resources("async_feature_extraction")
        
        try:
            db_manager = SQLiteManager(self.db_path)
            extractor = AsyncBankingFeatureExtractor(db_manager, max_workers=8)
            
            # دریافت sample users
            users_sample = db_manager.execute_query(
                f"SELECT user_id FROM users ORDER BY user_id LIMIT {sample_size}"
            )
            user_ids = users_sample['user_id'].tolist()
            
            # استخراج ویژگی async
            features_df = await extractor.extract_features_batch_async(user_ids, batch_size=100)
            
            processing_time = time.time() - start_time
            end_resources = self.monitor_resources("async_feature_extraction")
            
            result = {
                'processing_time': processing_time,
                'users_processed': len(features_df),
                'features_extracted': len(features_df.columns) if not features_df.empty else 0,
                'speed_users_per_sec': len(features_df) / processing_time if processing_time > 0 else 0,
                'start_resources': start_resources,
                'end_resources': end_resources,
                'memory_peak_mb': end_resources['memory_mb'] - start_resources['memory_mb']
            }
            
            logger.info(f"✅ Async Feature Extraction: {processing_time:.2f}s, {result['speed_users_per_sec']:.1f} users/sec")
            
            db_manager.close()
            return result
            
        except Exception as e:
            logger.error(f"❌ خطا در async feature extraction: {e}")
            return {'error': str(e)}
    
    def test_clustering_sync(self, sample_size: int = 5000):
        """تست sync clustering"""
        logger.info(f"🔄 تست Sync Clustering ({sample_size:,} samples)...")
        
        start_time = time.time()
        start_resources = self.monitor_resources("sync_clustering")
        
        try:
            db_manager = SQLiteManager(self.db_path)
            analyzer = BankingCustomerClustering(db_manager)
            
            # اجرای clustering
            results = analyzer.run_complete_clustering_analysis(sample_size=sample_size)
            
            processing_time = time.time() - start_time
            end_resources = self.monitor_resources("sync_clustering")
            
            result = {
                'processing_time': processing_time,
                'samples_processed': sample_size,
                'best_method': results.get('best_method', 'Unknown'),
                'best_score': results.get('best_score', 0),
                'speed_samples_per_sec': sample_size / processing_time if processing_time > 0 else 0,
                'start_resources': start_resources,
                'end_resources': end_resources,
                'memory_peak_mb': end_resources['memory_mb'] - start_resources['memory_mb']
            }
            
            logger.info(f"✅ Sync Clustering: {processing_time:.2f}s, {result['speed_samples_per_sec']:.1f} samples/sec")
            
            db_manager.close()
            return result
            
        except Exception as e:
            logger.error(f"❌ خطا در sync clustering: {e}")
            return {'error': str(e)}
    
    async def test_clustering_async(self, sample_size: int = 5000):
        """تست async clustering"""
        logger.info(f"🚀 تست Async Clustering ({sample_size:,} samples)...")
        
        start_time = time.time()
        start_resources = self.monitor_resources("async_clustering")
        
        try:
            from src.analysis.async_clustering import run_async_clustering_analysis
            
            db_manager = SQLiteManager(self.db_path)
            
            # اجرای async clustering
            results = await run_async_clustering_analysis(db_manager, sample_size, max_workers=4)
            
            processing_time = time.time() - start_time
            end_resources = self.monitor_resources("async_clustering")
            
            result = {
                'processing_time': processing_time,
                'samples_processed': sample_size,
                'best_method': results.get('best_method', 'Unknown'),
                'best_score': results.get('best_score', 0),
                'speed_samples_per_sec': sample_size / processing_time if processing_time > 0 else 0,
                'start_resources': start_resources,
                'end_resources': end_resources,
                'memory_peak_mb': end_resources['memory_mb'] - start_resources['memory_mb']
            }
            
            logger.info(f"✅ Async Clustering: {processing_time:.2f}s, {result['speed_samples_per_sec']:.1f} samples/sec")
            
            db_manager.close()
            return result
            
        except Exception as e:
            logger.error(f"❌ خطا در async clustering: {e}")
            return {'error': str(e)}
    
    def test_anomaly_detection_sync(self, sample_size: int = 3000):
        """تست sync anomaly detection"""
        logger.info(f"🔄 تست Sync Anomaly Detection ({sample_size:,} samples)...")
        
        start_time = time.time()
        start_resources = self.monitor_resources("sync_anomaly")
        
        try:
            db_manager = SQLiteManager(self.db_path)
            detector = BankingAnomalyDetector(db_manager)
            
            # اجرای anomaly detection
            results = detector.run_complete_anomaly_analysis(sample_size=sample_size)
            
            processing_time = time.time() - start_time
            end_resources = self.monitor_resources("sync_anomaly")
            
            result = {
                'processing_time': processing_time,
                'samples_processed': sample_size,
                'total_anomalies': results.get('total_anomalies_detected', 0),
                'best_method': results.get('best_method', 'Unknown'),
                'speed_samples_per_sec': sample_size / processing_time if processing_time > 0 else 0,
                'start_resources': start_resources,
                'end_resources': end_resources,
                'memory_peak_mb': end_resources['memory_mb'] - start_resources['memory_mb']
            }
            
            logger.info(f"✅ Sync Anomaly Detection: {processing_time:.2f}s, {result['speed_samples_per_sec']:.1f} samples/sec")
            
            db_manager.close()
            return result
            
        except Exception as e:
            logger.error(f"❌ خطا در sync anomaly detection: {e}")
            return {'error': str(e)}
    
    async def test_anomaly_detection_async(self, sample_size: int = 3000):
        """تست async anomaly detection"""
        logger.info(f"🚀 تست Async Anomaly Detection ({sample_size:,} samples)...")
        
        start_time = time.time()
        start_resources = self.monitor_resources("async_anomaly")
        
        try:
            from src.analysis.async_anomaly_detection import run_async_anomaly_analysis
            
            db_manager = SQLiteManager(self.db_path)
            
            # اجرای async anomaly detection
            results = await run_async_anomaly_analysis(db_manager, sample_size, max_workers=4)
            
            processing_time = time.time() - start_time
            end_resources = self.monitor_resources("async_anomaly")
            
            result = {
                'processing_time': processing_time,
                'samples_processed': sample_size,
                'total_anomalies': results.get('total_anomalies_detected', 0),
                'best_method': results.get('best_method', 'Unknown'),
                'speed_samples_per_sec': sample_size / processing_time if processing_time > 0 else 0,
                'start_resources': start_resources,
                'end_resources': end_resources,
                'memory_peak_mb': end_resources['memory_mb'] - start_resources['memory_mb']
            }
            
            logger.info(f"✅ Async Anomaly Detection: {processing_time:.2f}s, {result['speed_samples_per_sec']:.1f} samples/sec")
            
            db_manager.close()
            return result
            
        except Exception as e:
            logger.error(f"❌ خطا در async anomaly detection: {e}")
            return {'error': str(e)}
    
    async def run_comprehensive_performance_test(self):
        """تست جامع performance"""
        logger.info("🚀 شروع تست جامع Performance...")
        
        overall_start = time.time()
        
        # تست‌های مختلف
        test_configs = [
            # (process, sample_size)
            ('feature_extraction', 1000),
            ('clustering', 5000),
            ('anomaly_detection', 3000)
        ]
        
        for process, sample_size in test_configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔬 تست {process.upper()} (Sample: {sample_size:,})")
            logger.info(f"{'='*60}")
            
            # تست sync
            sync_method = getattr(self, f'test_{process}_sync')
            sync_result = sync_method(sample_size)
            
            # تست async
            async_method = getattr(self, f'test_{process}_async')
            async_result = await async_method(sample_size)
            
            # محاسبه بهبود
            if 'error' not in sync_result and 'error' not in async_result:
                improvement = sync_result['processing_time'] / async_result['processing_time']
                
                self.results[process] = {
                    'sync': sync_result,
                    'async': async_result,
                    'improvement_factor': improvement,
                    'time_saved_seconds': sync_result['processing_time'] - async_result['processing_time'],
                    'speed_improvement_percent': ((improvement - 1) * 100)
                }
                
                logger.info(f"📊 بهبود {process}: {improvement:.1f}x سریع‌تر")
                logger.info(f"⏱️  صرفه‌جویی زمان: {self.results[process]['time_saved_seconds']:.1f} ثانیه")
            else:
                self.results[process] = {
                    'sync': sync_result,
                    'async': async_result,
                    'error': True
                }
        
        total_time = time.time() - overall_start
        
        logger.info(f"\n🎉 تست جامع در {total_time:.2f} ثانیه تکمیل شد")
        
        return self.results
    
    def generate_performance_report(self):
        """تولید گزارش performance"""
        logger.info("📊 تولید گزارش performance...")
        
        try:
            # ایجاد DataFrame
            report_data = []
            
            for process, data in self.results.items():
                if 'error' not in data:
                    report_data.append({
                        'Process': process.replace('_', ' ').title(),
                        'Sync Time (s)': round(data['sync']['processing_time'], 2),
                        'Async Time (s)': round(data['async']['processing_time'], 2),
                        'Improvement (x)': round(data['improvement_factor'], 1),
                        'Time Saved (s)': round(data['time_saved_seconds'], 1),
                        'Speed Improvement (%)': round(data['speed_improvement_percent'], 1),
                        'Sync Memory (MB)': round(data['sync']['memory_peak_mb'], 1),
                        'Async Memory (MB)': round(data['async']['memory_peak_mb'], 1)
                    })
            
            if report_data:
                df = pd.DataFrame(report_data)
                
                # نمایش جدول
                print("\n" + "="*100)
                print("📊 گزارش Performance Sync vs Async")
                print("="*100)
                print(df.to_string(index=False))
                print("="*100)
                
                # ذخیره CSV
                Path("output/reports").mkdir(parents=True, exist_ok=True)
                df.to_csv("output/reports/performance_comparison.csv", index=False)
                
                # خلاصه کلی
                total_sync_time = df['Sync Time (s)'].sum()
                total_async_time = df['Async Time (s)'].sum()
                overall_improvement = total_sync_time / total_async_time
                
                print(f"\n🎯 خلاصه کلی:")
                print(f"   • کل زمان Sync: {total_sync_time:.1f} ثانیه")
                print(f"   • کل زمان Async: {total_async_time:.1f} ثانیه")
                print(f"   • بهبود کلی: {overall_improvement:.1f}x")
                print(f"   • صرفه‌جویی کل: {total_sync_time - total_async_time:.1f} ثانیه")
                
                return {
                    'report_df': df,
                    'overall_improvement': overall_improvement,
                    'total_time_saved': total_sync_time - total_async_time
                }
            else:
                logger.warning("هیچ داده معتبری برای گزارش یافت نشد")
                return None
                
        except Exception as e:
            logger.error(f"خطا در تولید گزارش: {e}")
            return None

async def main():
    """تابع اصلی"""
    logger.info("شروع تست Performance Async vs Sync...")
    
    try:
        # ایجاد tester
        tester = AsyncPerformanceTester()
        
        # اجرای تست‌ها
        results = await tester.run_comprehensive_performance_test()
        
        # تولید گزارش
        report = tester.generate_performance_report()
        
        if report:
            print(f"\n✅ تست performance با موفقیت تکمیل شد")
            print(f"📁 گزارش ذخیره شد: output/reports/performance_comparison.csv")
        else:
            print("❌ خطا در تولید گزارش")
            
    except Exception as e:
        logger.error(f"خطای کلی: {e}")
        print(f"❌ خطای کلی: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 