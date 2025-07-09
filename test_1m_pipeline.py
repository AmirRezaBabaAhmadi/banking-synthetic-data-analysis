#!/usr/bin/env python3
"""
تست عملکرد pipeline برای 1 میلیون داده
"""

import sys
import os
import time
import psutil
from datetime import datetime

# اضافه کردن مسیر src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.database.sqlite_manager import SQLiteManager
from src.data_generation.generators import BankingDataGenerator
from src.feature_engineering.extractors import BankingFeatureExtractor

def monitor_resources():
    """نظارت بر منابع سیستم"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent()
    
    return {
        'memory_mb': memory_mb,
        'cpu_percent': cpu_percent,
        'available_memory_gb': psutil.virtual_memory().available / (1024**3)
    }

def test_1m_data_generation():
    """تست تولید 1 میلیون داده"""
    print("🧪 تست تولید 1 میلیون داده...")
    
    start_time = time.time()
    initial_resources = monitor_resources()
    
    try:
        # ایجاد دیتابیس مخصوص تست
        db_manager = SQLiteManager("database/test_1m_banking.db")
        generator = BankingDataGenerator(db_manager)
        
        print(f"📊 شروع تولید در {datetime.now()}")
        print(f"💾 حافظه ابتدایی: {initial_resources['memory_mb']:.1f} MB")
        print(f"💻 حافظه آزاد: {initial_resources['available_memory_gb']:.1f} GB")
        
        # تولید dataset کامل
        stats = generator.generate_complete_dataset()
        
        # بررسی منابع پس از تولید
        final_resources = monitor_resources()
        generation_time = time.time() - start_time
        
        print("\n" + "="*50)
        print("✅ نتایج تولید داده:")
        print("="*50)
        print(f"👥 کاربران تولید شده: {stats['total_users_generated']:,}")
        print(f"💳 تراکنش‌های تولید شده: {stats['total_transactions_generated']:,}")
        print(f"⏱️  زمان تولید: {generation_time:.1f} ثانیه ({generation_time/60:.1f} دقیقه)")
        print(f"📈 سرعت: {stats['total_transactions_generated']/generation_time:.0f} تراکنش/ثانیه")
        print(f"💾 حافظه استفاده شده: {final_resources['memory_mb'] - initial_resources['memory_mb']:.1f} MB")
        print(f"💽 اندازه دیتابیس: {stats['database_stats']['database_size_mb']:.1f} MB")
        
        generator.close()
        return True
        
    except Exception as e:
        print(f"❌ خطا در تولید داده: {e}")
        return False

def test_feature_extraction_performance():
    """تست عملکرد استخراج ویژگی"""
    print("\n🧪 تست استخراج ویژگی برای 1M کاربر...")
    
    try:
        db_manager = SQLiteManager("database/test_1m_banking.db")
        
        # بررسی تعداد کاربران
        users_count = db_manager.execute_query("SELECT COUNT(*) as count FROM users").iloc[0]['count']
        print(f"📊 کاربران موجود: {users_count:,}")
        
        if users_count == 0:
            print("❌ هیچ داده‌ای یافت نشد. ابتدا تولید داده را اجرا کنید.")
            return False
        
        feature_extractor = BankingFeatureExtractor(db_manager)
        
        # تست با batch های مختلف
        batch_sizes = [500, 1000, 2000]
        
        for batch_size in batch_sizes:
            print(f"\n🔬 تست با batch_size = {batch_size}")
            
            start_time = time.time()
            start_resources = monitor_resources()
            
            # استخراج ویژگی برای نمونه کوچک
            sample_size = min(5000, users_count)
            all_users = db_manager.execute_query("SELECT user_id FROM users ORDER BY RANDOM()")
            sample_user_ids = all_users['user_id'].head(sample_size).tolist()
            
            features_df = feature_extractor.extract_features_batch(sample_user_ids)
            
            extraction_time = time.time() - start_time
            end_resources = monitor_resources()
            
            print(f"  ⏱️  زمان: {extraction_time:.1f}s برای {sample_size:,} کاربر")
            print(f"  📈 سرعت: {sample_size/extraction_time:.1f} کاربر/ثانیه")
            print(f"  💾 حافظه: {end_resources['memory_mb'] - start_resources['memory_mb']:.1f} MB")
            print(f"  🔧 ویژگی‌ها: {len(features_df.columns)} ویژگی")
            
            # تخمین زمان برای 1M
            estimated_time_1m = (users_count / sample_size) * extraction_time
            print(f"  🕐 تخمین زمان 1M: {estimated_time_1m/60:.1f} دقیقه")
        
        db_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ خطا در تست استخراج ویژگی: {e}")
        return False

def test_analysis_scalability():
    """تست مقیاس‌پذیری تحلیل‌ها"""
    print("\n🧪 تست مقیاس‌پذیری تحلیل‌ها...")
    
    try:
        from src.analysis.clustering import BankingCustomerClustering
        from src.analysis.anomaly_detection import BankingAnomalyDetector
        
        db_manager = SQLiteManager("database/test_1m_banking.db")
        
        # تست clustering با sample های مختلف
        sample_sizes = [1000, 5000, 10000, 25000, 50000]
        
        clustering_analyzer = BankingCustomerClustering(db_manager)
        
        print("\n📊 تست Clustering:")
        for sample_size in sample_sizes:
            start_time = time.time()
            start_resources = monitor_resources()
            
            try:
                results = clustering_analyzer.run_complete_clustering_analysis(sample_size=sample_size)
                
                analysis_time = time.time() - start_time
                end_resources = monitor_resources()
                
                print(f"  Sample {sample_size:,}: {analysis_time:.1f}s, "
                      f"Memory: {end_resources['memory_mb'] - start_resources['memory_mb']:.1f}MB, "
                      f"Best: {results.get('best_method', 'N/A')}")
                
            except Exception as e:
                print(f"  Sample {sample_size:,}: ❌ {str(e)[:50]}...")
        
        db_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ خطا در تست تحلیل: {e}")
        return False

def main():
    """اجرای تست‌های کامل"""
    print("🚀 شروع تست pipeline برای 1 میلیون داده")
    print("="*60)
    
    # نمایش مشخصات سیستم
    system_info = monitor_resources()
    print(f"💻 مشخصات سیستم:")
    print(f"   حافظه آزاد: {system_info['available_memory_gb']:.1f} GB")
    print(f"   CPU: {psutil.cpu_count()} cores")
    print()
    
    # اجرای تست‌ها
    tests = [
        ("تولید 1M داده", test_1m_data_generation),
        ("استخراج ویژگی", test_feature_extraction_performance),
        ("مقیاس‌پذیری تحلیل", test_analysis_scalability)
    ]
    
    results = {}
    total_start = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = "✅ موفق" if success else "❌ ناموفق"
        except Exception as e:
            results[test_name] = f"❌ خطا: {str(e)[:30]}..."
    
    total_time = time.time() - total_start
    
    # نمایش خلاصه نتایج
    print("\n" + "="*60)
    print("📋 خلاصه نتایج تست‌ها:")
    print("="*60)
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    print(f"\n⏱️  کل زمان تست: {total_time/60:.1f} دقیقه")
    print("="*60)

if __name__ == "__main__":
    main() 