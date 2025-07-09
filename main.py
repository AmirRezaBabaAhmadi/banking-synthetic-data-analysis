#!/usr/bin/env python3
"""
فایل اصلی اجرای پروژه تولید و تحلیل داده‌های بانکی Synthetic

استفاده:
    python main.py --sample       # تولید داده نمونه (1000 کاربر)
    python main.py --full         # تولید dataset کامل (1 میلیون کاربر)
    python main.py --analysis     # انجام تحلیل‌ها روی داده موجود
    python main.py --all          # انجام همه مراحل
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# اضافه کردن مسیر src به Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generation.generators import BankingDataGenerator, NewUserGenerator
from src.database.sqlite_manager import SQLiteManager
from src.utils.config import get_config, setup_directories

# تنظیم logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('banking_data_generation.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def generate_sample_data():
    """تولید داده نمونه برای تست"""
    logger.info("=== شروع تولید داده نمونه ===")
    
    try:
        # ایجاد generator
        db_manager = SQLiteManager("database/banking_sample.db")
        generator = BankingDataGenerator(db_manager)
        
        # تولید 1000 کاربر نمونه
        stats = generator.generate_sample_data(num_users=1000)
        
        # صادرات گزارش
        generator.export_generation_report("output/reports/sample_generation_report.txt")
        
        # صادرات آمار نویز
        generator.noise_injector.export_noise_analysis("output/reports/sample_noise_analysis.txt")
        
        logger.info("تولید داده نمونه با موفقیت به پایان رسید")
        logger.info(f"تولید شد: {stats['total_users_generated']:,} کاربر و {stats['total_transactions_generated']:,} تراکنش")
        
        return stats
        
    except Exception as e:
        logger.error(f"خطا در تولید داده نمونه: {e}")
        raise
    finally:
        if 'generator' in locals():
            generator.close()

def generate_full_dataset():
    """تولید dataset کامل 1 میلیون کاربر"""
    logger.info("=== شروع تولید dataset کامل ===")
    
    try:
        # ایجاد generator
        db_manager = SQLiteManager()
        generator = BankingDataGenerator(db_manager)
        
        # تولید dataset کامل
        stats = generator.generate_complete_dataset()
        
        # صادرات گزارش‌ها
        generator.export_generation_report("output/reports/full_generation_report.txt")
        generator.noise_injector.export_noise_analysis("output/reports/full_noise_analysis.txt")
        
        logger.info("تولید dataset کامل با موفقیت به پایان رسید")
        logger.info(f"تولید شد: {stats['total_users_generated']:,} کاربر و {stats['total_transactions_generated']:,} تراکنش")
        
        return stats
        
    except Exception as e:
        logger.error(f"خطا در تولید dataset کامل: {e}")
        raise
    finally:
        if 'generator' in locals():
            generator.close()

def run_analysis():
    """انجام تحلیل‌های پیشرفته روی داده"""
    logger.info("=== شروع تحلیل‌های پیشرفته ===")
    
    try:
        from src.feature_engineering.extractors import BankingFeatureExtractor
        from src.feature_engineering.transformers import FeatureTransformer
        from src.analysis.clustering import BankingCustomerClustering
        from src.analysis.anomaly_detection import BankingAnomalyDetector
        from src.analysis.similarity_search import BankingSimilaritySearch
        from src.utils.visualization import BankingVisualizationUtils
        
        # مدیر دیتابیس
        db_manager = SQLiteManager()
        
        # بررسی وجود داده
        users_count = db_manager.execute_query("SELECT COUNT(*) as count FROM users").iloc[0]['count']
        if users_count == 0:
            logger.error("هیچ داده‌ای یافت نشد. لطفاً ابتدا تولید داده را اجرا کنید.")
            return
        
        logger.info(f"یافت شد: {users_count:,} کاربر برای تحلیل")
        
        # 1. Feature Engineering
        logger.info("1️⃣ استخراج ویژگی...")
        feature_extractor = BankingFeatureExtractor(db_manager)
        
        # استخراج ویژگی (نمونه کوچک برای سرعت)
        sample_size = min(10000, users_count)
        all_users = db_manager.execute_query("SELECT user_id FROM users ORDER BY RANDOM()")
        sample_user_ids = all_users['user_id'].head(sample_size).tolist()
        
        features_df = feature_extractor.extract_features_batch(sample_user_ids)
        if not features_df.empty:
            feature_extractor.save_features_to_database(features_df)
            logger.info(f"استخراج شد: {len(features_df.columns)} ویژگی برای {len(features_df)} کاربر")
        
        # 2. Clustering Analysis
        logger.info("2️⃣ تحلیل دسته‌بندی...")
        clustering_analyzer = BankingCustomerClustering(db_manager)
        clustering_results = clustering_analyzer.run_complete_clustering_analysis(sample_size=sample_size)
        logger.info(f"بهترین روش clustering: {clustering_results.get('best_method', 'هیچ')}")
        
        # 3. Anomaly Detection
        logger.info("3️⃣ تشخیص ناهنجاری...")
        anomaly_detector = BankingAnomalyDetector(db_manager)
        anomaly_results = anomaly_detector.run_complete_anomaly_analysis(sample_size=sample_size)
        logger.info(f"کشف شد: {anomaly_results.get('total_anomalies_detected', 0)} ناهنجاری")
        
        # 4. Similarity Search
        logger.info("4️⃣ جستجوی تشابه...")
        similarity_searcher = BankingSimilaritySearch(db_manager)
        similarity_results = similarity_searcher.run_complete_similarity_analysis()
        logger.info(f"تولید شد: {similarity_results.get('new_test_users', 0)} کاربر تست جدید")
        
        # 5. Comprehensive Visualization
        logger.info("5️⃣ ایجاد تصویرسازی‌های جامع...")
        visualizer = BankingVisualizationUtils()
        
        # داشبورد جامع
        transactions_sample = db_manager.execute_query(
            "SELECT * FROM transactions ORDER BY RANDOM() LIMIT 10000"
        )
        
        dashboard_fig = visualizer.create_comprehensive_dashboard(
            features_df, 
            transactions_sample
        )
        
        # خلاصه نتایج
        analysis_summary = {
            'feature_engineering': {
                'features_extracted': len(features_df.columns),
                'users_processed': len(features_df)
            },
            'clustering': {
                'best_method': clustering_results.get('best_method'),
                'best_score': clustering_results.get('best_score'),
                'clusters_found': len(clustering_results.get('cluster_profiles', {}))
            },
            'anomaly_detection': {
                'total_anomalies': anomaly_results.get('total_anomalies_detected', 0),
                'best_method': anomaly_results.get('best_method'),
                'detection_methods': list(anomaly_results.get('methods_results', {}).keys())
            },
            'similarity_search': {
                'new_test_users': similarity_results.get('new_test_users', 0),
                'total_users': similarity_results.get('total_users', 0),
                'avg_similarity': similarity_results.get('search_stats', {}).get('avg_similarity_score', 0)
            }
        }
        
        # ذخیره خلاصه
        summary_path = "output/analysis_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            import json
            import numpy as np
            
            # تبدیل numpy types به Python native types
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            analysis_summary_clean = convert_numpy_types(analysis_summary)
            json.dump(analysis_summary_clean, f, indent=2, ensure_ascii=False)
        
        logger.info("تحلیل‌ها با موفقیت تکمیل شد")
        logger.info(f"خلاصه تحلیل ذخیره شد در: {summary_path}")
        
        # نمایش خلاصه
        print("\n" + "="*60)
        print("📊 خلاصه نتایج تحلیل")
        print("="*60)
        print(f"🔧 ویژگی‌های استخراج‌شده: {analysis_summary['feature_engineering']['features_extracted']}")
        print(f"👥 کاربران پردازش‌شده: {analysis_summary['feature_engineering']['users_processed']:,}")
        print(f"🎯 بهترین روش clustering: {analysis_summary['clustering']['best_method']}")
        print(f"⚠️  ناهنجاری‌های کشف‌شده: {analysis_summary['anomaly_detection']['total_anomalies']:,}")
        print(f"🆕 کاربران تست جدید: {analysis_summary['similarity_search']['new_test_users']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"خطا در انجام تحلیل‌ها: {e}")
        raise

def main():
    """تابع اصلی"""
    parser = argparse.ArgumentParser(description='پروژه تولید و تحلیل داده‌های بانکی Synthetic')
    
    parser.add_argument('--sample', action='store_true', 
                       help='تولید داده نمونه (1000 کاربر)')
    parser.add_argument('--full', action='store_true', 
                       help='تولید dataset کامل (1 میلیون کاربر)')
    parser.add_argument('--analysis', action='store_true', 
                       help='انجام تحلیل‌ها روی داده موجود')
    parser.add_argument('--all', action='store_true', 
                       help='انجام همه مراحل')
    
    args = parser.parse_args()
    
    # بررسی آرگومان‌ها
    if not any([args.sample, args.full, args.analysis, args.all]):
        parser.print_help()
        return
    
    # آماده‌سازی پوشه‌ها
    logger.info("آماده‌سازی ساختار پروژه...")
    setup_directories()
    
    start_time = datetime.now()
    
    try:
        if args.sample or args.all:
            logger.info("🔄 شروع تولید داده نمونه...")
            sample_stats = generate_sample_data()
            logger.info("✅ تولید داده نمونه تکمیل شد")
        
        if args.full or args.all:
            logger.info("🔄 شروع تولید dataset کامل...")
            full_stats = generate_full_dataset()
            logger.info("✅ تولید dataset کامل تکمیل شد")
        
        if args.analysis or args.all:
            logger.info("🔄 شروع تحلیل‌های پیشرفته...")
            run_analysis()
            logger.info("✅ تحلیل‌ها تکمیل شد")
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"🎉 تمام مراحل در {total_time:.2f} ثانیه تکمیل شد")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  عملیات توسط کاربر متوقف شد")
    except Exception as e:
        logger.error(f"❌ خطای غیرمنتظره: {e}")
        raise

if __name__ == "__main__":
    main() 