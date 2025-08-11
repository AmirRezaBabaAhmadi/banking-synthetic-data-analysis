#!/usr/bin/env python3
"""
تست جامع سیستم قبل از اجرای 1 میلیون کاربر
"""

import sys
import os
import asyncio
import logging
import traceback
import importlib
import psutil
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# اضافه کردن مسیر src
sys.path.append(str(Path(__file__).parent / "src"))

# تنظیم logging برای تست
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('system_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveSystemTester:
    """تست جامع تمام اجزای سیستم"""
    
    def __init__(self):
        self.test_results = {}
        self.critical_errors = []
        self.warnings = []
        self.passed_tests = 0
        self.total_tests = 0
        
    def log_test_result(self, test_name: str, success: bool, message: str = "", 
                       is_critical: bool = False):
        """ثبت نتیجه تست"""
        self.total_tests += 1
        
        if success:
            self.passed_tests += 1
            logger.info(f"✅ {test_name}: {message}")
            self.test_results[test_name] = {'status': 'PASS', 'message': message}
        else:
            if is_critical:
                self.critical_errors.append(f"{test_name}: {message}")
                logger.error(f"❌ {test_name}: {message}")
                self.test_results[test_name] = {'status': 'CRITICAL_FAIL', 'message': message}
            else:
                self.warnings.append(f"{test_name}: {message}")
                logger.warning(f"⚠️ {test_name}: {message}")
                self.test_results[test_name] = {'status': 'WARNING', 'message': message}
    
    def test_system_requirements(self):
        """تست سیستم و hardware requirements"""
        logger.info("🔧 تست System Requirements...")
        
        try:
            # بررسی CPU
            cpu_count = psutil.cpu_count()
            self.log_test_result(
                "CPU Cores", 
                cpu_count >= 4, 
                f"CPU cores: {cpu_count} (مینیمم: 4, پیشنهادی: 8+)",
                is_critical=cpu_count < 2
            )
            
            # بررسی RAM
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            self.log_test_result(
                "RAM", 
                memory_gb >= 8, 
                f"RAM: {memory_gb:.1f}GB (مینیمم: 8GB, پیشنهادی: 16GB+)",
                is_critical=memory_gb < 4
            )
            
            # بررسی فضای دیسک
            disk = psutil.disk_usage('.')
            disk_free_gb = disk.free / (1024**3)
            self.log_test_result(
                "Disk Space", 
                disk_free_gb >= 20, 
                f"فضای خالی: {disk_free_gb:.1f}GB (مینیمم: 20GB)",
                is_critical=disk_free_gb < 10
            )
            
            # بررسی Python version
            python_version = sys.version_info
            version_ok = python_version.major == 3 and python_version.minor >= 8
            self.log_test_result(
                "Python Version", 
                version_ok, 
                f"Python {python_version.major}.{python_version.minor}.{python_version.micro}",
                is_critical=not version_ok
            )
            
        except Exception as e:
            self.log_test_result("System Requirements", False, f"خطا: {str(e)}", is_critical=True)
    
    def test_dependencies(self):
        """تست تمام dependencies"""
        logger.info("📦 تست Dependencies...")
        
        # لیست کامل dependencies
        critical_packages = [
            ('pandas', 'pandas'),
            ('numpy', 'numpy'), 
            ('sklearn', 'scikit-learn'),
            ('matplotlib', 'matplotlib'),
            ('seaborn', 'seaborn'),
            ('plotly', 'plotly'),
            ('shap', 'shap'),
            ('umap', 'umap-learn'),
            ('hdbscan', 'hdbscan'),
            ('statsmodels', 'statsmodels'),
            ('psutil', 'psutil'),
            ('tqdm', 'tqdm')
        ]
        
        optional_packages = [
            ('faiss', 'faiss-cpu'),
            ('hazm', 'hazm'),
            ('aiofiles', 'aiofiles'),
        ]
        
        # تست critical packages
        for import_name, package_name in critical_packages:
            try:
                importlib.import_module(import_name)
                self.log_test_result(f"Package: {package_name}", True, "موجود است")
            except ImportError:
                self.log_test_result(
                    f"Package: {package_name}", 
                    False, 
                    f"پکیج یافت نشد - نصب: pip install {package_name}",
                    is_critical=True
                )
        
        # تست optional packages
        for import_name, package_name in optional_packages:
            try:
                importlib.import_module(import_name)
                self.log_test_result(f"Optional: {package_name}", True, "موجود است")
            except ImportError:
                self.log_test_result(
                    f"Optional: {package_name}", 
                    False, 
                    f"پکیج اختیاری یافت نشد - نصب: pip install {package_name}",
                    is_critical=False
                )
    
    def test_directory_structure(self):
        """تست ساختار پوشه‌ها"""
        logger.info("📁 تست Directory Structure...")
        
        required_dirs = [
            'src',
            'src/data_generation',
            'src/database', 
            'src/feature_engineering',
            'src/analysis',
            'src/utils',
            'database',
            'output',
            'output/plots',
            'output/reports'
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists():
                self.log_test_result(f"Directory: {dir_path}", True, "موجود است")
            else:
                # ایجاد پوشه اگر وجود نداشته باشد
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    self.log_test_result(f"Directory: {dir_path}", True, "ایجاد شد")
                except Exception as e:
                    self.log_test_result(
                        f"Directory: {dir_path}", 
                        False, 
                        f"نمی‌تواند ایجاد شود: {str(e)}",
                        is_critical=True
                    )
        
        # بررسی permissions
        try:
            test_file = Path("output/test_write.tmp")
            test_file.write_text("test")
            test_file.unlink()
            self.log_test_result("Write Permissions", True, "دسترسی نوشتن OK")
        except Exception as e:
            self.log_test_result(
                "Write Permissions", 
                False, 
                f"مشکل دسترسی نوشتن: {str(e)}",
                is_critical=True
            )
    
    def test_module_imports(self):
        """تست import تمام ماژول‌های پروژه"""
        logger.info("🔧 تست Module Imports...")
        
        # ماژول‌های اصلی
        main_modules = [
            ('src.database.sqlite_manager', 'SQLiteManager'),
            ('src.data_generation.generators', 'BankingDataGenerator'),
            ('src.feature_engineering.extractors', 'BankingFeatureExtractor'),
            ('src.analysis.clustering', 'BankingCustomerClustering'),
            ('src.analysis.anomaly_detection', 'BankingAnomalyDetector'),
            ('src.analysis.similarity_search', 'BankingSimilaritySearch'),
            ('src.utils.visualization', 'BankingVisualizationUtils')
        ]
        
        # ماژول‌های async
        async_modules = [
            ('src.feature_engineering.async_extractors', 'AsyncBankingFeatureExtractor'),
            ('src.analysis.async_clustering', 'AsyncBankingCustomerClustering'),
            ('src.analysis.async_anomaly_detection', 'AsyncBankingAnomalyDetection')
        ]
        
        # تست ماژول‌های اصلی
        for module_path, class_name in main_modules:
            try:
                module = importlib.import_module(module_path)
                getattr(module, class_name)
                self.log_test_result(f"Import: {class_name}", True, "موفق")
            except Exception as e:
                self.log_test_result(
                    f"Import: {class_name}", 
                    False, 
                    f"خطا: {str(e)}",
                    is_critical=True
                )
        
        # تست ماژول‌های async
        for module_path, class_name in async_modules:
            try:
                module = importlib.import_module(module_path)
                getattr(module, class_name)
                self.log_test_result(f"Async Import: {class_name}", True, "موفق")
            except Exception as e:
                self.log_test_result(
                    f"Async Import: {class_name}", 
                    False, 
                    f"خطا: {str(e)}",
                    is_critical=True
                )
    
    def test_database_functionality(self):
        """تست عملکرد دیتابیس"""
        logger.info("🗄️ تست Database Functionality...")
        
        try:
            from src.database.sqlite_manager import SQLiteManager
            
            # تست ایجاد database
            test_db_path = "database/test_db.db"
            db_manager = SQLiteManager(test_db_path)
            
            # تست عملیات پایه
            test_query = "SELECT 1 as test_col"
            result = db_manager.execute_query(test_query)
            
            if not result.empty and result.iloc[0]['test_col'] == 1:
                self.log_test_result("Database Basic Query", True, "عملیات پایه OK")
            else:
                self.log_test_result("Database Basic Query", False, "خطا در query", is_critical=True)
            
            # تست ایجاد table
            create_table_query = """
            CREATE TABLE IF NOT EXISTS test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL
            )
            """
            db_manager.execute_query(create_table_query)
            
            # تست insert
            insert_query = "INSERT INTO test_table (name, value) VALUES ('test', 123.45)"
            try:
                db_manager.execute_update(insert_query)
                
                # تست select
                select_query = "SELECT * FROM test_table WHERE name = 'test'"
                result = db_manager.execute_query(select_query)
                
                if not result.empty and len(result) > 0:
                    self.log_test_result("Database CRUD Operations", True, "عملیات CRUD OK")
                else:
                    self.log_test_result("Database CRUD Operations", False, "داده‌ای یافت نشد", is_critical=True)
            except Exception as e:
                self.log_test_result("Database CRUD Operations", False, f"خطا در insert/select: {str(e)}", is_critical=True)
            
            db_manager.close()
            
            # حذف test database
            if Path(test_db_path).exists():
                Path(test_db_path).unlink()
                
        except Exception as e:
            self.log_test_result(
                "Database Functionality", 
                False, 
                f"خطا: {str(e)}",
                is_critical=True
            )
    
    async def test_async_functionality(self):
        """تست عملکرد async"""
        logger.info("🚀 تست Async Functionality...")
        
        try:
            # تست async basic
            await asyncio.sleep(0.1)
            self.log_test_result("Async Basic", True, "asyncio کار می‌کند")
            
            # تست concurrent.futures
            import concurrent.futures
            
            def test_cpu_task(x):
                return x * x
            
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future = executor.submit(test_cpu_task, 5)
                result = await loop.run_in_executor(None, lambda: future.result())
            
            if result == 25:
                self.log_test_result("Async ThreadPool", True, "ThreadPoolExecutor OK")
            else:
                self.log_test_result("Async ThreadPool", False, "خطا در ThreadPool", is_critical=True)
            
            # تست asyncio.gather
            async def async_task(delay, value):
                await asyncio.sleep(delay)
                return value * 2
            
            tasks = [
                async_task(0.1, 1),
                async_task(0.1, 2),
                async_task(0.1, 3)
            ]
            
            results = await asyncio.gather(*tasks)
            if results == [2, 4, 6]:
                self.log_test_result("Async Gather", True, "asyncio.gather OK")
            else:
                self.log_test_result("Async Gather", False, "خطا در gather", is_critical=True)
                
        except Exception as e:
            self.log_test_result(
                "Async Functionality", 
                False, 
                f"خطا: {str(e)}",
                is_critical=True
            )
    
    def test_sample_data_generation(self):
        """تست تولید داده نمونه"""
        logger.info("🔬 تست Sample Data Generation...")
        
        try:
            from src.data_generation.generators import BankingDataGenerator
            from src.database.sqlite_manager import SQLiteManager
            
            # ایجاد database تست
            test_db_path = "database/test_sample.db"
            db_manager = SQLiteManager(test_db_path)
            generator = BankingDataGenerator(db_manager)
            
            # تولید 10 کاربر نمونه
            logger.info("تولید 10 کاربر نمونه...")
            stats = generator.generate_sample_data(num_users=10)
            
            if stats['total_users_generated'] == 10 and stats['total_transactions_generated'] > 0:
                self.log_test_result(
                    "Sample Data Generation", 
                    True, 
                    f"10 کاربر و {stats['total_transactions_generated']} تراکنش تولید شد"
                )
            else:
                self.log_test_result(
                    "Sample Data Generation", 
                    False, 
                    "خطا در تولید داده نمونه",
                    is_critical=True
                )
            
            generator.close()
            
            # حذف test database
            if Path(test_db_path).exists():
                Path(test_db_path).unlink()
                
        except Exception as e:
            self.log_test_result(
                "Sample Data Generation", 
                False, 
                f"خطا: {str(e)}",
                is_critical=True
            )
    
    def test_feature_extraction_sample(self):
        """تست استخراج ویژگی روی نمونه"""
        logger.info("🧬 تست Sample Feature Extraction...")
        
        try:
            # ایجاد database با داده نمونه
            from src.data_generation.generators import BankingDataGenerator
            from src.database.sqlite_manager import SQLiteManager
            from src.feature_engineering.extractors import BankingFeatureExtractor
            
            test_db_path = "database/test_features.db"
            db_manager = SQLiteManager(test_db_path)
            
            # تولید 5 کاربر
            generator = BankingDataGenerator(db_manager)
            generator.generate_sample_data(num_users=5)
            generator.close()
            
            # استخراج ویژگی
            extractor = BankingFeatureExtractor(db_manager)
            users = db_manager.execute_query("SELECT user_id FROM users LIMIT 5")
            
            if not users.empty:
                user_ids = users['user_id'].tolist()
                features_df = extractor.extract_features_batch(user_ids)
                
                if not features_df.empty and len(features_df.columns) > 20:
                    self.log_test_result(
                        "Feature Extraction", 
                        True, 
                        f"{len(features_df.columns)} ویژگی برای {len(features_df)} کاربر"
                    )
                else:
                    self.log_test_result(
                        "Feature Extraction", 
                        False, 
                        "خطا در استخراج ویژگی",
                        is_critical=True
                    )
            else:
                self.log_test_result(
                    "Feature Extraction", 
                    False, 
                    "کاربر برای تست یافت نشد",
                    is_critical=True
                )
            
            db_manager.close()
            
            # حذف test database
            if Path(test_db_path).exists():
                Path(test_db_path).unlink()
                
        except Exception as e:
            self.log_test_result(
                "Feature Extraction", 
                False, 
                f"خطا: {str(e)}",
                is_critical=True
            )
    
    async def test_async_modules_functionality(self):
        """تست عملکرد ماژول‌های async"""
        logger.info("⚡ تست Async Modules Functionality...")
        
        try:
            # ایجاد database با داده نمونه
            from src.data_generation.generators import BankingDataGenerator
            from src.database.sqlite_manager import SQLiteManager
            from src.feature_engineering.async_extractors import AsyncBankingFeatureExtractor
            
            test_db_path = "database/test_async.db"
            db_manager = SQLiteManager(test_db_path)
            
            # تولید 3 کاربر
            generator = BankingDataGenerator(db_manager)
            generator.generate_sample_data(num_users=3)
            generator.close()
            
            # تست async feature extraction
            async_extractor = AsyncBankingFeatureExtractor(db_manager, max_workers=2)
            users = db_manager.execute_query("SELECT user_id FROM users LIMIT 3")
            
            if not users.empty:
                user_ids = users['user_id'].tolist()
                features_df = await async_extractor.extract_features_batch_async(user_ids)
                
                if not features_df.empty:
                    self.log_test_result(
                        "Async Feature Extraction", 
                        True, 
                        f"async استخراج {len(features_df.columns)} ویژگی"
                    )
                else:
                    self.log_test_result(
                        "Async Feature Extraction", 
                        False, 
                        "خطا در async feature extraction",
                        is_critical=True
                    )
            
            db_manager.close()
            
            # حذف test database
            if Path(test_db_path).exists():
                Path(test_db_path).unlink()
                
        except Exception as e:
            self.log_test_result(
                "Async Modules", 
                False, 
                f"خطا: {str(e)}",
                is_critical=True
            )
    
    def generate_test_report(self):
        """تولید گزارش تست"""
        logger.info("📊 تولید گزارش تست...")
        
        # محاسبه آمار
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        # ایجاد گزارش
        report = f"""
{'='*70}
🧪 گزارش تست جامع سیستم
{'='*70}

📊 آمار کلی:
├── کل تست‌ها: {self.total_tests}
├── تست‌های موفق: {self.passed_tests}
├── نرخ موفقیت: {success_rate:.1f}%
├── خطاهای critical: {len(self.critical_errors)}
└── هشدارها: {len(self.warnings)}

"""
        
        if self.critical_errors:
            report += f"""
❌ خطاهای Critical:
"""
            for error in self.critical_errors:
                report += f"   • {error}\n"
        
        if self.warnings:
            report += f"""
⚠️ هشدارها:
"""
            for warning in self.warnings:
                report += f"   • {warning}\n"
        
        # نتیجه‌گیری
        if not self.critical_errors:
            report += f"""

✅ نتیجه‌گیری: سیستم آماده اجرای 1 میلیون کاربر است!

🚀 مراحل بعدی:
1. python main.py --full         # تولید 1M کاربر
2. python main.py --analysis     # تحلیل async
3. python generate_analysis_report.py  # گزارش جامع

{'='*70}
"""
        else:
            report += f"""

❌ نتیجه‌گیری: سیستم آماده نیست - لطفاً خطاهای critical را برطرف کنید

💡 راهکارها:
1. pip install -r requirements.txt  # نصب dependencies
2. بررسی Python version >= 3.8
3. بررسی فضای دیسک و RAM
4. بررسی permissions پوشه‌ها

{'='*70}
"""
        
        print(report)
        
        # ذخیره گزارش
        with open("output/reports/system_test_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        return not bool(self.critical_errors)
    
    async def run_all_tests(self):
        """اجرای تمام تست‌ها"""
        logger.info("🚀 شروع تست جامع سیستم...")
        
        start_time = datetime.now()
        
        try:
            # تست‌های sync
            self.test_system_requirements()
            self.test_dependencies()
            self.test_directory_structure()
            self.test_module_imports()
            self.test_database_functionality()
            await self.test_async_functionality()
            self.test_sample_data_generation()
            self.test_feature_extraction_sample()
            
            # تست‌های async
            await self.test_async_modules_functionality()
            
        except Exception as e:
            logger.error(f"خطا در اجرای تست‌ها: {e}")
            self.log_test_result("Overall Test Execution", False, str(e), is_critical=True)
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"تست‌ها در {total_time:.2f} ثانیه تکمیل شد")
        
        # تولید گزارش
        return self.generate_test_report()

async def main():
    """تابع اصلی"""
    print("🧪 شروع تست جامع سیستم...")
    
    tester = ComprehensiveSystemTester()
    system_ready = await tester.run_all_tests()
    
    return system_ready

if __name__ == "__main__":
    asyncio.run(main()) 