#!/usr/bin/env python3
"""
تخمین اندازه نهایی دیتاست بر اساس پیشرفت فعلی
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.database.sqlite_manager import SQLiteManager

def estimate_final_size():
    """تخمین اندازه نهایی دیتاست"""
    
    print("📊 تخمین اندازه نهایی دیتاست")
    print("="*50)
    
    # وضعیت فعلی
    db_path = "database/banking_data.db"
    wal_path = f"{db_path}-wal"
    shm_path = f"{db_path}-shm"
    
    # اندازه فایل‌های فعلی
    current_size = 0
    if Path(db_path).exists():
        current_size += Path(db_path).stat().st_size
    if Path(wal_path).exists():
        current_size += Path(wal_path).stat().st_size
    if Path(shm_path).exists():
        current_size += Path(shm_path).stat().st_size
    
    current_size_gb = current_size / (1024**3)
    
    # آمار فعلی
    try:
        db = SQLiteManager(db_path)
        users_result = db.execute_query("SELECT COUNT(*) as count FROM users")
        current_users = users_result.iloc[0]['count'] if not users_result.empty else 0
        
        transactions_result = db.execute_query("SELECT COUNT(*) as count FROM transactions")
        current_transactions = transactions_result.iloc[0]['count'] if not transactions_result.empty else 0
        
        db.close()
    except:
        current_users = 20000  # از آخرین گزارش
        current_transactions = 11139892
    
    target_users = 100_000
    
    print(f"📈 وضعیت فعلی:")
    print(f"   👥 کاربران: {current_users:,}")
    print(f"   💳 تراکنش‌ها: {current_transactions:,}")
    print(f"   💾 اندازه فعلی: {current_size_gb:.2f} GB")
    
    # محاسبه نسبت
    if current_users > 0:
        scale_factor = target_users / current_users
        avg_transactions_per_user = current_transactions / current_users
        
        print(f"\n🔢 محاسبات:")
        print(f"   📊 نسبت مقیاس: {scale_factor:.1f}x")
        print(f"   💳 میانگین تراکنش/کاربر: {avg_transactions_per_user:.1f}")
        
        # تخمین اندازه خطی
        linear_estimate = current_size_gb * scale_factor
        
        # تخمین‌های مختلف
        estimates = {
            "خطی (Linear)": linear_estimate,
            "با ضریب کمپرس (0.85x)": linear_estimate * 0.85,
            "با overhead اضافی (1.2x)": linear_estimate * 1.2
        }
        
        print(f"\n🎯 تخمین اندازه نهایی:")
        for method, size in estimates.items():
            print(f"   {method}: {size:.1f} GB")
        
        # اجزای اضافی
        additional_components = {
            "Feature extraction (1M users × 28 features)": 15,
            "Analysis results و plots": 3,
            "Indexes و metadata": 10,
            "Temporary processing space": 20,
            "Safety margin": 15
        }
        
        total_additional = sum(additional_components.values())
        
        print(f"\n🧮 اجزای اضافی:")
        for component, size in additional_components.items():
            print(f"   {component}: {size} GB")
        print(f"   ─────────────────────")
        print(f"   مجموع اضافی: {total_additional} GB")
        
        # تخمین نهایی
        final_estimates = {
            "حداقل": estimates["با ضریب کمپرس (0.85x)"] + total_additional,
            "احتمالی": estimates["خطی (Linear)"] + total_additional,
            "حداکثر": estimates["با overhead اضافی (1.2x)"] + total_additional
        }
        
        print(f"\n🎯 تخمین نهایی (شامل همه اجزا):")
        for scenario, size in final_estimates.items():
            print(f"   {scenario}: {size:.0f} GB")
        
        # توصیه‌ها
        recommended_space = final_estimates["حداکثر"] * 1.3  # 30% اضافی
        
        print(f"\n💡 توصیه‌ها:")
        print(f"   📀 فضای خالی پیشنهادی: {recommended_space:.0f} GB")
        print(f"   ⚡ SSD برای سرعت بهتر")
        print(f"   🗂️  پارتیشن جداگانه برای دیتابیس")
        
        # بررسی فضای فعلی
        try:
            import psutil
            disk = psutil.disk_usage('.')
            available_gb = disk.free / (1024**3)
            
            print(f"\n💾 فضای فعلی دیسک:")
            print(f"   آزاد: {available_gb:.1f} GB")
            
            if available_gb >= recommended_space:
                print(f"   ✅ فضای کافی موجود است")
            elif available_gb >= final_estimates["احتمالی"]:
                print(f"   ⚠️  فضای محدود - نظارت کنید")
            else:
                print(f"   ❌ فضای ناکافی - آزادسازی یا تغییر مکان")
                
        except ImportError:
            print(f"\n💾 لطفاً فضای دیسک را بررسی کنید")
    
    else:
        print("❌ نمی‌توان تخمین زد - داده‌ای موجود نیست")

if __name__ == "__main__":
    estimate_final_size() 