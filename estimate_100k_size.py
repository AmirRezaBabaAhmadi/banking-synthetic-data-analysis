#!/usr/bin/env python3
"""
تخمین اندازه دیتاست برای 100 هزار کاربر
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.database.sqlite_manager import SQLiteManager

def estimate_100k_size():
    """تخمین اندازه دیتاست برای 100 هزار کاربر"""
    
    print("📊 تخمین اندازه دیتاست برای 100,000 کاربر")
    print("="*55)
    
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
        current_users = 20000
        current_transactions = 11139892
    
    target_users_100k = 100_000
    target_users_1m = 1_000_000
    
    print(f"📈 وضعیت فعلی:")
    print(f"   👥 کاربران: {current_users:,}")
    print(f"   💳 تراکنش‌ها: {current_transactions:,}")
    print(f"   💾 اندازه فعلی: {current_size_gb:.2f} GB")
    
    if current_users > 0:
        # محاسبه برای 100K
        scale_factor_100k = target_users_100k / current_users
        avg_transactions_per_user = current_transactions / current_users
        
        print(f"\n🔢 محاسبات برای 100,000 کاربر:")
        print(f"   📊 نسبت مقیاس: {scale_factor_100k:.1f}x")
        print(f"   💳 میانگین تراکنش/کاربر: {avg_transactions_per_user:.1f}")
        
        # تخمین اندازه برای 100K
        linear_estimate_100k = current_size_gb * scale_factor_100k
        
        estimates_100k = {
            "خطی (Linear)": linear_estimate_100k,
            "با ضریب کمپرس (0.85x)": linear_estimate_100k * 0.85,
            "با overhead اضافی (1.2x)": linear_estimate_100k * 1.2
        }
        
        print(f"\n🎯 تخمین اندازه دیتابیس برای 100K:")
        for method, size in estimates_100k.items():
            print(f"   {method}: {size:.1f} GB")
        
        # اجزای اضافی برای 100K
        additional_components_100k = {
            "Feature extraction (100K users × 28 features)": 1.5,
            "Analysis results و plots": 2,
            "Indexes و metadata": 3,
            "Temporary processing space": 5,
            "Safety margin": 3
        }
        
        total_additional_100k = sum(additional_components_100k.values())
        
        print(f"\n🧮 اجزای اضافی برای 100K:")
        for component, size in additional_components_100k.items():
            print(f"   {component}: {size} GB")
        print(f"   ─────────────────────")
        print(f"   مجموع اضافی: {total_additional_100k} GB")
        
        # تخمین نهایی برای 100K
        final_estimates_100k = {
            "حداقل": estimates_100k["با ضریب کمپرس (0.85x)"] + total_additional_100k,
            "احتمالی": estimates_100k["خطی (Linear)"] + total_additional_100k,
            "حداکثر": estimates_100k["با overhead اضافی (1.2x)"] + total_additional_100k
        }
        
        print(f"\n🎯 تخمین نهایی برای 100,000 کاربر:")
        for scenario, size in final_estimates_100k.items():
            print(f"   {scenario}: {size:.0f} GB")
        
        # مقایسه با 1M
        scale_factor_1m = target_users_1m / current_users
        linear_estimate_1m = current_size_gb * scale_factor_1m
        final_estimate_1m = linear_estimate_1m + 63  # از محاسبه قبلی
        
        print(f"\n📊 مقایسه 100K vs 1M کاربر:")
        print(f"   100K کاربر: {final_estimates_100k['احتمالی']:.0f} GB")
        print(f"   1M کاربر: {final_estimate_1m:.0f} GB")
        
        ratio = final_estimate_1m / final_estimates_100k['احتمالی']
        saving = final_estimate_1m - final_estimates_100k['احتمالی']
        
        print(f"   💾 صرفه‌جویی: {saving:.0f} GB ({ratio:.1f}x کمتر)")
        
        # زمان پردازش
        print(f"\n⏰ تخمین زمان پردازش:")
        print(f"   100K کاربر: 2-4 ساعت")
        print(f"   1M کاربر: 20-40 ساعت")
        print(f"   صرفه‌جویی زمان: ~90%")
        
        # مزایا و معایب
        print(f"\n✅ مزایای 100K کاربر:")
        print(f"   • سرعت بالای پردازش")
        print(f"   • مصرف کم منابع")
        print(f"   • آزمایش و توسعه سریع‌تر")
        print(f"   • مناسب برای proof of concept")
        
        print(f"\n⚠️ محدودیت‌های 100K کاربر:")
        print(f"   • کمتر نماینده جمعیت واقعی")
        print(f"   • Pattern های پیچیده ممکن است نمایان نشود")
        print(f"   • برای production scale مناسب نیست")
        
        # توصیه
        print(f"\n💡 توصیه:")
        if final_estimates_100k['احتمالی'] < 20:
            print(f"   🚀 شروع با 100K برای تست و سپس scale up")
            print(f"   📈 اگر نتایج رضایت‌بخش بود، به 1M برسانید")
        
        # بررسی فضای فعلی
        try:
            import psutil
            disk = psutil.disk_usage('.')
            available_gb = disk.free / (1024**3)
            
            print(f"\n💾 فضای فعلی دیسک:")
            print(f"   آزاد: {available_gb:.1f} GB")
            print(f"   برای 100K: ✅ بیش از کافی")
            print(f"   برای 1M: {'✅ کافی' if available_gb > final_estimate_1m else '⚠️ محدود'}")
                
        except ImportError:
            pass
        
        print(f"\n🎯 پیشنهاد:")
        print(f"   1️⃣ شروع با 100K کاربر")
        print(f"   2️⃣ آزمایش الگوریتم‌ها و تحلیل‌ها")
        print(f"   3️⃣ در صورت نیاز، scale up به 1M")

if __name__ == "__main__":
    estimate_100k_size() 