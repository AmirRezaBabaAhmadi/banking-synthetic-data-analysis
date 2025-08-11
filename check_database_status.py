#!/usr/bin/env python3
"""
بررسی وضعیت دیتابیس و میزان پیشرفت پروژه
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.database.sqlite_manager import SQLiteManager

def check_database_status():
    """بررسی وضعیت دیتابیس اصلی"""
    
    print("🔍 بررسی وضعیت دیتابیس...")
    print("="*50)
    
    db_path = "database/banking_data.db"
    if not Path(db_path).exists():
        print("❌ دیتابیس اصلی یافت نشد!")
        return
    
    try:
        db = SQLiteManager(db_path)
        
        # بررسی تعداد کاربران
        users_result = db.execute_query("SELECT COUNT(*) as count FROM users")
        total_users = users_result.iloc[0]['count']
        
        # بررسی تعداد تراکنش‌ها
        transactions_result = db.execute_query("SELECT COUNT(*) as count FROM transactions")
        total_transactions = transactions_result.iloc[0]['count']
        
        # بررسی وضعیت ویژگی‌ها
        try:
            features_result = db.execute_query("SELECT COUNT(*) as count FROM user_features")
            total_features = features_result.iloc[0]['count']
        except:
            total_features = 0
        
        # اندازه فایل
        file_size = Path(db_path).stat().st_size / (1024**3)  # GB
        
        print(f"👥 کل کاربران: {total_users:,}")
        print(f"💳 کل تراکنش‌ها: {total_transactions:,}")
        print(f"🧬 ویژگی‌های استخراج شده: {total_features:,}")
        print(f"💾 اندازه دیتابیس: {file_size:.2f} GB")
        
        if total_users > 0:
            avg_transactions = total_transactions / total_users
            print(f"📊 میانگین تراکنش در کاربر: {avg_transactions:.1f}")
        
        # بررسی پیشرفت
        target_users = 100_000
        progress = (total_users / target_users) * 100
        print(f"📈 پیشرفت: {progress:.2f}% از هدف 1 میلیون کاربر")
        
        # وضعیت کلی
        print("\n🎯 وضعیت پردازش:")
        if total_users >= target_users:
            print("✅ تولید داده کامل شده")
        elif total_users > 0:
            print(f"🟡 در حال پردازش... ({total_users:,} کاربر تولید شده)")
        else:
            print("🔴 هنوز شروع نشده")
            
        if total_features > 0:
            print(f"✅ استخراج ویژگی انجام شده ({total_features:,} کاربر)")
        else:
            print("⏳ استخراج ویژگی انجام نشده")
        
        db.close()
        
    except Exception as e:
        print(f"❌ خطا در بررسی دیتابیس: {e}")

if __name__ == "__main__":
    check_database_status() 