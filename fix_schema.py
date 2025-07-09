#!/usr/bin/env python3
"""
اسکریپت برای آپدیت schema دیتابیس
"""

import sqlite3
import os

def fix_schema():
    """حذف جدول user_features برای ساخت مجدد با schema جدید"""
    db_path = "database/banking_data.db"
    
    if not os.path.exists(db_path):
        print("❌ فایل دیتابیس پیدا نشد")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # حذف جدول user_features
        cursor.execute("DROP TABLE IF EXISTS user_features;")
        
        # همچنین جداول تحلیلی را حذف کنیم
        cursor.execute("DROP TABLE IF EXISTS similarity_results;")
        cursor.execute("DROP TABLE IF EXISTS clustering_results;")
        cursor.execute("DROP TABLE IF EXISTS anomaly_results;")
        
        conn.commit()
        conn.close()
        
        print("✅ Schema بروزرسانی شد")
        
    except Exception as e:
        print(f"❌ خطا در بروزرسانی schema: {e}")

if __name__ == "__main__":
    fix_schema() 