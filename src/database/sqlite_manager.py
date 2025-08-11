"""
کلاس مدیریت دیتابیس SQLite با پشتیبانی از chunk-based operations
DEPRECATED: این فایل deprecated است. از async_sqlite_manager استفاده کنید.
"""

import warnings
warnings.warn(
    "sqlite_manager.py is deprecated. Use async_sqlite_manager.py instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import the new hybrid manager
from .async_sqlite_manager import HybridSQLiteManager, AsyncSQLiteManager

# برای سازگاری با کد موجود
SQLiteManager = HybridSQLiteManager