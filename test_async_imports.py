#!/usr/bin/env python3
"""
تست import کردن تمام ماژول‌های async
"""

import sys
from pathlib import Path

# اضافه کردن مسیر src
sys.path.append(str(Path(__file__).parent / "src"))

def test_async_imports():
    """تست import تمام ماژول‌های async"""
    
    print("🧪 تست import ماژول‌های async...")
    
    try:
        # تست async feature extractors
        print("1️⃣ تست async feature extractors...")
        from src.feature_engineering.async_extractors import AsyncBankingFeatureExtractor, run_async_feature_extraction
        print("   ✅ AsyncBankingFeatureExtractor imported successfully")
        
        # تست async clustering
        print("2️⃣ تست async clustering...")
        from src.analysis.async_clustering import AsyncBankingCustomerClustering, run_async_clustering_analysis
        print("   ✅ AsyncBankingCustomerClustering imported successfully")
        
        # تست async anomaly detection
        print("3️⃣ تست async anomaly detection...")
        from src.analysis.async_anomaly_detection import AsyncBankingAnomalyDetection, run_async_anomaly_analysis
        print("   ✅ AsyncBankingAnomalyDetection imported successfully")
        
        # تست main script updates
        print("4️⃣ تست main script...")
        import main
        print("   ✅ Updated main.py imported successfully")
        
        # تست report generator
        print("5️⃣ تست report generator...")
        import generate_analysis_report
        print("   ✅ AnalysisReportGenerator imported successfully")
        
        # تست performance tester
        print("6️⃣ تست performance tester...")
        import test_async_performance
        print("   ✅ AsyncPerformanceTester imported successfully")
        
        print("\n🎉 همه ماژول‌های async با موفقیت import شدند!")
        return True
        
    except ImportError as e:
        print(f"❌ خطا در import: {e}")
        return False
    except Exception as e:
        print(f"❌ خطای غیرمنتظره: {e}")
        return False

def test_dependencies():
    """تست وجود dependencies مورد نیاز"""
    
    print("\n📦 تست dependencies...")
    
    required_packages = [
        'asyncio',
        'concurrent.futures',
        'aiofiles',
        'pandas',
        'numpy',
        'sklearn',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'concurrent.futures':
                import concurrent.futures
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  پکیج‌های گم شده: {', '.join(missing_packages)}")
        print("💡 برای نصب: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ همه dependencies موجود هستند!")
        return True

def main():
    """تابع اصلی"""
    print("=" * 60)
    print("🚀 تست سیستم Async Processing")
    print("=" * 60)
    
    # تست dependencies
    deps_ok = test_dependencies()
    
    if not deps_ok:
        print("\n❌ لطفاً ابتدا dependencies را نصب کنید")
        return
    
    # تست imports
    imports_ok = test_async_imports()
    
    if imports_ok:
        print("\n" + "=" * 60)
        print("✅ سیستم async آماده استفاده است!")
        print("=" * 60)
        print("\n🔥 برای شروع:")
        print("   python main.py --all              # اجرای کامل با async")
        print("   python test_async_performance.py  # تست performance")
        print("   python generate_analysis_report.py # تولید گزارش")
        print("=" * 60)
    else:
        print("\n❌ سیستم async آماده نیست - لطفاً خطاها را برطرف کنید")

if __name__ == "__main__":
    main() 