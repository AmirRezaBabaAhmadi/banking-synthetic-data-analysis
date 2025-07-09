# راهنمای استفاده از سیستم تولید و تحلیل داده‌های بانکی

## 🚀 شروع سریع

### 1. نصب و راه‌اندازی
```bash
# کلون پروژه
git clone <repository-url>
cd banking_synthetic_data

# نصب وابستگی‌ها
pip install -r requirements.txt

# ایجاد ساختار پوشه‌ها
python -c "from src.utils.config import setup_directories; setup_directories()"
```

### 2. اجرای سریع (داده نمونه)
```bash
# تولید 1000 کاربر نمونه
python main.py --sample

# اجرای تحلیل‌ها
python main.py --analysis
```

### 3. اجرای کامل
```bash
# تولید کامل 1 میلیون کاربر + تحلیل
python main.py --all
```

## 📊 مراحل مختلف اجرا

### مرحله 1: تولید داده نمونه
```bash
python main.py --sample
```

**خروجی:**
- `database/banking_sample.db` - دیتابیس نمونه
- `output/reports/sample_generation_report.txt` - گزارش تولید
- `output/reports/sample_noise_analysis.txt` - تحلیل نویز

**آمار نمونه:**
- 1,000 کاربر
- ~80,000 تراکنش
- ~3% نویز
- زمان اجرا: 10-15 ثانیه

### مرحله 2: تولید dataset کامل
```bash
python main.py --full
```

**خروجی:**
- `database/banking_data.db` - دیتابیس کامل (~15 GB)
- گزارش‌های تفصیلی
- آمار کامل نویزها

**آمار کامل:**
- 1,000,000 کاربر
- ~80,000,000 تراکنش
- ~3.8% نویز
- زمان اجرا: 2-3 ساعت

### مرحله 3: تحلیل پیشرفته
```bash
python main.py --analysis
```

**شامل:**
1. **Feature Engineering** - استخراج 25+ ویژگی
2. **Clustering** - دسته‌بندی کاربران
3. **Anomaly Detection** - تشخیص ناهنجاری با SHAP
4. **Similarity Search** - جستجوی تشابه کاربران

**خروجی:**
- `output/analysis_summary.json` - خلاصه نتایج
- `output/plots/` - نمودارهای تحلیلی
- جداول تحلیل در دیتابیس

## 📓 استفاده از Jupyter Notebooks

### 1. نوت‌بوک تولید داده
```bash
jupyter notebook notebooks/01_data_generation.ipynb
```

### 2. نوت‌بوک Feature Engineering
```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```

### 3. نوت‌بوک Clustering
```bash
jupyter notebook notebooks/03_clustering_analysis.ipynb
```

### 4. نوت‌بوک Anomaly Detection
```bash
jupyter notebook notebooks/04_anomaly_detection.ipynb
```

## 🔧 استفاده برنامه‌نویسی

### تولید داده به صورت دستی
```python
from src.data_generation.generators import BankingDataGenerator
from src.database.sqlite_manager import SQLiteManager

# ایجاد مولد داده
db_manager = SQLiteManager()
generator = BankingDataGenerator(db_manager)

# تولید 5000 کاربر
stats = generator.generate_sample_data(num_users=5000)
print(f"تولید شد: {stats['total_transactions_generated']:,} تراکنش")

generator.close()
```

### استخراج ویژگی
```python
from src.feature_engineering.extractors import BankingFeatureExtractor

# استخراج ویژگی برای کاربران خاص
extractor = BankingFeatureExtractor()
user_ids = [1, 2, 3, 4, 5]

features_df = extractor.extract_features_batch(user_ids)
print(f"ویژگی‌های استخراج‌شده: {len(features_df.columns)}")
```

### اجرای Clustering
```python
from src.analysis.clustering import BankingCustomerClustering

# تحلیل clustering
clustering = BankingCustomerClustering()
results = clustering.run_complete_clustering_analysis(sample_size=5000)

print(f"بهترین روش: {results['best_method']}")
print(f"تعداد cluster ها: {len(results['cluster_profiles'])}")
```

### تشخیص ناهنجاری
```python
from src.analysis.anomaly_detection import BankingAnomalyDetector

# تشخیص ناهنجاری
detector = BankingAnomalyDetector()
results = detector.run_complete_anomaly_analysis(sample_size=5000)

print(f"ناهنجاری‌های کشف‌شده: {results['total_anomalies_detected']}")
```

### جستجوی تشابه
```python
from src.analysis.similarity_search import BankingSimilaritySearch

# جستجوی تشابه
similarity = BankingSimilaritySearch()
results = similarity.run_complete_similarity_analysis()

print(f"کاربران تست جدید: {results['new_test_users']}")
```

## 📈 تصویرسازی و گزارش‌ها

### ایجاد داشبورد
```python
from src.utils.visualization import BankingVisualizationUtils

visualizer = BankingVisualizationUtils()

# داشبورد جامع
features_df = # ... داده‌های ویژگی
transactions_df = # ... داده‌های تراکنش

dashboard = visualizer.create_comprehensive_dashboard(
    features_df, 
    transactions_df
)
```

### نمودارهای تخصصی
```python
# نمودار توزیع ویژگی‌ها
fig1 = visualizer.plot_feature_distributions(features_df)

# ماتریس همبستگی
fig2 = visualizer.plot_correlation_matrix(features_df)

# نتایج clustering
fig3 = visualizer.plot_clustering_results(features_df, cluster_labels, "kmeans")

# نتایج anomaly detection
fig4 = visualizer.plot_anomaly_analysis(features_df, anomaly_scores, is_anomaly, "isolation_forest")
```

## 🎛️ تنظیمات پیشرفته

### تغییر پارامترهای تولید داده
```python
# در src/utils/config.py
DATA_GENERATION_CONFIG = {
    'default_users_count': 1000000,  # تعداد کاربران
    'chunk_size': 10000,             # اندازه chunk
    'min_transactions_per_user': 20, # حداقل تراکنش
    'max_transactions_per_user': 2000, # حداکثر تراکنش
    
    # نرخ نویز
    'location_noise_rate': 0.02,      # 2%
    'age_amount_noise_rate': 0.005,   # 0.5%
    'time_pattern_noise_rate': 0.01,  # 1%
    'amount_outlier_rate': 0.003      # 0.3%
}
```

### تنظیم الگوریتم‌های ML
```python
# Clustering
clustering_results = clustering.compare_clustering_methods(X)

# Anomaly Detection با پارامترهای سفارشی
iso_results = detector.fit_isolation_forest(X, contamination=0.05)
svm_results = detector.fit_one_class_svm(X, nu=0.05)

# Similarity Search
similarity_results = similarity.find_similar_users(
    target_user_id=12345, 
    n_similar=20
)
```

## 🔍 بررسی نتایج

### بررسی دیتابیس
```python
# اتصال به دیتابیس
db_manager = SQLiteManager()

# آمار کلی
users_count = db_manager.execute_query("SELECT COUNT(*) FROM users")
trans_count = db_manager.execute_query("SELECT COUNT(*) FROM transactions")

# بررسی نویز
noise_stats = db_manager.execute_query("""
    SELECT 
        noise_type, 
        COUNT(*) as count,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions) as percentage
    FROM transactions 
    WHERE is_noise = 1 
    GROUP BY noise_type
""")

print(noise_stats)
```

### بررسی نتایج تحلیل
```python
# بررسی clustering
clustering_results = db_manager.execute_query("SELECT * FROM clustering_results")

# بررسی anomaly detection
anomaly_results = db_manager.execute_query("SELECT * FROM anomaly_results")

# بررسی similarity search
similarity_results = db_manager.execute_query("SELECT * FROM similarity_results")
```

## 🚨 رفع مشکلات رایج

### مشکل حافظه
```python
# کاهش sample size
results = clustering.run_complete_clustering_analysis(sample_size=1000)

# کاهش chunk size
generator = BankingDataGenerator(chunk_size=5000)
```

### مشکل سرعت
```python
# استفاده از دیتابیس کوچکتر
db_manager = SQLiteManager("database/banking_sample.db")

# محدود کردن ویژگی‌ها
important_features = ['total_amount', 'total_transactions', 'age']
```

### مشکل کتابخانه‌ها
```bash
# نصب کتابخانه‌های اختیاری
pip install umap-learn hdbscan shap

# در صورت مشکل
pip install --upgrade scikit-learn pandas numpy
```

## 📁 ساختار فایل‌های خروجی

```
output/
├── plots/
│   ├── comprehensive_dashboard.png
│   ├── clustering_kmeans.png
│   ├── anomaly_isolation_forest.png
│   └── similarity_analysis.png
├── reports/
│   ├── sample_generation_report.txt
│   ├── full_generation_report.txt
│   ├── sample_noise_analysis.txt
│   └── full_noise_analysis.txt
└── analysis_summary.json
```

## 🎯 مثال‌های کاربردی

### سناریو 1: تحلیل سریع داده کوچک
```bash
# تولید نمونه کوچک
python main.py --sample

# تحلیل سریع
python main.py --analysis
```

### سناریو 2: پروژه کامل برای نمایش
```bash
# اجرای کامل (2-3 ساعت)
python main.py --all
```

### سناریو 3: توسعه و تست
```bash
# فقط تولید داده
python main.py --sample

# استفاده از notebook برای آزمایش
jupyter notebook notebooks/
```

---

**💡 نکته:** برای اجرای سریع‌تر، از `--sample` استفاده کنید. برای نمایش کامل پروژه، از `--all` استفاده کنید. 