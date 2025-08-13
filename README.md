# 🏦 پروژه تولید و تحلیل داده‌های بانکی Synthetic

یک پروژه جامع برای تولید داده‌های تراکنش‌های بانکی synthetic و تحلیل آن‌ها با استفاده از تکنیک‌های یادگیری ماشین.

## 📋 فهرست مطالب

- [ویژگی‌ها](#ویژگیها)
- [نصب و راه‌اندازی](#نصب-و-راهاندازی)
- [استفاده](#استفاده)
- [ساختار پروژه](#ساختار-پروژه)
- [نتایج](#نتایج)
- [تکنولوژی‌ها](#تکنولوژیها)
- [مشارکت](#مشارکت)

## 🌟 ویژگی‌ها

### تولید داده‌های Synthetic
- تولید **1 میلیون کاربر** با ویژگی‌های واقعی
- هر کاربر **20-2000 تراکنش** در ماه
- استفاده از **توزیع‌های آماری واقعی** (Beta, Gamma, Log-normal, Poisson)
- **تزریق نویز کنترل‌شده (تماماً با Polars)** برای عملکرد سریع و کاهش وابستگی‌ها
- **پردازش chunk-based** برای مدیریت حافظه
- **ذخیره Checkpoint** پس از هر **۱۰٬۰۰۰** کاربر در مسیر `output/checkpoints`

### فیلدهای داده
```
📊 اطلاعات کاربر: user_id, age, birth_year, location (استان/شهر ایران)
💰 تراکنش‌ها: amount, date, time, card_type, device_type
🏷️ ویژگی‌های اضافی: is_weekend, hour_of_day, is_noise
```

### تحلیل‌های پیشرفته ML
- **Feature Engineering**: استخراج 28 ویژگی تخصصی
- **Clustering**: 4 الگوریتم (K-means, DBSCAN, Gaussian Mixture, HDBSCAN)
- **Anomaly Detection**: 3 روش (Isolation Forest, One-Class SVM, SHAP Analysis)
- **Similarity Search**: جستجوی کاربران مشابه با KNN

## 🚀 نصب و راه‌اندازی

### پیش‌نیازها
```bash
Python 3.8+
Git
```

### مراحل نصب

1. **کلون کردن پروژه**
```bash
git clone https://github.com/yourusername/banking-synthetic-data.git
cd banking-synthetic-data
```

2. **ایجاد virtual environment**
```bash
python -m venv banking_env
# Windows
banking_env\Scripts\activate
# Linux/Mac
source banking_env/bin/activate
```

3. **نصب وابستگی‌ها**
```bash
pip install -r requirements.txt
```

## 💻 استفاده

### تولید داده نمونه (سریع)
```bash
python main.py --sample
```

### تولید داده کامل (1 میلیون کاربر)
```bash
python main.py --full
```

### اجرای تحلیل‌های ML
```bash
python main.py --analysis
```

### نمایش گزینه‌ها
```bash
python main.py --help
```

## 📁 ساختار پروژه

```
banking-synthetic-data/
├── 📂 src/
│   ├── 📂 data_generation/     # تولید داده‌ها
│   │   ├── generators.py       # کلاس‌های تولید کننده
│   │   ├── distributions.py    # توزیع‌های آماری
│   │   └── noise_injection.py  # تزریق نویز
│   ├── 📂 database/           # مدیریت دیتابیس
│   │   ├── sqlite_manager.py   # اتصال SQLite
│   │   └── schema.py          # ساختار جداول
│   ├── 📂 analysis/           # تحلیل‌های ML
│   │   ├── feature_engineering.py  # استخراج ویژگی
│   │   ├── clustering.py           # خوشه‌بندی
│   │   ├── anomaly_detection.py    # تشخیص ناهنجاری
│   │   └── similarity_search.py    # جستجوی تشابه
│   └── 📂 utils/              # ابزارهای کمکی
│       └── visualization.py    # تصویرسازی
├── 📂 output/                 # نتایج و خروجی‌ها
│   ├── plots/                 # نمودارها
│   └── reports/               # گزارش‌ها
├── 📂 config/                 # تنظیمات
├── 📂 notebooks/              # Jupyter notebooks
├── main.py                    # فایل اصلی
├── requirements.txt           # وابستگی‌ها
└── README.md                  # این فایل
```

## 📊 نتایج

### آمار تولید داده (حالت Sample)
```
👥 کاربران: 1,000
💳 تراکنش‌ها: 1,038,758
🎯 نویز تزریق‌شده: 34,673 رکورد
⏱️ زمان تولید: ~659 ثانیه
```

### نتایج تحلیل ML
```
🔧 ویژگی‌های استخراج‌شده: 28
📊 بهترین Clustering: K-means (Silhouette Score: 0.389)
⚠️ ناهنجاری‌های کشف‌شده: 101
🔍 کاربران تست جدید: 100
📈 امتیاز تشابه میانگین: 0.955
```

### تصویرسازی‌ها
- 📈 **Dashboard جامع**: نمودار کلی تمام تحلیل‌ها
- 🎯 **Clustering Analysis**: تصویرسازی خوشه‌ها
- ⚠️ **Anomaly Detection**: نمایش ناهنجاری‌ها
- 🔍 **Similarity Search**: ماتریس تشابه

## 🛠️ تکنولوژی‌ها

### کتابخانه‌های اصلی
```
🐍 Python 3.8+
⚡ polars, numpy - پردازش دادهٔ سریع و کارآمد
🤖 scikit-learn - یادگیری ماشین
📈 matplotlib, seaborn, plotly - تصویرسازی
🗃️ SQLite + aiosqlite - دیتابیس و دسترسی async
🔍 SHAP - تفسیرپذیری مدل
🎯 HDBSCAN - خوشه‌بندی پیشرفته
```

### الگوریتم‌های ML
```
Clustering:
- K-means
- DBSCAN  
- Gaussian Mixture Models
- HDBSCAN

Anomaly Detection:
- Isolation Forest
- One-Class SVM
- SHAP-based Analysis

Similarity Search:
- K-Nearest Neighbors (KNN)
- Cosine Similarity
```

## 🎯 کاربردهای عملی

### صنعت بانکداری
- 🔍 **تشخیص تقلب**: شناسایی تراکنش‌های مشکوک
- 👥 **بخش‌بندی مشتریان**: گروه‌بندی بر اساس رفتار خرید
- 📊 **تحلیل رفتار**: درک الگوهای مصرف مشتریان
- 💼 **مدیریت ریسک**: ارزیابی پروفایل‌های پرخطر

### تحقیق و توسعه
- 🧪 **تست الگوریتم‌ها**: آزمایش روش‌های جدید ML
- 📚 **آموزش**: ایجاد dataset برای دوره‌های آموزشی
- 🔬 **تحقیقات دانشگاهی**: پایه‌ای برای پژوهش‌های مالی

## 🤝 مشارکت

خوشحال می‌شویم از مشارکت شما استقبال کنیم:

1. Fork کردن پروژه
2. ایجاد branch جدید (`git checkout -b feature/AmazingFeature`)
3. Commit کردن تغییرات (`git commit -m 'Add some AmazingFeature'`)
4. Push کردن به branch (`git push origin feature/AmazingFeature`)
5. باز کردن Pull Request

## 📄 مجوز

این پروژه تحت مجوز MIT منتشر شده است. فایل [LICENSE](LICENSE) را مطالعه کنید.

## 🙏 تشکر

از تمام کتابخانه‌های open source استفاده شده در این پروژه تشکر می‌کنیم.

## 📧 تماس

اگر سوال یا پیشنهادی دارید، لطفاً Issue جدید ایجاد کنید.

---

⭐ اگر این پروژه برایتان مفید بود، لطفاً ستاره بدهید! 