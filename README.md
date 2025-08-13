# Banking Synthetic Data Generation & Analysis | تولید و تحلیل داده‌های بانکی Synthetic

Language: [English](#english) | [فارسی](#فارسی)

---

## English

### Overview
A comprehensive project to generate realistic synthetic banking transaction data and analyze it with modern ML techniques.

### Features
- Generate up to 1,000,000 users with realistic attributes
- Each user has 20–2000 monthly transactions
- Realistic statistical distributions (Beta, Gamma, Log-normal, Poisson)
- Controlled noise injection implemented entirely with Polars for speed and fewer dependencies
- Chunk-based processing for memory efficiency
- Checkpoint files saved every 10,000 users at `output/checkpoints`

### Data Fields
```
User: user_id, age, birth_year, province, city
Transactions: amount, transaction_date, transaction_time, card_type, device_type
Extra: is_weekend, hour_of_day, is_noise, noise_type
```

### Installation
Prerequisites:
```
Python 3.8+
Git
```

Steps:
```
git clone <repo-url>
cd <repo-dir>
python -m venv banking_env
# Windows
banking_env\Scripts\activate
# Linux/Mac
source banking_env/bin/activate
pip install -r requirements.txt
```

### Usage
```
python main.py --sample     # quick sample (1,000 users)
python main.py --full       # full dataset (1,000,000 users)
python main.py --analysis   # run analysis on existing data
python main.py --help       # show options
```

### Project Structure
```
src/
  data_generation/
    generators.py
    distributions.py
    noise_injection.py  # Polars-only noise injection
  database/
    sqlite_manager.py
    async_sqlite_manager.py
    schema.py
  analysis/
  utils/
output/
  checkpoints/          # checkpoints per 10k users
  reports/
  plots/
```

### Technologies
```
Python 3.8+
Polars, NumPy
scikit-learn
matplotlib, seaborn, plotly
SQLite + aiosqlite (async)
SHAP, HDBSCAN
```

### Contributing
1) Fork 2) Create branch 3) Commit 4) Push 5) Open PR

### License
MIT (see LICENSE)

---

## فارسی

### معرفی
یک پروژه جامع برای تولید داده‌های تراکنش‌های بانکی synthetic و تحلیل آن‌ها با تکنیک‌های یادگیری ماشین.

### ویژگی‌ها
- تولید تا **۱٬۰۰۰٬۰۰۰** کاربر با ویژگی‌های واقعی
- برای هر کاربر **۲۰ تا ۲۰۰۰** تراکنش ماهانه
- استفاده از توزیع‌های آماری واقعی (Beta, Gamma, Log-normal, Poisson)
- **تزریق نویز کنترل‌شده با Polars** برای سرعت و کاهش وابستگی‌ها
- **پردازش chunk-based** برای مدیریت حافظه
- ذخیره **Checkpoint** پس از هر **۱۰٬۰۰۰** کاربر در `output/checkpoints`

### فیلدهای داده
```
اطلاعات کاربر: user_id, age, birth_year, province, city
تراکنش‌ها: amount, transaction_date, transaction_time, card_type, device_type
ویژگی‌های اضافی: is_weekend, hour_of_day, is_noise, noise_type
```

### نصب و راه‌اندازی
پیش‌نیازها:
```
Python 3.8+
Git
```
مراحل:
```
git clone <repo-url>
cd <repo-dir>
python -m venv banking_env
# Windows
banking_env\Scripts\activate
# Linux/Mac
source banking_env/bin/activate
pip install -r requirements.txt
```

### استفاده
```
python main.py --sample     # تولید نمونه (۱۰۰۰ کاربر)
python main.py --full       # تولید کامل (۱ میلیون کاربر)
python main.py --analysis   # اجرای تحلیل
python main.py --help       # راهنما
```

### ساختار پروژه
```
src/
  data_generation/
    generators.py
    distributions.py
    noise_injection.py  # تزریق نویز با Polars
  database/
    sqlite_manager.py
    async_sqlite_manager.py
    schema.py
  analysis/
  utils/
output/
  checkpoints/          # ذخیره‌ی چک‌پوینت هر ۱۰هزار کاربر
  reports/
  plots/
```

### تکنولوژی‌ها
```
Python 3.8+
Polars, NumPy
scikit-learn
matplotlib, seaborn, plotly
SQLite + aiosqlite (async)
SHAP, HDBSCAN
```

### مشارکت
۱) Fork ۲) Branch ۳) Commit ۴) Push ۵) Pull Request

### مجوز
MIT (فایل LICENSE)