"""
تعریف schema جداول دیتابیس
"""

# Schema جدول کاربران
USERS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    age INTEGER NOT NULL,
    birth_year INTEGER NOT NULL,
    registration_date TEXT NOT NULL,
    province TEXT NOT NULL,
    city TEXT NOT NULL,
    preferred_card_type TEXT NOT NULL,
    primary_device TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Schema جدول تراکنش‌ها
TRANSACTIONS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    transaction_date TEXT NOT NULL,
    transaction_time TEXT NOT NULL,
    province TEXT NOT NULL,
    city TEXT NOT NULL,
    card_type TEXT NOT NULL,
    device_type TEXT NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    hour_of_day INTEGER NOT NULL,
    day_of_month INTEGER NOT NULL,
    is_noise BOOLEAN DEFAULT FALSE,
    noise_type TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Key
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);
"""

# Schema جدول feature های محاسبه‌شده
USER_FEATURES_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_features (
    user_id INTEGER PRIMARY KEY,
    
    -- فیچرهای حجم تراکنش
    total_transactions INTEGER NOT NULL,
    total_amount REAL NOT NULL,
    avg_transaction_amount REAL NOT NULL,
    std_transaction_amount REAL NOT NULL,
    min_transaction_amount REAL NOT NULL,
    max_transaction_amount REAL NOT NULL,
    median_transaction_amount REAL NOT NULL,
    
    -- فیچرهای زمانی
    avg_hour_of_day REAL NOT NULL,
    std_hour_of_day REAL NOT NULL,
    weekend_transaction_ratio REAL NOT NULL,
    night_transaction_ratio REAL NOT NULL,
    transaction_frequency REAL NOT NULL,
    
    -- فیچرهای مکانی
    unique_cities_count INTEGER NOT NULL,
    unique_provinces_count INTEGER NOT NULL,
    most_frequent_city TEXT NOT NULL,
    most_frequent_province TEXT NOT NULL,
    geographical_diversity REAL NOT NULL,
    
    -- فیچرهای رفتاری
    card_type_diversity REAL NOT NULL,
    device_type_diversity REAL NOT NULL,
    regularity_score REAL NOT NULL,
    
    -- فیچرهای توزیع مبلغ
    amount_percentile_25 REAL NOT NULL,
    amount_percentile_75 REAL NOT NULL,
    amount_iqr REAL NOT NULL,
    amount_skewness REAL NOT NULL,
    
    -- فیچرهای دموگرافیک
    age INTEGER NOT NULL,
    birth_year INTEGER NOT NULL,
    
    -- برچسب‌های تحلیلی
    spending_category TEXT DEFAULT NULL,
    behavior_cluster INTEGER DEFAULT NULL,
    anomaly_score REAL DEFAULT NULL,
    is_anomaly BOOLEAN DEFAULT FALSE,
    
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Key
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);
"""

# Schema جدول نتایج similarity search
SIMILARITY_RESULTS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS similarity_results (
    query_user_id INTEGER NOT NULL,
    similar_user_id INTEGER NOT NULL,
    similarity_score REAL NOT NULL,
    similarity_method TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (query_user_id, similar_user_id, similarity_method),
    
    -- Foreign Keys
    FOREIGN KEY (query_user_id) REFERENCES users (user_id),
    FOREIGN KEY (similar_user_id) REFERENCES users (user_id)
);
"""

# Schema جدول clustering results
CLUSTERING_RESULTS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS clustering_results (
    user_id INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,
    cluster_method TEXT NOT NULL,
    distance_to_center REAL DEFAULT NULL,
    cluster_probability REAL DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (user_id, cluster_method),
    
    -- Foreign Key
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);
"""

# Schema جدول anomaly detection results
ANOMALY_RESULTS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS anomaly_results (
    user_id INTEGER NOT NULL,
    anomaly_method TEXT NOT NULL,
    anomaly_score REAL NOT NULL,
    is_anomaly BOOLEAN NOT NULL,
    shap_values TEXT DEFAULT NULL,  -- JSON string of SHAP values
    feature_importance TEXT DEFAULT NULL,  -- JSON string
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (user_id, anomaly_method),
    
    -- Foreign Key
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);
"""

# لیست تمام schemas
ALL_SCHEMAS = [
    USERS_TABLE_SCHEMA,
    TRANSACTIONS_TABLE_SCHEMA,
    USER_FEATURES_TABLE_SCHEMA,
    SIMILARITY_RESULTS_TABLE_SCHEMA,
    CLUSTERING_RESULTS_TABLE_SCHEMA,
    ANOMALY_RESULTS_TABLE_SCHEMA
]

def get_create_index_queries():
    """تولید کوئری‌های ایجاد ایندکس برای بهینه‌سازی"""
    return [
        # ایندکس‌های اصلی برای جدول users
        "CREATE INDEX IF NOT EXISTS idx_age ON users(age);",
        "CREATE INDEX IF NOT EXISTS idx_province ON users(province);",
        "CREATE INDEX IF NOT EXISTS idx_city ON users(city);",
        "CREATE INDEX IF NOT EXISTS idx_card_type ON users(preferred_card_type);",
        
        # ایندکس‌های اصلی برای جدول transactions
        "CREATE INDEX IF NOT EXISTS idx_user_id ON transactions(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_amount ON transactions(amount);",
        "CREATE INDEX IF NOT EXISTS idx_transaction_date ON transactions(transaction_date);",
        "CREATE INDEX IF NOT EXISTS idx_hour_of_day ON transactions(hour_of_day);",
        "CREATE INDEX IF NOT EXISTS idx_is_noise ON transactions(is_noise);",
        "CREATE INDEX IF NOT EXISTS idx_province_city ON transactions(province, city);",
        
        # ایندکس‌های برای user_features
        "CREATE INDEX IF NOT EXISTS idx_total_amount ON user_features(total_amount);",
        "CREATE INDEX IF NOT EXISTS idx_behavior_cluster ON user_features(behavior_cluster);",
        "CREATE INDEX IF NOT EXISTS idx_anomaly_score ON user_features(anomaly_score);",
        "CREATE INDEX IF NOT EXISTS idx_is_anomaly ON user_features(is_anomaly);",
        
        # ایندکس‌های برای similarity_results
        "CREATE INDEX IF NOT EXISTS idx_query_user ON similarity_results(query_user_id);",
        "CREATE INDEX IF NOT EXISTS idx_similarity_score ON similarity_results(similarity_score DESC);",
        "CREATE INDEX IF NOT EXISTS idx_method ON similarity_results(similarity_method);",
        
        # ایندکس‌های برای clustering_results
        "CREATE INDEX IF NOT EXISTS idx_cluster_id ON clustering_results(cluster_id);",
        "CREATE INDEX IF NOT EXISTS idx_cluster_method ON clustering_results(cluster_method);",
        "CREATE INDEX IF NOT EXISTS idx_distance ON clustering_results(distance_to_center);",
        
        # ایندکس‌های برای anomaly_results
        "CREATE INDEX IF NOT EXISTS idx_anomaly_score_desc ON anomaly_results(anomaly_score DESC);",
        "CREATE INDEX IF NOT EXISTS idx_is_anomaly_result ON anomaly_results(is_anomaly);",
        "CREATE INDEX IF NOT EXISTS idx_anomaly_method ON anomaly_results(anomaly_method);",
        
        # ایندکس‌های ترکیبی برای queries پیچیده
        "CREATE INDEX IF NOT EXISTS idx_user_amount_date ON transactions(user_id, amount, transaction_date);",
        "CREATE INDEX IF NOT EXISTS idx_date_hour ON transactions(transaction_date, hour_of_day);",
        "CREATE INDEX IF NOT EXISTS idx_user_features_cluster ON user_features(behavior_cluster, total_amount);",
        "CREATE INDEX IF NOT EXISTS idx_amount_range ON transactions(amount) WHERE amount > 1000000;",  # برای تراکنش‌های بزرگ
        
        # ایندکس برای anomaly detection
        "CREATE INDEX IF NOT EXISTS idx_high_amount_young_age ON transactions(user_id, amount) WHERE amount > 5000000;",
        
        # ایندکس برای geographical analysis
        "CREATE INDEX IF NOT EXISTS idx_geo_analysis ON transactions(province, city, amount);",
    ] 