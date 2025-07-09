"""
ابزارهای تصویرسازی برای تحلیل داده‌های بانکی
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# تنظیمات matplotlib برای فونت فارسی
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

logger = logging.getLogger(__name__)

class BankingVisualizationUtils:
    """کلاس ابزارهای تصویرسازی برای داده‌های بانکی"""
    
    def __init__(self, output_dir: str = "output/plots"):
        """مقداردهی اولیه"""
        self.output_dir = output_dir
        
        # ایجاد پوشه خروجی
        os.makedirs(output_dir, exist_ok=True)
        
        # تنظیمات رنگ
        self.color_palette = sns.color_palette("husl", 10)
        
        logger.info(f"BankingVisualizationUtils initialized, output_dir: {output_dir}")
    
    def plot_feature_distributions(self, features_df: pd.DataFrame, 
                                 save_name: str = "feature_distributions") -> plt.Figure:
        """نمایش توزیع ویژگی‌های مختلف"""
        numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # حذف user_id
        numeric_features = [col for col in numeric_features if col != 'user_id']
        
        # انتخاب ویژگی‌های مهم
        important_features = [
            'total_amount', 'avg_transaction_amount', 'total_transactions',
            'transaction_frequency', 'age', 'geographical_diversity'
        ]
        
        plot_features = [f for f in important_features if f in numeric_features][:6]
        
        if not plot_features:
            logger.warning("No suitable features for distribution plot")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(plot_features):
            data = features_df[feature].dropna()
            
            axes[i].hist(data, bins=50, alpha=0.7, color=self.color_palette[i])
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # اضافه کردن آمار
            mean_val = data.mean()
            median_val = data.median()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            axes[i].legend()
        
        # حذف subplot های اضافی
        for i in range(len(plot_features), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        self._save_plot(fig, save_name)
        
        return fig
    
    def plot_correlation_matrix(self, features_df: pd.DataFrame, 
                              save_name: str = "correlation_matrix") -> plt.Figure:
        """نمایش ماتریس همبستگی ویژگی‌ها"""
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        # حذف user_id
        numeric_features = numeric_features.drop(columns=['user_id'], errors='ignore')
        
        # محدود کردن به ویژگی‌های مهم
        important_features = [
            'total_amount', 'avg_transaction_amount', 'total_transactions',
            'transaction_frequency', 'age', 'geographical_diversity',
            'night_transaction_ratio', 'weekend_transaction_ratio',
            'amount_skewness', 'regularity_score'
        ]
        
        available_features = [f for f in important_features if f in numeric_features.columns]
        
        if len(available_features) < 2:
            logger.warning("Not enough features for correlation matrix")
            return None
        
        correlation_data = numeric_features[available_features]
        correlation_matrix = correlation_data.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   ax=ax)
        
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        self._save_plot(fig, save_name)
        
        return fig
    
    def plot_clustering_results(self, features_df: pd.DataFrame, cluster_labels: np.ndarray,
                              method_name: str, save_name: str = None) -> plt.Figure:
        """نمایش نتایج clustering"""
        if save_name is None:
            save_name = f"clustering_{method_name}"
        
        # انتخاب دو ویژگی اصلی برای نمایش
        numeric_features = features_df.select_dtypes(include=[np.number])
        numeric_features = numeric_features.drop(columns=['user_id'], errors='ignore')
        
        if len(numeric_features.columns) >= 2:
            feature_x = 'total_amount' if 'total_amount' in numeric_features.columns else numeric_features.columns[0]
            feature_y = 'total_transactions' if 'total_transactions' in numeric_features.columns else numeric_features.columns[1]
        else:
            logger.warning("Not enough features for clustering visualization")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # نمودار scatter
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:  # noise points
                mask = cluster_labels == label
                ax1.scatter(numeric_features.loc[mask, feature_x], 
                           numeric_features.loc[mask, feature_y],
                           c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                mask = cluster_labels == label
                ax1.scatter(numeric_features.loc[mask, feature_x], 
                           numeric_features.loc[mask, feature_y],
                           c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
        
        ax1.set_xlabel(feature_x)
        ax1.set_ylabel(feature_y)
        ax1.set_title(f'{method_name} Clustering Results')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # نمودار اندازه cluster ها
        cluster_sizes = []
        cluster_ids = []
        
        for label in unique_labels:
            if label != -1:
                cluster_sizes.append(np.sum(cluster_labels == label))
                cluster_ids.append(f'Cluster {label}')
        
        if cluster_sizes:
            ax2.pie(cluster_sizes, labels=cluster_ids, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Cluster Size Distribution')
        
        plt.tight_layout()
        self._save_plot(fig, save_name)
        
        return fig
    
    def plot_anomaly_analysis(self, features_df: pd.DataFrame, anomaly_scores: np.ndarray,
                            is_anomaly: np.ndarray, method_name: str,
                            save_name: str = None) -> plt.Figure:
        """نمایش نتایج anomaly detection"""
        if save_name is None:
            save_name = f"anomaly_{method_name}"
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. توزیع anomaly scores
        axes[0, 0].hist(anomaly_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title(f'{method_name}: Anomaly Score Distribution')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. نسبت normal vs anomaly
        anomaly_counts = [np.sum(is_anomaly == 0), np.sum(is_anomaly == 1)]
        axes[0, 1].pie(anomaly_counts, labels=['Normal', 'Anomaly'], autopct='%1.1f%%',
                      colors=['lightgreen', 'red'], startangle=90)
        axes[0, 1].set_title('Normal vs Anomaly Distribution')
        
        # 3. Anomaly score vs feature
        numeric_features = features_df.select_dtypes(include=[np.number])
        if 'total_amount' in numeric_features.columns:
            axes[1, 0].scatter(numeric_features['total_amount'], anomaly_scores, 
                              c=is_anomaly, cmap='coolwarm', alpha=0.6)
            axes[1, 0].set_xlabel('Total Amount')
            axes[1, 0].set_ylabel('Anomaly Score')
            axes[1, 0].set_title('Anomaly Score vs Total Amount')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Box plot anomaly scores
        normal_scores = anomaly_scores[is_anomaly == 0]
        anomaly_scores_filtered = anomaly_scores[is_anomaly == 1]
        
        if len(anomaly_scores_filtered) > 0:
            axes[1, 1].boxplot([normal_scores, anomaly_scores_filtered], 
                              labels=['Normal', 'Anomaly'])
            axes[1, 1].set_title('Anomaly Score Comparison')
            axes[1, 1].set_ylabel('Anomaly Score')
        
        plt.tight_layout()
        self._save_plot(fig, save_name)
        
        return fig
    
    def plot_similarity_analysis(self, similarity_scores: List[float], 
                               ranks: List[int] = None,
                               save_name: str = "similarity_analysis") -> plt.Figure:
        """نمایش نتایج similarity search"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. توزیع similarity scores
        axes[0].hist(similarity_scores, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0].set_title('Distribution of Similarity Scores')
        axes[0].set_xlabel('Similarity Score')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # آمار
        mean_sim = np.mean(similarity_scores)
        median_sim = np.median(similarity_scores)
        axes[0].axvline(mean_sim, color='red', linestyle='--', label=f'Mean: {mean_sim:.3f}')
        axes[0].axvline(median_sim, color='blue', linestyle='--', label=f'Median: {median_sim:.3f}')
        axes[0].legend()
        
        # 2. Similarity vs Rank (اگر موجود باشد)
        if ranks:
            axes[1].scatter(ranks, similarity_scores, alpha=0.6, color='purple')
            axes[1].set_xlabel('Rank')
            axes[1].set_ylabel('Similarity Score')
            axes[1].set_title('Similarity Score vs Rank')
            axes[1].grid(True, alpha=0.3)
        else:
            # نمودار دسته‌بندی similarity scores
            high_sim = len([s for s in similarity_scores if s > 0.8])
            medium_sim = len([s for s in similarity_scores if 0.5 <= s <= 0.8])
            low_sim = len([s for s in similarity_scores if s < 0.5])
            
            categories = ['High (>0.8)', 'Medium (0.5-0.8)', 'Low (<0.5)']
            counts = [high_sim, medium_sim, low_sim]
            
            axes[1].bar(categories, counts, color=['green', 'orange', 'red'], alpha=0.7)
            axes[1].set_title('Similarity Score Categories')
            axes[1].set_ylabel('Count')
        
        plt.tight_layout()
        self._save_plot(fig, save_name)
        
        return fig
    
    def plot_transaction_patterns(self, transactions_df: pd.DataFrame,
                                save_name: str = "transaction_patterns") -> plt.Figure:
        """نمایش الگوهای تراکنش"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. توزیع مبلغ تراکنش
        amounts = transactions_df['amount'].values
        axes[0, 0].hist(np.log10(amounts + 1), bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Transaction Amount Distribution (Log Scale)')
        axes[0, 0].set_xlabel('Log10(Amount + 1)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. تراکنش بر حسب ساعت روز
        if 'hour_of_day' in transactions_df.columns:
            hourly_counts = transactions_df['hour_of_day'].value_counts().sort_index()
            axes[0, 1].plot(hourly_counts.index, hourly_counts.values, marker='o')
            axes[0, 1].set_title('Transactions by Hour of Day')
            axes[0, 1].set_xlabel('Hour')
            axes[0, 1].set_ylabel('Number of Transactions')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. تراکنش بر حسب نوع کارت
        if 'card_type' in transactions_df.columns:
            card_counts = transactions_df['card_type'].value_counts()
            axes[1, 0].pie(card_counts.values, labels=card_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Transactions by Card Type')
        
        # 4. تراکنش بر حسب استان
        if 'province' in transactions_df.columns:
            province_counts = transactions_df['province'].value_counts().head(10)
            axes[1, 1].barh(range(len(province_counts)), province_counts.values)
            axes[1, 1].set_yticks(range(len(province_counts)))
            axes[1, 1].set_yticklabels(province_counts.index)
            axes[1, 1].set_title('Top 10 Provinces by Transaction Count')
            axes[1, 1].set_xlabel('Number of Transactions')
        
        plt.tight_layout()
        self._save_plot(fig, save_name)
        
        return fig
    
    def create_comprehensive_dashboard(self, features_df: pd.DataFrame,
                                     transactions_df: pd.DataFrame = None,
                                     save_name: str = "comprehensive_dashboard") -> plt.Figure:
        """ایجاد داشبورد جامع"""
        fig = plt.figure(figsize=(20, 16))
        
        # تقسیم صفحه به grid
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. توزیع سن کاربران
        ax1 = fig.add_subplot(gs[0, 0])
        if 'age' in features_df.columns:
            ax1.hist(features_df['age'], bins=20, alpha=0.7, color='skyblue')
            ax1.set_title('Age Distribution')
            ax1.set_xlabel('Age')
            ax1.set_ylabel('Count')
        
        # 2. توزیع تعداد تراکنش
        ax2 = fig.add_subplot(gs[0, 1])
        if 'total_transactions' in features_df.columns:
            ax2.hist(features_df['total_transactions'], bins=30, alpha=0.7, color='lightgreen')
            ax2.set_title('Total Transactions Distribution')
            ax2.set_xlabel('Number of Transactions')
            ax2.set_ylabel('Count')
        
        # 3. توزیع مجموع مبلغ تراکنش
        ax3 = fig.add_subplot(gs[0, 2])
        if 'total_amount' in features_df.columns:
            ax3.hist(np.log10(features_df['total_amount'] + 1), bins=30, alpha=0.7, color='orange')
            ax3.set_title('Total Amount Distribution (Log)')
            ax3.set_xlabel('Log10(Total Amount)')
            ax3.set_ylabel('Count')
        
        # 4. تنوع جغرافیایی
        ax4 = fig.add_subplot(gs[0, 3])
        if 'geographical_diversity' in features_df.columns:
            ax4.hist(features_df['geographical_diversity'], bins=20, alpha=0.7, color='purple')
            ax4.set_title('Geographical Diversity')
            ax4.set_xlabel('Diversity Score')
            ax4.set_ylabel('Count')
        
        # 5-6. نمودارهای scatter
        ax5 = fig.add_subplot(gs[1, 0:2])
        if 'total_amount' in features_df.columns and 'total_transactions' in features_df.columns:
            ax5.scatter(features_df['total_transactions'], features_df['total_amount'], alpha=0.6)
            ax5.set_xlabel('Total Transactions')
            ax5.set_ylabel('Total Amount')
            ax5.set_title('Total Amount vs Total Transactions')
        
        # 7-8. نمودارهای box plot
        ax6 = fig.add_subplot(gs[1, 2:4])
        if 'age' in features_df.columns and 'avg_transaction_amount' in features_df.columns:
            # تقسیم سن به گروه‌ها
            age_groups = pd.cut(features_df['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])
            features_df['age_group'] = age_groups
            
            data_for_box = [features_df[features_df['age_group'] == group]['avg_transaction_amount'].dropna() 
                           for group in ['Young', 'Middle', 'Senior']]
            
            ax6.boxplot(data_for_box, labels=['Young', 'Middle', 'Senior'])
            ax6.set_title('Average Transaction Amount by Age Group')
            ax6.set_ylabel('Average Amount')
        
        # 9. ماتریس همبستگی کوچک
        ax7 = fig.add_subplot(gs[2:4, 0:2])
        numeric_features = features_df.select_dtypes(include=[np.number])
        key_features = ['total_amount', 'total_transactions', 'age', 'transaction_frequency']
        available_key_features = [f for f in key_features if f in numeric_features.columns]
        
        if len(available_key_features) > 1:
            corr_matrix = numeric_features[available_key_features].corr()
            im = ax7.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            ax7.set_xticks(range(len(available_key_features)))
            ax7.set_yticks(range(len(available_key_features)))
            ax7.set_xticklabels(available_key_features, rotation=45)
            ax7.set_yticklabels(available_key_features)
            ax7.set_title('Key Features Correlation')
            
            # اضافه کردن مقادیر
            for i in range(len(available_key_features)):
                for j in range(len(available_key_features)):
                    ax7.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                            ha='center', va='center', color='white')
        
        # 10. آمار کلی
        ax8 = fig.add_subplot(gs[2:4, 2:4])
        ax8.axis('off')
        
        # متن آمار
        stats_text = f"""
        Dataset Statistics:
        
        Total Users: {len(features_df):,}
        
        Key Metrics:
        """
        
        if 'total_amount' in features_df.columns:
            stats_text += f"• Avg Total Amount: {features_df['total_amount'].mean():,.0f}\n"
        
        if 'total_transactions' in features_df.columns:
            stats_text += f"• Avg Transactions: {features_df['total_transactions'].mean():.1f}\n"
        
        if 'age' in features_df.columns:
            stats_text += f"• Avg Age: {features_df['age'].mean():.1f}\n"
        
        if transactions_df is not None:
            stats_text += f"\nTransaction Data:\n"
            stats_text += f"• Total Transactions: {len(transactions_df):,}\n"
            
            if 'is_noise' in transactions_df.columns:
                noise_pct = transactions_df['is_noise'].mean() * 100
                stats_text += f"• Noise Percentage: {noise_pct:.2f}%\n"
        
        ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Banking Data Analysis Dashboard', fontsize=16, y=0.98)
        self._save_plot(fig, save_name)
        
        return fig
    
    def _save_plot(self, fig: plt.Figure, filename: str):
        """ذخیره نمودار در فایل"""
        if not filename.endswith('.png'):
            filename += '.png'
        
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved: {filepath}")
    
    def set_style(self, style: str = 'default'):
        """تنظیم استایل نمودارها"""
        if style == 'seaborn':
            sns.set_style("whitegrid")
        elif style == 'ggplot':
            plt.style.use('ggplot')
        elif style == 'dark':
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        
        logger.info(f"Plot style set to: {style}") 