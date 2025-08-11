#!/usr/bin/env python3
"""
Real-time Progress Monitor for Banking Data Analysis
Shows live progress bars for all pipeline stages
"""

import sys
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
import psutil
from tqdm import tqdm
import subprocess
import os

sys.path.append(str(Path(__file__).parent / "src"))
from src.database.sqlite_manager import SQLiteManager

class BankingProgressMonitor:
    """Real-time progress monitor with live progress bars"""
    
    def __init__(self):
        self.target_users = 100_000
        self.monitoring = True
        self.start_time = datetime.now()
        self.db_path = "database/banking_data.db"
        
        # Progress tracking
        self.last_users = 0
        self.last_transactions = 0
        self.last_features = 0
        
        # Progress bars
        self.pbar_users = None
        self.pbar_transactions = None
        self.pbar_features = None
        self.pbar_analysis = None
        
    def setup_progress_bars(self):
        """Initialize progress bars"""
        print("🚀 Banking Data Analysis Pipeline - Real-time Progress Monitor")
        print("="*80)
        
        # Main progress bars
        self.pbar_users = tqdm(
            total=self.target_users,
            desc="👥 Users Generated",
            unit="users",
            position=0,
            leave=True,
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        self.pbar_transactions = tqdm(
            total=0,  # Will update dynamically
            desc="💳 Transactions",
            unit="txns",
            position=1,
            leave=True,
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}, {rate_fmt}]"
        )
        
        self.pbar_features = tqdm(
            total=self.target_users,
            desc="🧬 Feature Extraction",
            unit="users",
            position=2,
            leave=True,
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        self.pbar_analysis = tqdm(
            total=4,  # Clustering, Anomaly Detection, Similarity, Report
            desc="📊 Analysis Tasks",
            unit="tasks",
            position=3,
            leave=True,
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
        )
        
    def get_database_stats(self):
        """Get current database statistics"""
        if not Path(self.db_path).exists():
            return 0, 0, 0
            
        try:
            db = SQLiteManager(self.db_path)
            
            # Get counts
            users_result = db.execute_query("SELECT COUNT(*) as count FROM users")
            users_count = users_result.iloc[0]['count'] if not users_result.empty else 0
            
            transactions_result = db.execute_query("SELECT COUNT(*) as count FROM transactions")
            transactions_count = transactions_result.iloc[0]['count'] if not transactions_result.empty else 0
            
            try:
                features_result = db.execute_query("SELECT COUNT(*) as count FROM user_features")
                features_count = features_result.iloc[0]['count'] if not features_result.empty else 0
            except:
                features_count = 0
                
            db.close()
            return users_count, transactions_count, features_count
            
        except Exception as e:
            return 0, 0, 0
    
    def update_progress_bars(self):
        """Update progress bars with current stats"""
        users, transactions, features = self.get_database_stats()
        
        # Update progress bars
        if self.pbar_users:
            self.pbar_users.n = users
            self.pbar_users.refresh()
            
        if self.pbar_transactions:
            self.pbar_transactions.n = transactions
            # Update total dynamically based on average
            if users > 0:
                avg_txns = transactions / users
                estimated_total = int(avg_txns * self.target_users)
                self.pbar_transactions.total = estimated_total
            self.pbar_transactions.refresh()
            
        if self.pbar_features:
            self.pbar_features.n = features
            self.pbar_features.refresh()
            
        # Calculate rates
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        
        if elapsed > 0:
            users_rate = users / elapsed
            txns_rate = transactions / elapsed
            
            # Update descriptions with rates
            self.pbar_users.set_description(f"👥 Users ({users_rate:.0f}/sec)")
            self.pbar_transactions.set_description(f"💳 Transactions ({txns_rate:.0f}/sec)")
            
        return users, transactions, features
    
    def show_system_stats(self):
        """Show system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        print(f"\n💻 System Stats:")
        print(f"   CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}% | Disk Free: {disk.free/(1024**3):.1f}GB")
    
    def check_analysis_progress(self):
        """Check analysis progress from output files"""
        analysis_count = 0
        
        # Check if clustering is done
        if Path("output/plots").exists() and any(Path("output/plots").glob("*cluster*")):
            analysis_count += 1
            
        # Check if anomaly detection is done
        if Path("output/plots").exists() and any(Path("output/plots").glob("*anomaly*")):
            analysis_count += 1
            
        # Check if similarity search is done
        if Path("output/plots").exists() and any(Path("output/plots").glob("*similarity*")):
            analysis_count += 1
            
        # Check if final report is done
        if Path("output/reports/ANALYSIS_REPORT.md").exists():
            analysis_count += 1
            
        if self.pbar_analysis:
            self.pbar_analysis.n = analysis_count
            self.pbar_analysis.refresh()
            
        return analysis_count
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.setup_progress_bars()
        
        consecutive_no_change = 0
        last_users = 0
        
        try:
            while self.monitoring:
                users, transactions, features = self.update_progress_bars()
                analysis_progress = self.check_analysis_progress()
                
                # Check if process is stalled
                if users == last_users:
                    consecutive_no_change += 1
                else:
                    consecutive_no_change = 0
                    last_users = users
                
                # Show periodic system stats
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    self.show_system_stats()
                
                # Check completion
                if users >= self.target_users and features >= self.target_users and analysis_progress >= 4:
                    print("\n🎉 Pipeline completed successfully!")
                    self.monitoring = False
                    break
                    
                # Check if stalled for too long
                if consecutive_no_change > 60:  # 1 minute without progress
                    print(f"\n⚠️  Warning: No progress detected for {consecutive_no_change} seconds")
                
                time.sleep(1)  # Update every second
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
            
        finally:
            self.cleanup_progress_bars()
    
    def cleanup_progress_bars(self):
        """Clean up progress bars"""
        for pbar in [self.pbar_users, self.pbar_transactions, self.pbar_features, self.pbar_analysis]:
            if pbar:
                pbar.close()
    
    def start_pipeline_and_monitor(self):
        """Start the data pipeline and monitor progress"""
        print("🚀 Starting Banking Data Analysis Pipeline...")
        print("📊 Target: 1,000,000 users with full analysis")
        print("⏱️  Estimated time: 20-40 minutes depending on system")
        print("\nPress Ctrl+C to stop monitoring\n")
        
        # Start the main pipeline in background
        pipeline_process = subprocess.Popen([
            sys.executable, "main.py", "--all"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print(f"📋 Pipeline process started (PID: {pipeline_process.pid})")
        
        # Start monitoring
        monitor_thread = threading.Thread(target=self.monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            # Wait for pipeline to complete
            stdout, stderr = pipeline_process.communicate()
            
            if pipeline_process.returncode == 0:
                print("\n✅ Pipeline completed successfully!")
            else:
                print(f"\n❌ Pipeline failed with exit code: {pipeline_process.returncode}")
                if stderr:
                    print(f"Error: {stderr}")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Stopping pipeline...")
            pipeline_process.terminate()
            
        finally:
            self.monitoring = False
            monitor_thread.join(timeout=2)

def main():
    """Main function"""
    monitor = BankingProgressMonitor()
    
    # Check if we should resume or start fresh
    users, transactions, features = monitor.get_database_stats()
    
    if users > 0:
        print(f"📊 Resuming from: {users:,} users, {transactions:,} transactions, {features:,} features")
        response = input("Continue from current progress? (y/n): ").lower().strip()
        if response != 'y':
            print("Starting fresh...")
            # Could add database cleanup here if needed
    
    monitor.start_pipeline_and_monitor()

if __name__ == "__main__":
    main() 